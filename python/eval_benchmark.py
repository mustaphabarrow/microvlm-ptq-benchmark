"""Offline benchmark: per-variant size, greedy caption accuracy, latency.

Replicates the exact Android workflow (tf.lite.Interpreter == the AAR's
runtime) so passing here means the same numbers will appear on-device.
"""
import os

import numpy as np
import tensorflow as tf

from common import (CAPTION_LEN, build_tokenizer_from_meta, VOCAB_FILE)
from synthetic import SHAPES, COLORS, SIZES, POSITIONS, SyntheticDataset
from train import META


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def run_once(interp, image, ids, mask):
    """Feed one sample and return the TFLite output."""
    inp = interp.get_input_details()
    out = interp.get_output_details()[0]
    inp_map = {"pixel_values": image, "input_ids": ids, "attention_mask": mask}
    for d in inp:
        arr = inp_map[d["name"]]
        if d["dtype"] == np.int8:
            scale, zp = d["quantization"]
            arr = np.round(arr / scale + zp).astype(np.int8)
        elif d["dtype"] == np.uint8:
            scale, zp = d["quantization"]
            arr = np.round(arr / scale + zp).astype(np.uint8)
        else:
            arr = arr.astype(d["dtype"])
        interp.set_tensor(d["index"], np.expand_dims(arr, 0))
    interp.invoke()
    return interp.get_tensor(out["index"])


def generate_tflite(interp, image, tokenizer):
    """Greedy autoregressive caption from a TFLite model."""
    real = [tokenizer.vocab["<bos>"]]
    for step in range(CAPTION_LEN - 1):
        ids = real + [tokenizer.vocab["<pad>"]] * (CAPTION_LEN - len(real))
        mask = [1] * len(real) + [0] * (CAPTION_LEN - len(real))
        logits = run_once(interp, image, np.asarray(ids, np.int64),
                          np.asarray(mask, np.int64))[0]
        nxt = int(np.argmax(logits[len(real) - 1]))
        if nxt in (tokenizer.vocab["<eos>"], tokenizer.vocab["<pad>"]):
            break
        real.append(nxt)
    return tokenizer.decode(real[1:])


def generate_model(fwd, image, tokenizer):
    """Greedy autoregressive caption from the SavedModel wrapper."""
    real = [tokenizer.vocab["<bos>"]]
    for step in range(CAPTION_LEN - 1):
        ids = real + [tokenizer.vocab["<pad>"]] * (CAPTION_LEN - len(real))
        mask = [1] * len(real) + [0] * (CAPTION_LEN - len(real))
        logits = fwd(np.expand_dims(image, 0), np.asarray([ids], np.int64),
                     np.asarray([mask], np.int64)).numpy()[0]
        nxt = int(np.argmax(logits[len(real) - 1]))
        if nxt in (tokenizer.vocab["<eos>"], tokenizer.vocab["<pad>"]):
            break
        real.append(nxt)
    return tokenizer.decode(real[1:])


def _decode_ids(pred_logits, mask, tokenizer):
    ids = np.argmax(pred_logits, -1)
    real = int(mask.sum())
    return tokenizer.decode(ids[:real])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _bleu(ref_words, hyp_words, n):
    if not hyp_words:
        return 0.0
    ref_ng, hyp_ng = {}, {}
    for i in range(len(ref_words) - n + 1):
        g = tuple(ref_words[i:i + n])
        ref_ng[g] = ref_ng.get(g, 0) + 1
    for i in range(len(hyp_words) - n + 1):
        g = tuple(hyp_words[i:i + n])
        hyp_ng[g] = hyp_ng.get(g, 0) + 1
    hits = sum(min(hyp_ng.get(g, 0), c) for g, c in ref_ng.items())
    total = max(sum(hyp_ng.values()), 1)
    prec = hits / total
    bp = min(1.0, np.exp(1 - len(ref_words) / max(len(hyp_words), 1)))
    return prec * bp


def evaluate_generation(predict_fn, tokenizer, images, texts, masks, n_eval):
    exact = word_acc = b1 = b2 = matched = 0
    for i in range(min(n_eval, len(images))):
        hyp = predict_fn(images[i])
        ref = texts[i]
        exact += int(hyp == ref)
        matched += int(hyp.split() == ref.split())
        rw = ref.split(); hw = hyp.split()
        if rw and hw:
            word_acc += np.mean([a == b for a, b in zip(rw, hw)])
            b1 += _bleu(rw, hw, 1)
            b2 += _bleu(rw, hw, 2)
    n = min(n_eval, len(images))
    return {"exact_match": exact / n, "word_acc": word_acc / n,
            "bleu1": b1 / n, "bleu2": b2 / n}


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def measure_latency(interp, image, tokenizer, repeat=100):
    """Median/fastest full forward (encode+decode step) latency in ms."""
    ids = np.asarray(tokenizer.encode("a large blue star at the right")[0], np.int64)
    mask = np.asarray(tokenizer.encode("a large blue star at the right")[1], np.int64)
    for _ in range(5):  # warm-up
        run_once(interp, image, ids, mask)
    times = []
    for _ in range(repeat):
        t0 = tf.timestamp().numpy() * 1e3
        run_once(interp, image, ids, mask)
        times.append(tf.timestamp().numpy() * 1e3 - t0)
    times = np.asarray(times)
    return {"lat_ms_median": float(np.median(times)),
            "lat_ms_mean": float(np.mean(times)),
            "lat_ms_p95": float(np.percentile(times, 95))}


def measure_generation_time(interp, image, tokenizer, repeat=10):
    times = []
    for _ in range(repeat):
        t0 = tf.timestamp().numpy() * 1e3
        generate_tflite(interp, image, tokenizer)
        times.append(tf.timestamp().numpy() * 1e3 - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# figures + csv
# ---------------------------------------------------------------------------
def save_figures(paths, report, out="artifacts/results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    rows = list(report.keys())
    labels = {"fp32": "FP32", "fp16": "FP16", "dyq_int8": "INT8 (weights)",
              "int8": "INT8 (static)"}

    sizes = [report[r]["size_kb"] for r in rows]
    em = [report[r]["metrics"]["exact_match"] for r in rows]
    wa = [report[r]["metrics"]["word_acc"] for r in rows]
    b1 = [report[r]["metrics"]["bleu1"] for r in rows]
    lat = [report[r]["latency"]["lat_ms_median"] for r in rows]

    # size
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    bars = ax.bar([labels[r] for r in rows], sizes, color=["#5b8ff9"] * len(rows))
    for b, v in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Model size (KB)")
    ax.set_title("Quantized model size")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_model_size.png")); plt.close(fig)

    # accuracy
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    x = np.arange(len(rows)); w = 0.27
    ax.bar(x - w, em, w, label="Exact match")
    ax.bar(x, wa, w, label="Word accuracy")
    ax.bar(x + w, b1, w, label="BLEU-1")
    ax.set_xticks(x); ax.set_xticklabels([labels[r] for r in rows])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.set_title("Caption quality vs quantization")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_accuracy.png")); plt.close(fig)

    # latency
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    bars = ax.bar([labels[r] for r in rows], lat, color=["#52c41a"] * len(rows))
    for b, v in zip(bars, lat):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Median latency (ms / forward)")
    ax.set_title("On-device (CPU) latency per quantized variant")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_latency.png")); plt.close(fig)

    # size-accuracy trade-off
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.scatter(sizes, [v * 100 for v in em], s=80, c="#fa8c16")
    for r, x_, y_ in zip(rows, sizes, em):
        ax.annotate(labels[r], (x_, y_ * 100), textcoords="offset points",
                    xytext=(8, -6), fontsize=9)
    ax.set_xlabel("Model size (KB)")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_title("Size vs accuracy trade-off of PTQ variants")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_tradeoff.png")); plt.close(fig)

    return out


def write_csv(paths, report, path="artifacts/results/benchmark_results.csv"):
    import csv
    rows = sorted(report.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "size_kb", "exact_match", "word_acc", "bleu1",
                    "bleu2", "lat_ms_median", "lat_ms_p95", "gen_ms_median",
                    "logit_mae", "token_agreement"])
        for r in rows:
            rec = report[r]
            w.writerow([r, f"{rec['size_kb']:.1f}", f"{rec['metrics']['exact_match']:.4f}",
                        f"{rec['metrics']['word_acc']:.4f}",
                        f"{rec['metrics']['bleu1']:.4f}", f"{rec['metrics']['bleu2']:.4f}",
                        f"{rec['latency']['lat_ms_median']:.3f}",
                        f"{rec['latency']['lat_ms_p95']:.3f}",
                        f"{rec.get('gen_ms_median', float('nan')):.3f}",
                        f"{rec.get('logit_mae', float('nan')):.4f}",
                        f"{rec.get('token_agreement', float('nan')):.3f}"])
    print(f"[benchmark] CSV -> {path}")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_benchmark(model, tokenizer, images, ids, masks, texts,
                  paths, n_eval=300, repeat=50, regen_repeat=10):
    report = {}
    ref = model(images[:8], ids[:8], masks[:8]).numpy() if hasattr(model, "numpy") else None

    for name, path in paths.items():
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        metrics = evaluate_generation(
            lambda im: generate_tflite(interp, im, tokenizer),
            tokenizer, images, texts, masks, n_eval)
        latency = measure_latency(interp, images[0], tokenizer, repeat=repeat)
        gen_ms = measure_generation_time(interp, images[0], tokenizer,
                                         repeat=regen_repeat)
        size_kb = os.path.getsize(path) / 1024
        report[name] = {"size_kb": size_kb, "metrics": metrics,
                        "latency": latency, "gen_ms_median": gen_ms}
        print(f"[benchmark] {name:<14} size={size_kb:7.1f} KiB "
              f"exact={metrics['exact_match']:.3f} word={metrics['word_acc']:.3f} "
              f"bleu1={metrics['bleu1']:.3f} lat={latency['lat_ms_median']:.1f}ms "
              f"gen={gen_ms:.1f}ms")

    # logit agreement (quantization error) for fp-ng comparisons
    for name, path in paths.items():
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        preds = []
        for i in range(8):
            preds.append(run_once(interp, images[i], ids[i], masks[i]))
        ref = model(images[:8], ids[:8], masks[:8]).numpy()
        preds = np.concatenate(preds, 0)
        report[name]["logit_mae"] = float(np.mean(np.abs(preds - ref)))
        report[name]["token_agreement"] = float(
            np.mean(np.argmax(preds, -1) == np.argmax(ref, -1)))
    return report


def main():
    tokenizer = build_tokenizer_from_meta(META)
    ds = SyntheticDataset(tokenizer, n_train=200, n_val=64)
    images, ids, masks, texts = ds.val()
    paths = {"fp32": "artifacts/microvlm_fp32.tflite",
             "fp16": "artifacts/microvlm_fp16.tflite",
             "dyq_int8": "artifacts/microvlm_dyq_int8.tflite",
             "int8": "artifacts/microvlm_int8.tflite"}
    from model import LoadedModel
    model = LoadedModel("artifacts/model")
    report = run_benchmark(model, tokenizer, images, ids, masks, texts, paths)
    write_csv(paths, report)
    save_figures(paths, report)
    return report


if __name__ == "__main__":
    main()
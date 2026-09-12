"""End-to-end pipeline: train/skip -> convert -> benchmark -> figures -> zip."""
import argparse
import glob
import json
import os
import re
import shutil
import zipfile

import numpy as np
import tensorflow as tf

from common import build_tokenizer_from_meta, CAPTION_LEN, IMG_SIZE
from model import LoadedModel, gather_trainable_variables, MicroVLM
from synthetic import (SHAPES, COLORS, SIZES, POSITIONS, SyntheticDataset,
                       save_sample_grid)
from train import META, evaluate


def parse_train_log(path="train.out.log"):
    """Best-epoch caption accuracy + loss curve from training log."""
    curve = []
    best = -1.0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = re.search(r"loss=([\d.]+)", line)
                if m:
                    curve.append(float(m.group(1)))
                a = re.search(r"caption_acc=([\d.]+) best=([\d.]+)", line)
                if a:
                    best = max(best, float(a.group(2)))
    return curve, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="use artifacts/model if it already exists")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    tokenizer = build_tokenizer_from_meta(META)
    tokenizer.save("artifacts/vocab.json")

    # ---- 1. train --------------------------------------------------------
    model_exists = os.path.exists("artifacts/model")
    if not model_exists or not args.skip_train:
        print("[pipeline] training not found, launch training...")
        raise SystemExit("Run: python train.py --epochs 80 --batch 128 --lr 0.002")

    # ---- 2. build dataset + sample fig ------------------------------------
    print("[pipeline] building dataset ...")
    ds = SyntheticDataset(tokenizer, n_train=6000, n_val=1200)
    images, ids, masks, texts = ds.val()
    tr_imgs = ds.images[:ds.n_train]
    save_sample_grid(ds, "artifacts/results/fig01_dataset_samples.png")

    # ---- 3. load trained model + convert ----------------------------------
    print("[pipeline] loading trained model ...")
    model = LoadedModel("artifacts/model")
    probe = model(np.expand_dims(images[0], 0),
                  np.expand_dims(ids[0], 0),
                  np.expand_dims(masks[0], 0))
    print(f"[pipeline] probe logits shape={probe.shape}")

    import eval_benchmark as eb
    from export_quant import convert_all
    paths = convert_all(model, images, ids, masks, out="artifacts")

    # ---- 4. benchmark ------------------------------------------------------
    print("[pipeline] benchmarking ...")
    report = eb.run_benchmark(model, tokenizer, images, ids, masks, texts,
                              paths, n_eval=args.n_eval, repeat=50, regen_repeat=10)
    eb.write_csv(paths, report)
    eb.save_figures(paths, report, out="artifacts/results")

    # training curve figure
    curve, best = parse_train_log()
    if curve:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        ax.plot(curve)
        ax.set_xlabel("logged step"); ax.set_ylabel("train loss")
        ax.set_title(f"Training loss curve (best caption acc = {best:.3f})")
        fig.tight_layout()
        fig.savefig("artifacts/results/fig02_train_curve.png")
        plt.close(fig)

    # ---- 5. android assets + package --------------------------------------
    pkg_android(paths, model, tokenizer, report)

    print("[pipeline] DONE -> artifacts/results/, artifacts/*.zip")


def run_lite_verify(model, paths, images, ids, masks, n=8):
    import numpy as np
    import tensorflow as tf
    from eval_benchmark import run_once
    ref = model(images[:n], ids[:n], masks[:n]).numpy()
    out = {}
    for name, path in paths.items():
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        preds = [run_once(interp, images[i], ids[i], masks[i]) for i in range(n)]
        preds = np.concatenate(preds, 0)
        out[name] = {"mae": float(np.mean(np.abs(preds - ref))),
                     "token_agreement": float(
                         np.mean(np.argmax(preds, -1) == np.argmax(ref, -1)))}
    return out


def pkg_android(paths, model, tokenizer, report):
    asset_dir = os.path.join("..", "android", "OnDeviceVLM",
                             "app", "src", "main", "assets")
    os.makedirs(asset_dir, exist_ok=True)

    variant_list = [{"name": k, "file": os.path.basename(v),
                     "size_kb": round(report[k]["size_kb"], 1)} for k, v in paths.items()]
    config = {
        "img_size": IMG_SIZE, "caption_len": CAPTION_LEN,
        "variants": variant_list,
        "preprocess": "divide_by_255",
        "lighting": "synthetic",
    }
    with open(os.path.join(asset_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy("artifacts/vocab.json",
                os.path.join(asset_dir, "vocab.json"))
    for name, path in paths.items():
        shutil.copy(path, os.path.join(asset_dir, os.path.basename(path)))
    print(f"[pipeline] android assets -> {asset_dir}")

    # zip
    zip_path = "artifacts/microvlm_pipeline.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob("artifacts/*.tflite") + glob.glob("artifacts/*.json") \
                + glob.glob("artifacts/results/*.png") \
                + glob.glob("artifacts/results/*.csv"):
            z.write(f, os.path.relpath(f, "artifacts"))
    print(f"[pipeline] package -> {zip_path}")


if __name__ == "__main__":
    main()
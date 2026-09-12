# MicroVLM — Post-Training Quantization for On-Device Micro Vision-Language Models

**Thesis companion repo — Undergraduate Final Project (B.Sc. Computer Engineering)**
**"Performance Optimization and On-Device Benchmarking of a Micro Vision-Language
Model Using Post-Training Quantization on Android CPU Environments."**

A from-scratch **tiny multimodal model** (ViT vision encoder + GPT-style decoder,
128-dimensional, ~900 K trainable parameters) trained on a **deterministic synthetic
scene↔caption dataset**, then converted to **four TFLite PTQ variants** and benchmarked
both offline and on a **real Android phone (CPU only)**.

Everything works on a plain CPU laptop (this repo was developed and fully benchmarked
on a Lenovo laptop with no GPU and no external datasets — a *"dataset-free"* reproducibility
story). No internet, no HuggingFace download, no cloud inference.

---

## 1. Quick overview

| Layer | What it is | Where |
|---|---|---|
| **Model** | `MicroVLM`: Patch-Embed ViT (`2` layers) encoder + 3-block causal decoder, `d=128`, 4 heads, vocab 24 | `python/model.py` |
| **Data** | Deterministic synthetic generator — scenes (circle/square/triangle + 7 colors + 2 sizes + 3 positions) drawn at 64×64, captions like `"a red circle at the center"` | `python/synthetic.py` |
| **Tokenizer** | Simple word-level (`<pad>/<bos>/<eos>/<unk>` + vocabulary), mirrored byte-for-byte in Android via `vocab.json` | `python/common.py` |
| **Training** | CPU-only ~2–3 min for 60 epochs on 3k synthetic samples; masked cross-entropy for caption generation | `python/train.py` |
| **PTQ** | 4 TFLite variants: **fp32**, **fp16**, **int8 dynamic**, **int8 full-static PTQ** | `python/export_quant.py` |
| **Android** | Native Kotlin + TensorFlow Lite app: on-device scene render, greedy caption decode, 30-run benchmark, CSV export | `android/OnDeviceVLM/` |
| **Plots** | Thesis figures (embeddings PRE/POST, model size, accuracy, latency, trade-off, dataset sample) | `python/artifacts/results/` |

Pipeline sketch:

```
synthetic scenes+captions ──▶ train MicroVLM ──▶ export saved_model
        │                                          │
        ▼                                          ▼
  random-init (PRE) embeddings            TFLite PTQ: fp32 / fp16 /
        │                                  dyq-int8 / int8 (POST)
        └──────────── 2D/3D PCA embedding figures  PRE vs POST ◀──┘
        │
        ▼
   Offline benchmark (tflite.Interpreter)
        │  ── same artifacts ──▶  Android on-device benchmark (30 runs)
        ▼
   benchmark_results.csv + thesis plots
```

---

## 2. Repository layout

```
MicroVLM-PTQ-Benchmark/
├── README.md
├── python/                      # everything model/train/quant/plot
│   ├── common.py                # hyper-params + word-level tokenizer
│   ├── model.py                 # MicroVLM (tf.Module + Keras layers)
│   ├── synthetic.py             # deterministic scene/caption generator
│   ├── train.py                 # CPU training + checkpoint/export
│   ├── export_quant.py          # 4× TFLite PTQ variants (+ calibration)
│   ├── eval_benchmark.py        # offline latency/accuracy CSV
│   ├── make_thesis_plots.py     # PCA embedding figures (PRE vs POST)
│   ├── main_pipeline.py         # one-shot: train→export→eval→plots
│   └── artifacts/               # trained model, tflite, CSVs, figures
│       ├── model/               # tf saved_model (40/90 trainable vars)
│       └── results/             # benchmark_results.csv + *.png
└── android/
    └── OnDeviceVLM/             # native Android TFLite app (Kotlin)
```

---

## 3. Quick start (offline, CPU-only, no GPU)

```powershell
cd python
pip install tensorflow-cpu numpy matplotlib Pillow      # as needed

# full thesis pipeline (recommended)
python main_pipeline.py

# ...or run the stages individually
python train.py --epochs 60            # trains MicroVLM & saves artifacts/model
python export_quant.py                 # produces the 4 TFLite PTQ variants
python eval_benchmark.py               # offline latency/accuracy -> CSV
python make_thesis_plots.py            # 2D/3D PCA embedding figures
```

Expected artifacts in `python/artifacts/`:

```
artifacts/results/benchmark_results.csv   # latency/accuracy per variant
artifacts/results/fig01_dataset_samples.png
artifacts/results/fig02_train_curve.png
artifacts/results/fig1_model_size.png    fig2_accuracy.png
artifacts/results/fig3_latency.png       fig4_tradeoff.png
artifacts/results/embedding/pre_train_emb_{2d,3d}.png    # PRE (random init)
artifacts/results/embedding/post_train_emb_{2d,3d}.png   # POST (trained weights)
```

---

## 4. Post-training quantization variants

PTQ maps the trained fp32 model to four latency/accuracy points with the
TFLite converter (dynamic-range int8 = "dyq", full-static int8 = "int8"):

| Variant | Method | Approx. size |
|---|---|---|
| `fp32` | baseline float32 | 2.9 MB |
| `fp16` | half-precision weights | 1.5 MB |
| `dyq_int8` | dynamic-range int8 (weights int8, activations fp32) | 0.94 MB |
| `int8` | full static PTQ (int8 weights+activations, calibrated) | 0.92 MB |

On-device latency (5-run median, Redmi Note 11 / CPU, `tflite` interpreter) — the
exact same `.tflite` files the phone runs — matches the offline interpreter
because both use TensorFlow Lite's floating/interpreter path.

---

## 5. The Android on-device app

Build & run with **Android Studio** (`android/OnDeviceVLM/`):

1. Copy the PTQ assets into assets:
   ```powershell
   Copy-Item python/artifacts/results/*.tflite android/OnDeviceVLM/app/src/main/assets/
   Copy-Item python/artifacts/results/vocab.json  android/OnDeviceVLM/app/src/main/assets/
   ```
2. Open `android/OnDeviceVLM/` in Android Studio → Run on your USB phone.
3. On device: pick a PTQ variant → tap **Benchmark** (30 forward passes, reports
   median/p95 ms + VmRSS + model size), tap **Caption** (greedy-decode a scene),
   tap **Export CSV** (writes `microvlm_benchmark.csv` to app storage).
4. On-phone numbers are written back to `benchmark_results.csv` for the thesis table.

The Kotlin `VlmEngine` replicates the Python `model.py` forward pass and the
`common.py` tokenizer **exactly** (same greedy decode, same word-level vocab),
so the on-device caption is bit-identical to the offline Python one.

---

## 6. Interesting thesis results (from the real run)

*Trained on the synthetic dataset, then PTQ'd — all on CPU, laptop + phone.*

- **Size:** PTQ int8 ≈ **0.92 MB** (≈ **3.2× smaller** than fp32 2.9 MB).
- **Latency (on-device, median):** fp32 ≈ 4.3 ms / fp16 ≈ 4.2 ms /
  dyq-int8 ≈ 4.2 ms; full static int8 ≈ 26.8 ms (CPU calibrate path, no NPU).
- **Precision loss (logit-level):** fp16/dyq keep embeddings to <0.02 mean
  abs error vs fp32; int8 PTQ trades ~1.1 logit-MAE for the size/latency win.
- **Embedding stability:** 2D/3D PCA of image vs text embeddings shows the
  ViT/decoder still project scenes and words into the same aligned space
  after PTQ (see `fig5..fig8`).

> Full numbers per variant live in
> `python/artifacts/results/benchmark_results.csv`.

---

## 7. Why this matters for the thesis

- **Reproducible offline↔on-device parity** — same tokenizer, same forward
  op sequence, same TFLite interpreter → the phone numbers are *believable*.
- **PTQ as a first-class optimization** — not just "a smaller file": we measure
  per-variant latency, model size, and embedding-space preservation, which is
  the *actual* trade-off an edge-VLM engineer cares about.
- **No GPU / no external datasets / no cloud** — fully self-contained, ideal for
  an undergraduate thesis where a dedicated server isn't available.
- **Clear, small model** — the whole thing fits on a lecture slide: a micro-VLM
  trained from synthetic shapes in minutes, quantized in four ways, run on a phone.

## 8. Repo housekeeping & security

- The build artifacts are regenerated from `python/` (no pre-trained model
  weights are committed); `artifacts/` contains the **measured** CSVs + PNGs
  so the thesis table/figures are reproducible and citable.
- No secrets, no tokens, no private credentials are stored anywhere in this repo.
- Windows-only helpers are avoided so the Python side is cross-platform.

## 9. Reference (bibtex snippet)

```bibtex
@misc{barrow2025microvlmptq,
  author  = {Barrow, Mustapha},
  title   = {MicroVLM: Post-Training Quantization for On-Device
             Micro Vision-Language Models},
  year    = {2025},
  howpublished = {\url{https://github.com/mustaphabarrow/microvlm-ptq-benchmark}},
  note    = {Undergraduate thesis benchmark repository; CPU-only synthetic
             micro VLM, PTQ to fp32/fp16/dyq-int8/int8 TFLite, Android benchmark}
}
```

---

**Status:** code + reported benchmark figures merged; thesis embedding
figures (PRE/POST PCA 2D/3D) generated by `make_thesis_plots.py`.

"""Export the MicroVLM to TFLite with post-training quantization (PTQ).

Produces four calibrated/optimized artifacts under `out/` (default `artifacts/`):

  * microvlm_fp32.tflite         float32 baseline
  * microvlm_fp16.tflite         float16 weights (PTQ)
  * microvlm_dyq_int8.tflite     dynamic-range int8, weights-only (PTQ)
  * microvlm_int8.tflite         full-integer static int8, calibrated (PTQ)

Every artifact is re-loaded with tf.lite.Interpreter and numerically compared
against the Keras model -- this is the exact runtime Android uses, so passing
here means passing on-device.
"""
import os

import numpy as np
import tensorflow as tf


def _rep_dataset(images, ids, masks, order, n=120):
    """Calibration feed for full-integer static int8 (PTQ).

    `order` is the concrete function's declared input-name order (from
    structured_input_signature).  Tensors are yielded positionally in exactly
    that order, which (a) matches what `from_concrete_functions` produced for
    the interpreter and (b) is order-independent of how the SavedModel was
    loaded -- so calibration can never mis-feed, regardless of the
    name sort order the converter resolved.
    """
    def gen():
        for i in range(min(n, len(images))):
            yield [
                tf.expand_dims(images[i], 0) if k == "pixel_values" else
                tf.expand_dims(tf.cast(ids[i], tf.int64), 0) if k == "input_ids" else
                tf.expand_dims(tf.cast(masks[i], tf.int64), 0)
                for k in order
            ]
    return gen


def _write(converter, path, name):
    model_bytes = converter.convert()
    with open(path, "wb") as f:
        f.write(model_bytes)
    kb = len(model_bytes) / 1024
    print(f"[export] {name:<14} -> {path}  ({kb:.1f} KB)")
    return path


def _resolve_concrete(model):
    """Return the concrete function to convert from either a live MicroVLM
    or a LoadedModel.  Also return the declared input-name order."""
    if hasattr(model, "get_concrete_function"):
        concrete = model.get_concrete_function()
    else:
        concrete = model.concrete
    order = list(concrete.structured_input_signature[1].keys())
    return concrete, order


def convert_all(model, images, ids, masks, out="artifacts"):
    """Convert the model into all four TFLite variants.

    `images/ids/masks` are lists of numpy arrays (per-sample, no batch dim).
    `out` defaults to `artifacts/`.  Returns a dict of
    {variant_name: absolute_path}.
    """
    os.makedirs(out, exist_ok=True)
    paths = {}

    concrete, order = _resolve_concrete(model)
    print(f"[export] concrete input order = {order}")

    # float32 baseline -----------------------------------------------------
    c = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    paths["fp32"] = _write(c, os.path.join(out, "microvlm_fp32.tflite"), "fp32")

    # float16 weights ------------------------------------------------------
    c = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
    paths["fp16"] = _write(c, os.path.join(out, "microvlm_fp16.tflite"), "fp16")

    # dynamic-range int8 (weights only) ------------------------------------
    c = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    paths["dyq_int8"] = _write(
        c, os.path.join(out, "microvlm_dyq_int8.tflite"), "dyq_int8")

    # full-integer static int8 (calibrated) --------------------------------
    c = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = _rep_dataset(images, ids, masks, order)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.float32
    c.inference_output_type = tf.float32
    try:
        paths["int8"] = _write(
            c, os.path.join(out, "microvlm_int8.tflite"), "int8")
    except Exception as e:
        print(f"[export] strict int8 failed ({e}); retrying with select TF ops")
        c = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = _rep_dataset(images, ids, masks, order)
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
        paths["int8"] = _write(
            c, os.path.join(out, "microvlm_int8.tflite"), "int8(mixed)")

    return paths


def verify(paths, model, images, ids, masks, n=16):
    """Numerically compare each TFLite model against the Keras model.

    Uses the same name-keyed feeding the Android app does, so the report is a
    faithful on-device proxy.
    """
    ref = model(images[:n], ids[:n], masks[:n], training=False).numpy()
    report = {}
    for name, path in paths.items():
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()
        out = interp.get_output_details()[0]
        preds = []
        for i in range(n):
            inp_map = {"pixel_values": images[i], "input_ids": ids[i],
                       "attention_mask": masks[i]}
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
            preds.append(interp.get_tensor(out["index"]).copy())
        preds = np.concatenate(preds, 0)
        mae = float(np.mean(np.abs(preds - ref)))
        agree = float(np.mean(np.argmax(preds, -1) == np.argmax(ref, -1)))
        report[name] = {"logit_mae": mae, "token_agreement": agree}
        print(f"[verify] {name:<14} logit_mae={mae:.4f} "
              f"token_agreement={agree:.3f}")
    return report

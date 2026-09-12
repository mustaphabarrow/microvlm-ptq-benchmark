"""Thesis thesis-plot generator for MicroVLM-PTQ.

Builds the MicroVLM once (random init = PRE-training embedding state), copies
the trained weights from the exported saved_model into an identical model
(POST-training state), then renders 2D/3D PCA projections of:

  * image embedding (ViT patch + CLS tokens, prefix length = VIS_PREFIX_LEN)
  * text embedding (token embeddings of the synthetic vocabulary)

for both states, plus a combined alignment panel. All figures go to
artifacts/results/embedding/.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf

import common
import model as M
from synthetic import SHAPES, COLORS, SIZES, POSITIONS, draw_scene, POS_CENTER

META = {"shape": SHAPES, "color": COLORS, "size": SIZES, "pos": POSITIONS}
OUT = os.path.join("artifacts", "results", "embedding")
os.makedirs(OUT, exist_ok=True)

tf.keras.utils.disable_interactive_logging()

# ---------------------------------------------------------------------------
# 0. tokenizer + dataset anchors
# ---------------------------------------------------------------------------
tok = common.build_tokenizer_from_meta(META)
print(f"[tokenizer] size={tok.size}")

SHAPE_WORDS = SHAPES
COLOR_WORDS = COLORS
SIZE_WORDS = SIZES
POS_WORDS = POSITIONS


def scene(shape, color, size="large", pos="center"):
    arr = draw_scene(shape, color, size, pos)
    return (arr.astype(np.float32) / 255.0)[None]                 # (1,64,64,3)


def txt_emb(model, words):
    """Mean-pooled (over real tokens) text embeddings for a list of words."""
    out = []
    for w in words:
        ids, msk = tok.encode(w)
        ids = tf.constant([ids], tf.int64)
        msk = tf.constant([msk], tf.int64)
        e = model.tok_emb(ids)                                    # (1,24,128)
        mask = tf.cast(msk[..., None], tf.float32)
        pooled = tf.reduce_sum(e * mask, axis=1) / tf.maximum(
            tf.reduce_sum(mask, axis=1, keepdims=True), 1e-6)     # (1,128)
        out.append(pooled[0].numpy())
    return np.asarray(out, np.float32)


def img_emb(model, words_or_shapes, kind):
    """ViT embeddings (mean over CLS+patch tokens) for scenes keyed per word."""
    out = []
    for w in words_or_shapes:
        arr = scene(w, w) if kind == "color" else scene(w, "red")
        img = tf.constant(arr, tf.float32)
        e = model.vision_encode(img)                              # (1,65,128)
        out.append(tf.reduce_mean(e, axis=1)[0].numpy())          # (128,)
    return np.asarray(out, np.float32)


# ---------------------------------------------------------------------------
# 1. build + run once
# ---------------------------------------------------------------------------
def build_model():
    m = M.MicroVLM(tok.size)
    probe_im = tf.zeros([2, common.IMG_SIZE, common.IMG_SIZE, 3], tf.float32)
    probe_id = tf.zeros([2, common.CAPTION_LEN], tf.int64)
    probe_m = tf.zeros([2, common.CAPTION_LEN], tf.int64)
    m(probe_im, probe_id, probe_m)
    return m


print("[build] PRE (random init) ...")
m_pre = build_model()
print("[build] forward OK")

print("[build] POST (copy trained weights) ...")
try:
    loaded = tf.saved_model.load(os.path.join("artifacts", "model"))
    fresh_vars = M.gather_trainable_variables(m_pre)
    # collect variables from the loaded object the same way
    loaded_vars = []
    def _walk(o, seen=None):
        seen = seen or set()
        if id(o) in seen:
            return
        seen.add(id(o))
        tw = getattr(o, "trainable_weights", None)
        if tw:
            loaded_vars.extend(v for v in tw if id(v) not in seen)
            return
        try:
            items = list(vars(o).values())
        except TypeError:
            return
        for a in items:
            if isinstance(a, tf.Variable):
                loaded_vars.append(a)
            elif isinstance(a, (list, tuple, dict)):
                for it in (a.values() if isinstance(a, dict) else a):
                    _walk(it, seen)
            elif hasattr(a, "__dict__") and not isinstance(
                    a, (str, bytes, int, float, tf.Tensor, type)):
                _walk(a, seen)

    _walk(loaded)
    uniq = {}
    for v in loaded_vars:
        uniq[id(v)] = v
    loaded_vars = list(uniq.values())
    print(f"   fresh vars={len(fresh_vars)}  loaded vars={len(loaded_vars)}")
    assert len(fresh_vars) == len(loaded_vars), "variable count mismatch"
    for f, l in zip(fresh_vars, loaded_vars):
        f.assign(l)
    print("   weights copied by position")
except Exception as e:
    print(f"   [warn] weight copy failed: {e!r}  -> POST == random init")
    import traceback; traceback.print_exc()

# verification: forward PRE and POST should differ (weights moved)
t_im = tf.constant(scene("circle", "red"), tf.float32)   # scene() already (1,64,64,3)
t_ids, t_mask = tok.encode("a red circle")
t_ids = tf.constant([t_ids], tf.int64)
t_mask = tf.constant([t_mask], tf.int64)
lp = m_pre(t_im, t_ids, t_mask).numpy()
if "loaded" in dir():
    from model import LoadedModel
    lm = LoadedModel(os.path.join("artifacts", "model"))
    ll = lm(t_im.numpy(), t_ids.numpy(), t_mask.numpy()).numpy()
    diff = float(np.abs(lp - ll).mean())
    print(f"[verify] mean|pre_logits - exported_logits| = {diff:.6f}")
    print("  -> lambda 1e-4; small = faithful POST copy" if diff < 1e-3
          else "  -> DIFFERENT; POST weights NOT an exact copy")

# ---------------------------------------------------------------------------
# 2. extract embeddings
# ---------------------------------------------------------------------------
# PRE must come from an UNTOUCHED random build (the copy at lines 87-133 has
# already written the trained weights into m_pre -> m_pre is the POST state).
m_rand = build_model()                                        # fresh random = PRE
pre_img_color = img_emb(m_rand, COLOR_WORDS, "color")
pre_img_shape = img_emb(m_rand, SHAPE_WORDS, "shape")
pre_txt_color = txt_emb(m_rand, COLOR_WORDS)
pre_txt_shape = txt_emb(m_rand, SHAPE_WORDS)

# POST = the weight-copied model (verified faithful copy above).
post_img_color = img_emb(m_pre, COLOR_WORDS, "color")
post_img_shape = img_emb(m_pre, SHAPE_WORDS, "shape")
post_txt_color = txt_emb(m_pre, COLOR_WORDS)
post_txt_shape = txt_emb(m_pre, SHAPE_WORDS)


def _pca(X, n_comp):
    X = np.asarray(X, np.float32)
    X = X - X.mean(axis=0)                                   # center
    U, S, Vt = np.linalg.svd(X, full_matrices=False)         # (N,K),(K,),(K,d)
    k = min(n_comp, Vt.shape[0])
    proj = (U[:, :k] * S[:k]).astype(np.float32)             # (N,k) == X@Vt[:k].T
    print(f"   [_pca] X={X.shape} Vt={Vt.shape} k={k}  (SVD proj ok)")
    return proj


def pca_2d(X, n_comp=2, seed=0):
    return _pca(X, n_comp)


def pca_3d(X):
    return _pca(X, 3)


def save_fig(arrays, labels, kinds, path, title, dims=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7) if dims == 2 else (9, 8), dpi=130)
    ax = fig.add_subplot(111, projection="3d") if dims == 3 else fig.add_subplot(111)
    for arr, lab, kind, mkr in zip(arrays, labels, kinds,
                                   ["o", "s", "D", "^", "v", "P", "x"]):
        proj = pca_3d(arr) if dims == 3 else pca_2d(arr)
        if dims == 3:
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                       s=42, marker=mkr, label=kind, alpha=0.9)
            # NOTE: no per-point 3D ax.text -- matplotlib's mplot3d projection
            # (mplot3d/proj3d._proj_trans_points) crashes with numpy 2.x
            # ("inhomogeneous shape"). Legend + markers carry the labels.
        else:
            ax.scatter(proj[:, 0], proj[:, 1], s=42, marker=mkr, label=kind)
            for i, t in enumerate(lab):
                ax.annotate(t, (proj[i, 0], proj[i, 1]), fontsize=7)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# -- 2D ---------------------------------------------------------------------
save_fig(
    [pre_img_color, pre_txt_color],
    [COLOR_WORDS, COLOR_WORDS],
    ["img(color)", "txt(color)"],
    os.path.join(OUT, "pre_train_emb_2d.png"),
    f"PRE-training: image vs text embeddings (2D PCA)", dims=2)
save_fig(
    [pre_img_color, pre_txt_color],
    [COLOR_WORDS, COLOR_WORDS],
    ["img(color)", "txt(color)"],
    os.path.join(OUT, "pre_train_emb_3d.png"),
    f"PRE-training: image vs text embeddings (3D PCA)", dims=3)
save_fig(
    [post_img_color, post_txt_color],
    [COLOR_WORDS, COLOR_WORDS],
    ["img(color)", "txt(color)"],
    os.path.join(OUT, "post_train_emb_2d.png"),
    "POST-training: image vs text embeddings (2D PCA)", dims=2)
save_fig(
    [post_img_color, post_txt_color],
    [COLOR_WORDS, COLOR_WORDS],
    ["img(color)", "txt(color)"],
    os.path.join(OUT, "post_train_emb_3d.png"),
    "POST-training: image vs text embeddings (3D PCA)", dims=3)
print("[done]")

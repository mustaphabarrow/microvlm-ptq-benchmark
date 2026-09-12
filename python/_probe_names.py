import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
import common
import model as M
from synthetic import SHAPES, COLORS, SIZES, POSITIONS

def log(*a):
    print(*a, flush=True)

META = {"shape": SHAPES, "color": COLORS, "size": SIZES, "pos": POSITIONS}
tok = common.build_tokenizer_from_meta(META)
log("tokenizer size:", tok.size)

log("building fresh MicroVLM ...")
m_fresh = M.MicroVLM(tok.size)
log("built OK")

probe = tf.zeros([2, common.IMG_SIZE, common.IMG_SIZE, 3], tf.float32)
ids = tf.zeros([2, common.CAPTION_LEN], tf.int64)
msk = tf.ones([2, common.CAPTION_LEN], tf.int64)
log("running forward on fresh ...")
m_fresh(probe, ids, msk)
log("forward OK")

log("loading saved_model ...")
loaded = tf.saved_model.load("artifacts/model")
log("loaded OK")

fv = [v.name for v in m_fresh.variables]
lv = [v.name for v in loaded.variables]
log("=== counts fresh=%d loaded=%d" % (len(fv), len(lv)))
n_same = sum(1 for a, b in zip(fv, lv) if a == b)
log("=== positionally matching names: %d/%d" % (n_same, min(len(fv), len(lv))))
for i, (a, b) in enumerate(zip(fv, lv)):
    if i >= 150:
        break
    flag = "ok " if a == b else "DIFF"
    log("  [%03d] %s  f=%s\n            l=%s" % (i, flag, a, b))
log("=== loaded callables ===")
log("vision_encode:", hasattr(loaded, "vision_encode"))
log("__call__:", hasattr(loaded, "__call__"))
log("signatures:", sorted(getattr(loaded, "signatures", {}).keys()))

"""Fast de-risk: build + train 1 step + convert all TFLite variants."""
import numpy as np
import tensorflow as tf

from common import build_tokenizer_from_meta, CAPTION_LEN, IMG_SIZE
from model import MicroVLM
from synthetic import SHAPES, COLORS, SIZES, POSITIONS
from train import META
import export_quant as eq

tokenizer = build_tokenizer_from_meta(META)
model = MicroVLM(tokenizer.size)

img = np.random.rand(2, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
ids, msk = zip(*[tokenizer.encode("a red circle at the left") for _ in range(2)])
ids = np.asarray(ids, np.int64); msk = np.asarray(msk, np.int64)

logits = model(img, ids, msk)
print("forward logits:", logits.shape)

opt = tf.keras.optimizers.Adam(1e-3)
with tf.GradientTape() as t:
    l = model.masked_loss(ids, model(img, ids, msk), msk)
g = t.gradient(l, model.trainable_variables)
opt.apply_gradients([(gi, w) for gi, w in zip(g, model.trainable_variables) if gi is not None])
print("one train step OK")

paths = eq.convert_all(model, img, ids, msk, out="artifacts_smoke")
eq.verify(paths, model, img, ids, msk, n=2)
print("SMOKE TEST PASSED")
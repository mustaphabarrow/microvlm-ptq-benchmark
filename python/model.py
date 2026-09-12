"""Micro vision-language model using tf.Module + Keras layers.

Avoids Keras 3 functional API issues with TFLite conversion.
All layers are standard Keras layers; the outer wrapper is tf.Module
with a @tf.function call for clean concrete-function tracing.
"""
import numpy as np
import tensorflow as tf

from common import (IMG_SIZE, PATCH, NUM_PATCHES, VIS_PREFIX_LEN,
                    CAPTION_LEN, D_MODEL, VIT_DEPTH, DECODER_DEPTH, N_HEADS)


def _sinusoid_table(seq_len, dim):
    pos = np.arange(seq_len)[:, None]
    k = np.arange(dim)[None, :]
    theta = pos / np.power(10000.0, (2 * (k // 2)) / dim)
    return np.where(k % 2 == 0, np.sin(theta), np.cos(theta)).astype(np.float32)


class PreNormBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, causal=False, **kw):
        super().__init__(**kw)
        self._dim, self._heads, self._causal = dim, heads, causal

    def build(self, input_shape):
        d = self._dim
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self._heads, key_dim=d // self._heads)
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.ff1 = tf.keras.layers.Dense(d * 2, activation="gelu")
        self.ff2 = tf.keras.layers.Dense(d)

    def call(self, x):
        h = self.ln1(x)
        h = self.mha(h, h, use_causal_mask=self._causal)
        x = x + h
        x = x + self.ff2(self.ff1(self.ln2(x)))
        return x

    def get_config(self):
        return {"dim": self._dim, "heads": self._heads, "causal": self._causal}


def gather_trainable_variables(obj):
    """Recursively collect every trainable tf.Variable under any object."""
    out, seen, seen_obj = [], set(), set()

    def add(vs):
        for v in vs:
            if id(v) not in seen and getattr(v, "trainable", True):
                seen.add(id(v))
                out.append(v)

    def walk(o):
        if id(o) in seen_obj:
            return
        seen_obj.add(id(o))
        tw = getattr(o, "trainable_weights", None)
        if tw:
            add(tw)
            return
        try:
            items = list(vars(o).values())
        except TypeError:
            return
        for a in items:
            if isinstance(a, tf.Variable):
                add([a])
            elif isinstance(a, (list, tuple, dict)):
                for it in (a.values() if isinstance(a, dict) else a):
                    walk(it)
            elif hasattr(a, "__dict__") and not isinstance(
                    a, (str, bytes, int, float, tf.Tensor, type)):
                walk(a)

    walk(obj)
    return out


class MicroVLM(tf.Module):
    def __init__(self, vocab_size, name="micro_vlm"):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        d = D_MODEL

        # vision
        self.patch_embed = tf.keras.layers.Conv2D(
            d, PATCH, strides=PATCH, padding="valid", name="patch_embed")
        self.vis_cls = tf.Variable(
            tf.random.truncated_normal([1, 1, d], stddev=0.02), name="vis_cls")
        self.vis_norm = tf.keras.layers.LayerNormalization(axis=-1, name="vis_norm")
        self.vit_0 = PreNormBlock(d, N_HEADS, causal=False, name="vit_block_0")
        self.vit_1 = PreNormBlock(d, N_HEADS, causal=False, name="vit_block_1")

        # text
        self.tok_emb = tf.keras.layers.Embedding(vocab_size, d, name="tok_emb")

        # decoder
        self.vis_proj = tf.keras.layers.Dense(d, name="vis_proj")
        self.dec_0 = PreNormBlock(d, N_HEADS, causal=True, name="dec_block_0")
        self.dec_1 = PreNormBlock(d, N_HEADS, causal=True, name="dec_block_1")
        self.dec_2 = PreNormBlock(d, N_HEADS, causal=True, name="dec_block_2")
        self.lm_head = tf.keras.layers.Dense(vocab_size, name="lm_head")

        # fixed sinusoidal positional tables (non-trainable)
        S = VIS_PREFIX_LEN + CAPTION_LEN
        self._vis_pos = tf.constant(_sinusoid_table(NUM_PATCHES, d))
        self._seq_pos = tf.constant(_sinusoid_table(S, d))

    def vision_encode(self, x):
        x = self.patch_embed(x)                                          # (B,8,8,d)
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, NUM_PATCHES, -1])                         # (B,64,d)
        x = x + self._vis_pos
        cls = tf.tile(self.vis_cls, [B, 1, 1])                           # (B,1,d)
        x = tf.concat([cls, x], axis=1)                                  # (B,65,d)
        x = self.vis_norm(x)                     # (B,65,d)
        for block in (self.vit_0, self.vit_1):
            x = block(x)
        return self.vis_proj(x)                   # (B,65,d)

    def __call__(self, pixel_values, input_ids, attention_mask, training=False):
        vis = self.vision_encode(pixel_values)                           # (B,65,d)
        emb = self.tok_emb(input_ids)                                    # (B,24,d)
        emb = emb * tf.cast(attention_mask[..., None], tf.float32)
        seq = tf.concat([vis, emb], axis=1)                              # (B,89,d)
        seq = seq + self._seq_pos
        for block in (self.dec_0, self.dec_1, self.dec_2):
            seq = block(seq)
        logits = self.lm_head(seq)                                       # (B,89,V)
        return logits[:, VIS_PREFIX_LEN:, :] * tf.cast(
            attention_mask[..., None], tf.float32)

    def masked_loss(self, target_ids, logits, mask):
        """Autoregressive CE with shifted targets: logits[t] predicts id[t+1]."""
        logits_p = logits[:, :-1, :]                      # (B, L-1, V)
        tgt = target_ids[:, 1:]                           # (B, L-1)
        m = tf.cast(mask[:, 1:], tf.float32)              # (B, L-1)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tgt, logits_p, from_logits=True)
        return tf.reduce_sum(loss * m) / tf.maximum(tf.reduce_sum(m), 1e-6)

    def get_concrete_function(self):
        @tf.function
        def forward(pixel_values, input_ids, attention_mask):
            return self(pixel_values, input_ids, attention_mask)
        return forward.get_concrete_function(
            tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32,
                          name="pixel_values"),
            tf.TensorSpec([None, CAPTION_LEN], tf.int64, name="input_ids"),
            tf.TensorSpec([None, CAPTION_LEN], tf.int64, name="attention_mask"),
        )

    def export(self, path):
        tf.saved_model.save(self, path, signatures={"serving_default": self.get_concrete_function()})


class LoadedModel:
    """Wrapper around tf.saved_model.load for a trained MicroVLM."""

    def __init__(self, path):
        self._m = tf.saved_model.load(path)
        self._sig = self._m.signatures["serving_default"]

    @property
    def concrete(self):
        return self._sig

    def __call__(self, images, ids, mask):
        out = self._sig(pixel_values=tf.convert_to_tensor(images),
                        input_ids=tf.convert_to_tensor(ids),
                        attention_mask=tf.convert_to_tensor(mask))
        return out["output_0"]
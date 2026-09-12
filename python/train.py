"""Train the micro VLM on the synthetic dataset (CPU friendly)."""
import argparse
import os
import time

import numpy as np
import tensorflow as tf

from common import build_tokenizer_from_meta, CAPTION_LEN, VOCAB_FILE
from model import MicroVLM, gather_trainable_variables
from synthetic import SyntheticDataset, SHAPES, COLORS, SIZES, POSITIONS

META = {"shape": SHAPES, "color": COLORS, "size": SIZES, "pos": POSITIONS}


@tf.function
def train_step(model, images, ids, mask, optimizer, trainable_vars):
    with tf.GradientTape() as tape:
        logits = model(images, ids, mask)
        loss = model.masked_loss(ids, logits, mask)
    grads = tape.gradient(loss, trainable_vars)
    grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, trainable_vars) if g is not None])
    return loss


def make_train_step(model, optimizer, trainable_vars):
    """tf.function with trainable vars captured via closure (TF requirement:
    never pass Variables as plain function arguments - updates get lost)."""
    @tf.function
    def step(images, ids, mask):
        with tf.GradientTape() as tape:
            logits = model(images, ids, mask)
            loss = model.masked_loss(ids, logits, mask)
        grads = tape.gradient(loss, trainable_vars)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, trainable_vars) if g is not None])
        return loss
    return step


def evaluate(model, images, ids, masks, texts, tokenizer, bs=128, limit=None):
    correct = total = 0
    n = limit or len(images)
    images, ids, masks, texts = images[:n], ids[:n], masks[:n], texts[:n]
    for s in range(0, len(images), bs):
        b = slice(s, s + bs)
        logits = model(images[b], ids[b], masks[b], training=False)
        pred = tf.argmax(logits, axis=-1).numpy()
        for i in range(len(images[b])):
            n_real = int(masks[b][i].sum()) - 1
            if list(pred[i][:n_real]) == list(ids[b][i][1:1 + n_real]):
                correct += 1
            total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokenizer = build_tokenizer_from_meta(META)
    print(f"[tokenizer] vocab={tokenizer.size} -> {VOCAB_FILE}")
    tokenizer.save(os.path.join(args.out, "vocab.json"))

    print("[data] building synthetic dataset ...")
    ds = SyntheticDataset(tokenizer, n_train=args.train, n_val=args.val)
    tr = ds.train()
    va = ds.val()
    print(f"[data] train={tr[0].shape[0]} val={va[0].shape[0]}")

    model = MicroVLM(tokenizer.size)
    # trace once to init
    _ = model(tr[0][:2], tr[1][:2], tr[2][:2])
    trainable = gather_trainable_variables(model)
    n_params = int(sum(np.prod(v.shape) for v in trainable))
    print(f"[model] trainable params: {n_params:,} ({len(trainable)} vars)")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        args.lr, int(np.ceil(len(tr[0]) / args.batch) * args.epochs),
        alpha=0.02)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule,
                                          weight_decay=1e-4)
    train_step = make_train_step(model, optimizer, trainable)

    best_acc = -1.0
    t0 = time.time()
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        idx = np.arange(len(tr[0]))
        rng.shuffle(idx)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(tr[0]), args.batch):
            b = idx[s:s + args.batch]
            loss = train_step(tr[0][b], tr[1][b], tr[2][b])
            ep_loss += float(loss)
            nb += 1
        if epoch == 1:
            t_per_ep = (time.time() - t0) / 1
            print(f"[epoch {epoch}] loss={ep_loss / nb:.4f} "
                  f"({t_per_ep:.0f}s/epoch, ETA {t_per_ep * (args.epochs - 1) / 60:.1f}min)")
        elif epoch % 2 == 0 or epoch == args.epochs:
            text_acc = evaluate(model, *va, tokenizer, limit=400)
            if text_acc > best_acc:
                best_acc = text_acc
                model.export(os.path.join(args.out, "model"))
            print(f"[{epoch:3d}/{args.epochs}] loss={ep_loss / nb:.4f} "
                  f"caption_acc={text_acc:.3f} best={best_acc:.3f} "
                  f"({time.time() - t0:.0f}s)")

    print(f"[done] best caption accuracy = {best_acc:.3f}")
    print(f"[done] saved to {os.path.join(args.out, 'model')}")


if __name__ == "__main__":
    main()
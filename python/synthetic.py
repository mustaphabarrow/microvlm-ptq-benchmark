"""Procedural synthetic dataset: simple geometric scenes + templated captions.

Fully deterministic (seeded), CPU-only, no external assets. This gives the
thesis a learnable visual-language task where caption accuracy is a
meaningful, measurable number before and after quantization.
"""
import io
import os

import numpy as np
from PIL import Image, ImageDraw

from common import IMG_SIZE

SHAPES = ["circle", "square", "triangle", "star", "diamond"]
COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange"]
SIZES = ["small", "large"]
POSITIONS = ["left", "center", "right"]

COLOR_RGB = {
    "red": (220, 60, 60),
    "green": (60, 200, 90),
    "blue": (70, 120, 235),
    "yellow": (235, 220, 70),
    "magenta": (220, 70, 190),
    "cyan": (70, 200, 220),
    "orange": (240, 150, 50),
}
BG_RGB = (18, 18, 30)

# deterministic anchor points for each position on a 64x64 canvas
POS_CENTER = {
    "left": (16, 32),
    "center": (32, 32),
    "right": (48, 32),
}


def caption_of(shape, color, size, pos):
    return f"a {size} {color} {shape} at the {pos}"


def draw_scene(shape, color, size, pos, img_size=IMG_SIZE):
    """Render one scene to an RGB uint8 array (C,H,W order, values 0-255)."""
    img = Image.new("RGB", (img_size, img_size), BG_RGB)
    d = ImageDraw.Draw(img)
    cx, cy = POS_CENTER[pos]
    rgb = COLOR_RGB[color]
    r = 16 if size == "large" else 9

    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
    elif shape == "star":
        pts = []
        for i in range(10):
            ang = -90 + 36 * i
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * np.cos(np.radians(ang)),
                        cy + rad * np.sin(np.radians(ang))))
        d.polygon(pts, fill=rgb)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb)

    arr = np.asarray(img, dtype=np.uint8)          # (H,W,3)
    return arr


class SyntheticDataset:
    """Builds full train/val arrays in memory (small enough for Laptop)."""

    def __init__(self, tokenizer, n_train=6000, n_val=1200, seed=0):
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        self.images, self.ids, self.masks, self.texts = self._build_all(
            n_train, n_val)
        self.n_train = n_train

    def _build_all(self, n_train, n_val):
        imgs, ids, masks, texts = [], [], [], []
        for n in (n_train, n_val):
            ni = 0
            while ni < n:
                shape = self.rng.choice(SHAPES)
                color = self.rng.choice(COLORS)
                size = self.rng.choice(SIZES)
                pos = self.rng.choice(POSITIONS)
                cap = caption_of(shape, color, size, pos)
                arr = draw_scene(shape, color, size, pos).astype(np.float32) / 255.0
                tok, msk = self.tokenizer.encode(cap)
                imgs.append(arr)
                ids.append(tok)
                masks.append(msk)
                texts.append(cap)
                ni += 1
        n_total = n_train + n_val
        images = np.asarray(imgs, dtype=np.float32)[:n_total]
        ids_arr = np.asarray(ids, dtype=np.int64)[:n_total]
        masks_arr = np.asarray(masks, dtype=np.int64)[:n_total]
        texts_arr = np.asarray(texts)[:n_total]
        return images, ids_arr, masks_arr, texts_arr

    def train(self):
        return (self.images[: self.n_train], self.ids[: self.n_train],
                self.masks[: self.n_train], self.texts[: self.n_train])

    def val(self):
        return (self.images[self.n_train:], self.ids[self.n_train:],
                self.masks[self.n_train:], self.texts[self.n_train:])


def save_sample_grid(dataset, path, n=6, seed=7):
    """Sample check figure: image + caption pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset.images), n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 6),
                             dpi=110, squeeze=False)
    for ax, i in zip(axes.flat, idx):
        img = dataset.images[i]                   # already HWC
        ax.imshow(img)
        ax.set_title(dataset.texts[i], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
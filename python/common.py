"""Shared configuration and the simple word-level tokenizer.

Everything here is framework-agnostic so the Android app can replicate the
exact same tokenizer from a single JSON file (no tokenization mismatch
between Python training and on-device generation).
"""
import json

# ---------------------------------------------------------------------------
# Hyper-parameters (mirrored in the Android app)
# ---------------------------------------------------------------------------
IMG_SIZE = 64            # synthetic image size (square)
PATCH = 8                # ViT patch size
NUM_PATCHES = (IMG_SIZE // PATCH) ** 2          # 64
VIS_PREFIX_LEN = NUM_PATCHES + 1                # 1 CLS + 64 patch tokens
CAPTION_LEN = 24                                # BOS + words + EOS, padded

D_MODEL = 128            # embedding width
VIT_DEPTH = 2            # vision encoder transformer layers
DECODER_DEPTH = 3        # language decoder transformer layers
N_HEADS = 4

VOCAB_FILE = "artifacts/vocab.json"


class SimpleTokenizer:
    """Word-level tokenizer with a compact, deterministic vocabulary.

    Specials: <pad>=0, <bos>=1, <eos>=2, <unk>=3. Sentences are encoded as
    [<bos>, w_1, ..., w_m, <eos>] then padded to CAPTION_LEN.
    """

    def __init__(self, words):
        words = sorted(set(words))
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        for w in words:
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
        self.itos = {i: w for w, i in self.vocab.items()}
        self.size = len(self.vocab)

    # -- encoding ----------------------------------------------------------
    def encode(self, text, pad_to=CAPTION_LEN):
        tok = [self.vocab["<bos>"]]
        tok += [self.vocab.get(w, self.vocab["<unk>"]) for w in text.split()]
        tok += [self.vocab["<eos>"]]
        tok = tok[: pad_to - 1] + [self.vocab["<eos>"]] if len(tok) >= pad_to else tok
        ids = tok + [self.vocab["<pad>"]] * (pad_to - len(tok))
        mask = [1] * len(tok) + [0] * (pad_to - len(tok))
        return list(ids[:pad_to]), list(mask[:pad_to])

    def decode(self, ids, ignore_specials=True):
        out = []
        for i in ids:
            w = self.itos.get(int(i), self.vocab["<unk>"])
            if ignore_specials and w in ("<pad>", "<bos>", "<eos>", "<unk>"):
                continue
            out.append(w)
        return " ".join(out)

    # -- persistence -------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "pad_len": CAPTION_LEN}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.vocab = data["vocab"]
        obj.itos = {int(i): w for w, i in obj.vocab.items()}
        obj.size = len(obj.vocab)
        return obj

    @staticmethod
    def load_minimal(path):
        """Returns (vocab dict, pad_len) without constructing an instance."""
        with open(path) as f:
            data = json.load(f)
        return data["vocab"], data["pad_len"]


def build_tokenizer_from_meta(meta):
    """Tokenize the full deterministic synthetic vocabulary."""
    words = sorted({w for cat in meta.values() for w in cat})
    return SimpleTokenizer(words)
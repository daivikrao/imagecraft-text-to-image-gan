"""GloVe loading and caption encoding.

Captions are encoded at character level: the caption is lower-cased, spaces are
removed, and the GloVe vectors of the individual characters are averaged into
one 300-d vector. This mirrors the notebook exactly.
"""
import numpy as np

from config import EMBEDDING_SIZE


def load_glove(glove_file):
    print("Loading Glove Model")
    model = {}
    with open(glove_file, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            try:
                model[parts[0]] = np.array([float(v) for v in parts[1:]])
            except ValueError:
                print(parts[0])
    print("Done.", len(model), " words loaded!")
    return model


def normalize_caption(text):
    return text.lower().replace(" ", "")


def embed_caption(text, glove):
    """Average GloVe vector over the characters of the caption."""
    embedding = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
    count = 0
    for ch in normalize_caption(text):
        if ch in glove:
            embedding += glove[ch]
            count += 1
    return embedding / max(count, 1)

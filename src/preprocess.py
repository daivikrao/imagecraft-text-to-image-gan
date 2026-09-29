"""One-time preprocessing: images -> normalized .npy batches, captions -> GloVe embeddings.

    python src/preprocess.py
"""
import os
import time

import numpy as np
from PIL import Image

import config as C
from embeddings import embed_caption, load_glove


def preprocess_images():
    """Resize every RGB image to 64x64, scale to [-1, 1] and save in chunks of 100."""
    os.makedirs(C.IMAGE_NPY_DIR, exist_ok=True)
    prefix = os.path.join(C.IMAGE_NPY_DIR, f"training_data_{C.GENERATE_SQUARE}_{C.GENERATE_SQUARE}_")
    start = time.time()
    batch = []
    for idx, name in enumerate(sorted(os.listdir(C.IMAGES_DIR))):
        try:
            image = Image.open(os.path.join(C.IMAGES_DIR, name)).resize(
                (C.GENERATE_SQUARE, C.GENERATE_SQUARE), Image.LANCZOS)
        except OSError:
            continue
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            batch.append(arr)
        if len(batch) == 100:
            data = np.asarray(batch, dtype=np.float32) / 127.5 - 1.0
            np.save(f"{prefix}{100000 + idx}.npy", data)
            print(f"Saved chunk ending at image {idx} ({time.time() - start:.1f}s)")
            batch = []


def preprocess_captions(glove):
    """Embed the first caption of every caption file (one per image)."""
    files = sorted(os.listdir(C.CAPTIONS_DIR))
    embeddings = np.zeros((len(files), C.EMBEDDING_SIZE), dtype=np.float32)
    for i, name in enumerate(files):
        with open(os.path.join(C.CAPTIONS_DIR, name)) as f:
            first_caption = f.read().split("\n")[0]
        embeddings[i] = embed_caption(first_caption, glove)
        if i % 100 == 0:
            print("Files completed:", i)
    os.makedirs(os.path.dirname(C.EMBEDDING_NPY), exist_ok=True)
    np.save(C.EMBEDDING_NPY, embeddings)


def load_training_data():
    """Concatenate the image chunks and pair them with caption embeddings."""
    chunks = sorted(os.listdir(C.IMAGE_NPY_DIR))
    images = np.concatenate([np.load(os.path.join(C.IMAGE_NPY_DIR, c)) for c in chunks])
    embeddings = np.load(C.EMBEDDING_NPY)[: len(images)]
    return images, embeddings


if __name__ == "__main__":
    preprocess_images()
    preprocess_captions(load_glove(C.GLOVE_PATH))

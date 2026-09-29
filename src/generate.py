"""Generate a 4x7 grid of flowers from a text prompt.

    python src/generate.py "this flower is yellow in color with oval shaped petals" \
        --weights flowers/model/generator_epoch_500.weights.h5
"""
import argparse
import os

import numpy as np
from PIL import Image

import config as C


def save_grid(generator, noise, embeds, filename):
    """Render generator outputs into one preview image (white margins between tiles)."""
    size, margin = C.GENERATE_SQUARE, C.PREVIEW_MARGIN
    grid = np.full((margin + C.PREVIEW_ROWS * (size + margin),
                    margin + C.PREVIEW_COLS * (size + margin), 3), 255, dtype=np.uint8)

    images = generator.predict((noise, embeds), verbose=0) * 0.5 + 0.5
    for i in range(C.PREVIEW_ROWS * C.PREVIEW_COLS):
        r = (i // C.PREVIEW_COLS) * (size + margin) + margin
        c = (i % C.PREVIEW_COLS) * (size + margin) + margin
        grid[r:r + size, c:c + size] = (images[i] * 255).astype(np.uint8)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(C.OUTPUT_DIR, filename)
    Image.fromarray(grid).save(path)
    return path


def main():
    from embeddings import embed_caption, load_glove
    from models import build_generator

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="prompt.png")
    args = parser.parse_args()

    generator = build_generator(C.SEED_SIZE, C.EMBEDDING_SIZE, C.IMAGE_CHANNELS)
    generator.load_weights(args.weights)

    n = C.PREVIEW_ROWS * C.PREVIEW_COLS
    embedding = embed_caption(args.prompt, load_glove(C.GLOVE_PATH))
    embeds = np.repeat(embedding[None, :], n, axis=0)
    noise = np.random.normal(0, 1, (n, C.SEED_SIZE)).astype(np.float32)
    print("Saved", save_grid(generator, noise, embeds, args.out))


if __name__ == "__main__":
    main()

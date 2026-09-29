"""Paths and hyperparameters. Values match notebooks/text_to_image_gan.ipynb."""
import os

# Data layout (Oxford-102 Flowers + text_c10 captions)
DATA_ROOT = os.environ.get("FLOWERS_ROOT", "flowers")
IMAGES_DIR = os.path.join(DATA_ROOT, "images", "jpg")
CAPTIONS_DIR = os.path.join(DATA_ROOT, "text_c10", "captions")
IMAGE_NPY_DIR = os.path.join(DATA_ROOT, "images", "npy64")
EMBEDDING_NPY = os.path.join(DATA_ROOT, "images", "embedding_npy", "embedding_data.npy")
MODEL_DIR = os.path.join(DATA_ROOT, "model")
OUTPUT_DIR = os.path.join(DATA_ROOT, "output_64_character_extended")

# Pre-trained GloVe vectors (300d)
GLOVE_PATH = os.environ.get("GLOVE_PATH", "glove.6B.300d.txt")

# Image generation
GENERATE_RES = 2                      # 1=32px, 2=64px, 3=96px, ...
GENERATE_SQUARE = 32 * GENERATE_RES   # 64x64 output
IMAGE_CHANNELS = 3

# Preview grid saved each epoch
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Model / training
SEED_SIZE = 100
EMBEDDING_SIZE = 300
BATCH_SIZE = 64
BUFFER_SIZE = 4000
LEARNING_RATE = 2.0e-4
BETA_1 = 0.5

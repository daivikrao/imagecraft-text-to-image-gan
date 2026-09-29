"""Train the text-conditioned GAN.

    python src/train.py --epochs 500
    python src/train.py --epochs 600 --resume 500   # continue from saved weights
"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf

import config as C
from generate import save_grid
from models import build_discriminator, build_generator
from preprocess import load_training_data

cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_image_real_text, fake_image_real_text, real_image_fake_text):
    # One-sided noisy labels: real ~ U(0.8, 1.0), fake ~ U(0.0, 0.2)
    real_loss = cross_entropy(tf.random.uniform(tf.shape(real_image_real_text), 0.8, 1.0),
                              real_image_real_text)
    fake_loss = (cross_entropy(tf.random.uniform(tf.shape(fake_image_real_text), 0.0, 0.2),
                               fake_image_real_text)
                 + cross_entropy(tf.random.uniform(tf.shape(real_image_fake_text), 0.0, 0.2),
                                 real_image_fake_text)) / 2
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_train_step(generator, discriminator, gen_opt, disc_opt):
    @tf.function
    def train_step(images, captions, fake_captions):
        seed = tf.random.normal([tf.shape(images)[0], C.SEED_SIZE])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = generator((seed, captions), training=True)
            real_real = discriminator((images, captions), training=True)
            real_fake = discriminator((images, fake_captions), training=True)
            fake_real = discriminator((generated, captions), training=True)

            gen_loss = generator_loss(fake_real)
            disc_loss = discriminator_loss(real_real, fake_real, real_fake)

        gen_opt.apply_gradients(zip(gen_tape.gradient(gen_loss, generator.trainable_variables),
                                    generator.trainable_variables))
        disc_opt.apply_gradients(zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables),
                                     discriminator.trainable_variables))
        return gen_loss, disc_loss

    return train_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--resume", type=int, default=0, help="epoch checkpoint to resume from")
    args = parser.parse_args()

    images, embeddings = load_training_data()
    p = np.random.permutation(len(images))
    dataset = (tf.data.Dataset.from_tensor_slices({"images": images[p], "embeddings": embeddings[p]})
               .shuffle(C.BUFFER_SIZE).batch(C.BATCH_SIZE, drop_remainder=True))

    generator = build_generator(C.SEED_SIZE, C.EMBEDDING_SIZE, C.IMAGE_CHANNELS)
    discriminator = build_discriminator((C.GENERATE_SQUARE, C.GENERATE_SQUARE, C.IMAGE_CHANNELS),
                                        C.EMBEDDING_SIZE)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    if args.resume:
        generator.load_weights(os.path.join(C.MODEL_DIR, f"generator_epoch_{args.resume}.weights.h5"))
        discriminator.load_weights(os.path.join(C.MODEL_DIR, f"discriminator_epoch_{args.resume}.weights.h5"))

    gen_opt = tf.keras.optimizers.Adam(C.LEARNING_RATE, beta_1=C.BETA_1)
    disc_opt = tf.keras.optimizers.Adam(C.LEARNING_RATE, beta_1=C.BETA_1)
    train_step = make_train_step(generator, discriminator, gen_opt, disc_opt)

    # Fixed noise + first 28 captions, so preview grids are comparable across epochs
    n_preview = C.PREVIEW_ROWS * C.PREVIEW_COLS
    fixed_seed = np.random.normal(0, 1, (n_preview, C.SEED_SIZE)).astype(np.float32)
    fixed_embed = embeddings[:n_preview]

    for epoch in range(args.resume, args.epochs):
        start = time.time()
        g_losses, d_losses = [], []
        for batch in dataset:
            fake_captions = tf.random.shuffle(batch["embeddings"])
            g, d = train_step(batch["images"], batch["embeddings"], fake_captions)
            g_losses.append(float(g))
            d_losses.append(float(d))
        print(f"Epoch {epoch + 1}, gen loss={np.mean(g_losses):.4f}, "
              f"disc loss={np.mean(d_losses):.4f}, {time.time() - start:.1f}s")

        save_grid(generator, fixed_seed, fixed_embed, f"train-{epoch}.png")
        generator.save_weights(os.path.join(C.MODEL_DIR, f"generator_epoch_{epoch + 1}.weights.h5"))
        discriminator.save_weights(os.path.join(C.MODEL_DIR, f"discriminator_epoch_{epoch + 1}.weights.h5"))


if __name__ == "__main__":
    main()

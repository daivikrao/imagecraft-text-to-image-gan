"""Conditional DCGAN generator and discriminator (text-conditioned via GloVe)."""
from tensorflow.keras import initializers
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten, Input,
                                     LeakyReLU, Reshape, UpSampling2D)
from tensorflow.keras.models import Model

from config import GENERATE_RES


def _init():
    return initializers.RandomNormal(stddev=0.02)


def _up_block(x, filters, kernel_size, size=(2, 2)):
    x = UpSampling2D(size=size)(x)
    x = Conv2DTranspose(filters, kernel_size=kernel_size, padding="same",
                        kernel_initializer=_init())(x)
    x = BatchNormalization(momentum=0.8)(x)
    return LeakyReLU(0.2)(x)


def build_generator(seed_size, embedding_size, channels):
    """noise (100) + projected text embedding (128) -> 64x64x3 image in [-1, 1]."""
    input_seed = Input(shape=(seed_size,))
    input_embed = Input(shape=(embedding_size,))

    text = LeakyReLU(0.2)(Dense(128)(input_embed))
    x = Concatenate()([input_seed, text])
    x = Dense(4 * 4 * 256, activation="relu")(x)
    x = Reshape((4, 4, 256))(x)

    x = _up_block(x, 256, 5)                                   # 8x8
    x = _up_block(x, 256, 5)                                   # 16x16
    x = _up_block(x, 128, 4)                                   # 32x32
    x = _up_block(x, 128, 4, size=(GENERATE_RES, GENERATE_RES))  # 64x64

    x = Conv2DTranspose(channels, kernel_size=3, padding="same",
                        kernel_initializer=_init())(x)
    out = Activation("tanh")(x)
    return Model(inputs=[input_seed, input_embed], outputs=out)


def _down_block(x, filters, batch_norm=True):
    x = Dropout(0.25)(x)
    x = Conv2D(filters, kernel_size=4, strides=2, padding="same",
               kernel_initializer=_init())(x)
    if batch_norm:
        x = BatchNormalization(momentum=0.8)(x)
    return LeakyReLU(0.2)(x)


def build_discriminator(image_shape, embedding_size):
    """(image, text embedding) -> probability that the pair is real and matching."""
    input_image = Input(shape=image_shape)
    input_embed = Input(shape=(embedding_size,))

    x = Conv2D(32, kernel_size=4, strides=2, padding="same",
               kernel_initializer=_init())(input_image)             # 32x32
    x = LeakyReLU(0.2)(x)
    x = _down_block(x, 64)                                           # 16x16
    x = _down_block(x, 128)                                          # 8x8
    x = _down_block(x, 256)                                          # 4x4

    # Project text to 128 values and tile it onto the 4x4 feature map
    text = Dense(128, kernel_initializer=_init())(input_embed)
    text = Reshape((4, 4, 8))(LeakyReLU(0.2)(text))
    x = Concatenate()([x, text])

    x = Dropout(0.25)(x)
    x = Conv2D(512, kernel_size=4, kernel_initializer=_init())(x)   # 1x1
    x = LeakyReLU(0.2)(BatchNormalization(momentum=0.8)(x))
    x = Flatten()(Dropout(0.25)(x))
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=[input_image, input_embed], outputs=out)

import os
import keras
import numpy as np


# --------------------------------------------------------------------------------


def resnet_block(input_layer,
                 filters=64,
                 downsample=False,
                 use_dropout=False,
                 use_batchnorm=True):
    """

    :param input_layer:
    :param filters:
    :param downsample:
    :param use_dropout:
    :param use_batchnorm:
    :param skip_connection:
    :return:
    """
    x = input_layer

    if downsample:
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            activation="linear",
            kernel_initializer="glorot_uniform")(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="linear",
        kernel_initializer="glorot_uniform")(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization()(x)

    if use_dropout:
        x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.ReLU(x)

    return x


# --------------------------------------------------------------------------------


def basic_block(input_layer,
                block_type="encoder",
                filters=[64],
                kernel_size=[(3, 3)],
                strides=[(1, 1)],
                use_batchnorm=True,
                use_dropout=True,
                prefix="block_", ):
    """

    :param input_layer:
    :param block_type:
    :param filters:
    :param kernel_size:
    :param strides:
    :param use_batchnorm:
    :param use_dropout:
    :param prefix:
    :return:
    """
    if len(filters) != len(kernel_size) or \
            len(filters) != len(strides) or \
            len(filters) <= 0:
        raise ValueError("len(filters) should be equal to len(kernel_size) and len(strides)")

    if block_type != "encoder" and block_type != "decoder":
        raise ValueError("block_type should be encoder or decoder")

    x = input_layer

    for i in range(len(filters)):
        tmp_layer = x

        if block_type == "encoder":
            x = keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernel_size[i],
                strides=strides[i],
                padding="same",
                activation="linear",
                kernel_initializer="glorot_uniform")(x)
        elif block_type == "decoder":
            x = keras.layers.Conv2DTranspose(
                filters=filters[i],
                kernel_size=kernel_size[i],
                strides=strides[i],
                padding="same",
                activation="linear",
                kernel_initializer="glorot_uniform")(x)

        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size[i],
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_initializer="glorot_uniform")(x)

        if use_batchnorm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        # --------- Add bottleneck layer
        if strides[i][0] == 1 and \
                strides[i][0] == strides[i][1]:
            tmp_layer = keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=(1, 1),
                strides=strides[i],
                activation="linear",
                kernel_initializer="glorot_uniform",
                padding="same")(tmp_layer)
            x = keras.layers.Add()([
                x,
                tmp_layer
            ])
        x = keras.layers.LeakyReLU()(x)

        if use_dropout:
            x = keras.layers.Dropout(rate=0.1)(x)

    return x

# --------------------------------------------------------------------------------

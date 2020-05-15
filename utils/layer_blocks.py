import os
import sys
import keras
import logging
import utils.coord

# --------------------------------------------------------------------------------
# setup logger
# --------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s")
logging.getLogger("layer-blocks").setLevel(logging.INFO)
logger = logging.getLogger("layer-blocks")

# --------------------------------------------------------------------------------


def absence_block(t):
    x_greater = keras.backend.greater(t, 0.0)
    x_greater_float = keras.backend.cast(x_greater, t.dtype)
    return 1.0 - x_greater_float

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
    skip_layer = input_layer
    previous_no_filters = keras.backend.int_shape(input_layer)[3]

    if downsample:
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            activation="linear",
            kernel_initializer="glorot_uniform")(x)
        skip_layer = keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=None,
            padding="valid")(skip_layer)
    else:
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_initializer="glorot_uniform")(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.LeakyReLU(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="linear",
        kernel_initializer="glorot_uniform")(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization()(x)

    if previous_no_filters != filters:
        skip_layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_initializer="glorot_uniform")(skip_layer)

    x = keras.layers.Add()([
        x,
        skip_layer
    ])

    if use_dropout:
        x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.LeakyReLU(x)

    return x


# --------------------------------------------------------------------------------


def basic_block(input_layer,
                block_type="encoder",
                filters=[64],
                kernel_size=[(3, 3)],
                strides=[(1, 1)],
                use_batchnorm=True,
                use_dropout=True,
                use_absense_block=False,
                use_coordconv=True,
                prefix="block_", ):
    """

    :param input_layer:
    :param block_type:
    :param filters:
    :param kernel_size:
    :param strides:
    :param use_batchnorm:
    :param use_dropout:
    :param use_absense_block:
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

    if use_coordconv:
        x = utils.coord.CoordinateChannel2D()(x)

    for i in range(len(filters)):

        if i > 0:
            if use_absense_block:
                absence_x = keras.layers.Lambda(
                    lambda y: absence_block(y))(x)
                x = keras.layers.Concatenate()([
                    absence_x,
                    x
                ])

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
        # --------- Relu combined result
        x = keras.layers.LeakyReLU()(x)

        if use_dropout:
            x = keras.layers.Dropout(rate=0.1)(x)

    return x

# --------------------------------------------------------------------------------

import keras
import logging
import mvae.utils.coord

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


def basic_block(input_layer,
                block_type="encoder",
                filters=[64],
                kernel_size=[(3, 3)],
                strides=[(1, 1)],
                use_batchnorm=False,
                use_dropout=False,
                prefix="block_"):
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
    previous_no_filters = keras.backend.int_shape(input_layer)[3]

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

        x = keras.layers.ReLU()(x)

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
        if (strides[i][0] == 1 and strides[i][0] == strides[i][1]) and \
                previous_no_filters != filters[i]:

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
        x = keras.layers.ReLU()(x)

        if use_dropout:
            x = keras.layers.Dropout(rate=0.1)(x)

        previous_no_filters = filters[i]

    return x

# --------------------------------------------------------------------------------

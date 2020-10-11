import keras
import logging
import mvae.utils.coord

# ------------------------------------------------------------------------------
# setup logger
# ------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s")
logging.getLogger("layer-blocks").setLevel(logging.INFO)
logger = logging.getLogger("layer-blocks")

# ------------------------------------------------------------------------------


def absence_block(t):
    x_greater = keras.backend.greater(t, 0.0)
    x_greater_float = keras.backend.cast(x_greater, t.dtype)
    return 1.0 - x_greater_float

# ------------------------------------------------------------------------------


def resnet_block(input_layer,
                 filters=32,
                 kernel_size=[3, 3],
                 strides=(1, 1),
                 activation="relu",
                 initializer="glorot_normal",
                 prefix="resnet_",
                 channels_index=3,
                 use_dropout=False,
                 use_batchnorm=False,
                 dropout_ratio=0.5):
    """
    Build a resnet block
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param strides:
    :param activation:
    :param initializer:
    :param prefix:
    :param channels_index:
    :param use_dropout:
    :param use_batchnorm:
    :param dropout_ratio:
    :return: Resnet block
    """
    # -------- argument checking
    if filter <= 0:
        raise ValueError("Filters should be > 0")
    previous_no_filters = keras.backend.int_shape(input_layer)[channels_index]

    # -------- build block
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=activation,
        name=prefix + "conv0",
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.Conv2D(
        filters=filters[i],
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation="linear",
        name=prefix + "conv1",
        kernel_initializer=initializer)(x)

    if previous_no_filters == filter:
        tmp_layer = keras.layers.Layer(name=prefix + "skip")(input_layer)
    else:
        tmp_layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="linear",
            name=prefix + "skip",
            kernel_initializer=initializer)(input_layer)

    # -------- build skip layer and main
    x = keras.layers.Add(name=prefix + "add")([
        x,
        tmp_layer
    ])

    x = keras.layers.Activation(
        activation,
        name=prefix + "activation")(x)

    if use_dropout and dropout_ratio > 0.0:
        x = keras.layers.Dropout(
            name=prefix + "dropout",
            rate=dropout_ratio)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm")(x)

    return x

# ==========================================================================


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
        raise ValueError("len(filters) should be equal to "
                         "len(kernel_size) and len(strides)")

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
                kernel_initializer="glorot_normal")(x)
        elif block_type == "decoder":
            x = keras.layers.Conv2DTranspose(
                filters=filters[i],
                kernel_size=kernel_size[i],
                strides=strides[i],
                padding="same",
                activation="linear",
                kernel_initializer="glorot_normal")(x)

        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size[i],
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_initializer="glorot_normal")(x)

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

# ------------------------------------------------------------------------------

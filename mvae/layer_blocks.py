import keras
import numpy as np
import scipy.stats as st
from keras import backend as K
from .custom_logger import logger

# ==============================================================================


def squeeze_excite_block(input_layer,
                         squeeze_units=32,
                         prefix="squeeze_excite_",
                         channels_index=3):
    """

    """
    # -------- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if squeeze_units <= 0:
        raise ValueError("squeeze_units must be > 0")
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("works only on 4d tensors")
    channels = shape[channels_index]
    # -------- squeeze
    x = keras.layers.GlobalMaxPool2D(name=prefix + "max_pool")(input_layer)
    x = keras.layers.Dense(units=squeeze_units,
                           name=prefix + "dense0",
                           activation="relu")(x)
    x = keras.layers.Dense(units=channels,
                           name=prefix + "dense1",
                           activation="hard_sigmoid")(x)
    # -------- scale input
    x = keras.layers.Multiply(name=prefix + "multiply",)([x, input_layer])
    return x

# ==============================================================================


def mobilenetV2_block(input_layer,
                      filters=32,
                      initializer="glorot_normal",
                      prefix="mobilenetV2_",
                      channels_index=3,
                      dropout_ratio=None,
                      use_batchnorm=False):
    """
    Build a mobilenet V2 bottleneck with residual block
    :param input_layer:
    :param filters:
    :param initializer:
    :param prefix:
    :param channels_index:
    :param use_batchnorm:
    :param dropout_ratio:
    :return: mobilenet V2 bottleneck with residual block
    """
    # -------- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    if dropout_ratio is not None:
        if dropout_ratio > 1.0 or dropout_ratio < 0.0:
            raise ValueError("Dropout ration must be [0, 1]")
    # -------- build block
    previous_no_filters = K.int_shape(input_layer)[channels_index]

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation="linear",
        name=prefix + "conv0",
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation="relu",
        name=prefix + "conv1",
        kernel_initializer=initializer)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm0")(x)

    x = keras.layers.Conv2D(
        filters=previous_no_filters,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation="relu",
        name=prefix + "conv2",
        kernel_initializer=initializer)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm1")(x)

    # -------- build skip layer and main
    x = keras.layers.Add(name=prefix + "add")([
        x,
        input_layer
    ])

    if dropout_ratio is not None:
        if dropout_ratio > 0.0:
            x = keras.layers.Dropout(
                name=prefix + "dropout",
                rate=dropout_ratio)(x)

    return x

# ==============================================================================


def mobilenetV3_block(input_layer,
                      filters=32,
                      squeeze_dim=4,
                      initializer="glorot_normal",
                      prefix="mobilenetV3_",
                      activation="relu",
                      channels_index=3,
                      dropout_ratio=None,
                      use_batchnorm=False):
    """
    Build a mobilenet V3 block is bottleneck with residual + squeeze and excite
    :param input_layer:
    :param filters:
    :param initializer:
    :param activation:
    :param squeeze_dim:
    :param prefix:
    :param channels_index:
    :param use_batchnorm:
    :param dropout_ratio:
    :return: mobilenet V3 block
    """
    # -------- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    if dropout_ratio is not None:
        if dropout_ratio > 1.0 or dropout_ratio < 0.0:
            raise ValueError("Dropout ration must be [0, 1]")
    # -------- build block
    previous_no_filters = K.int_shape(input_layer)[channels_index]

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "conv0",
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "conv1",
        kernel_initializer=initializer)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm0")(x)

    x = squeeze_excite_block(x,
                             squeeze_units=squeeze_dim,
                             prefix=prefix + "squeeze_excite_")

    x = keras.layers.Conv2D(
        filters=previous_no_filters,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation="linear",
        name=prefix + "conv2",
        kernel_initializer=initializer)(x)

    # -------- build skip layer and main
    x = keras.layers.Add(name=prefix + "add")([
        x,
        input_layer
    ])

    if dropout_ratio is not None:
        if dropout_ratio > 0.0:
            x = keras.layers.Dropout(
                name=prefix + "dropout",
                rate=dropout_ratio)(x)

    return x

# ==============================================================================


def attention_block(input_layer,
                    filters=32,
                    kernel_size=[1, 1],
                    activation="linear",
                    initializer="glorot_normal",
                    prefix="attention_"):
    """
    Builds a attention block
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param activation:
    :param initializer:
    :param prefix:
    :return:
    """
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("only supports 4d tensors")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    padding = "same"
    strides = (1, 1)
    # -------- theta, phi, g
    theta = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=prefix + "theta",
        kernel_initializer=initializer)(input_layer)
    phi = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=prefix + "phi",
        kernel_initializer=initializer)(input_layer)
    g = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=prefix + "g",
        kernel_initializer=initializer)(input_layer)
    # -------- build block
    h_x_w = np.prod(shape[1:3])
    new_shape = (h_x_w, filters)
    theta_flat = keras.layers.Reshape(new_shape)(theta)
    phi_flat = keras.layers.Reshape(new_shape)(phi)
    phi_flat = keras.layers.Permute((2, 1))(phi_flat)
    theta_x_phi = keras.layers.Dot(axes=(1, 2))([theta_flat, phi_flat])
    theta_x_phi = keras.layers.Softmax()(theta_x_phi)
    g_flat = keras.layers.Reshape(new_shape)(g)
    # -------- multiply with attention map
    theta_x_phi_xg = keras.layers.Dot(axes=(1, 2))([theta_x_phi, g_flat])
    theta_x_phi_xg = keras.layers.Reshape((shape[1:3] + (filters,)))(theta_x_phi_xg)

    return theta_x_phi_xg

# ==============================================================================


def self_attention_block(input_layer,
                         filters=32,
                         kernel_size=[1, 1],
                         activation="linear",
                         initializer="glorot_normal",
                         prefix="self_attention_",
                         channels_index=3):
    """
    Builds a self-attention block, attention block plus residual skip
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param activation:
    :param initializer:
    :param prefix:
    :param channels_index:
    :return:
    """
    # -------- argument checking
    attention = attention_block(input_layer=input_layer,
                                filters=filters,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer,
                                prefix=prefix)
    # -------- convolve to match output channelsv
    shape = K.int_shape(input_layer)
    previous_no_filters = shape[channels_index]
    attention = keras.layers.Conv2D(
        filters=previous_no_filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "result",
        kernel_initializer=initializer)(attention)
    # -------- residual connection with input
    return keras.layers.Add()([attention, input_layer])


# ==============================================================================


def resnet_block(input_layer,
                 filters=32,
                 kernel_size=[3, 3],
                 activation="relu",
                 initializer="glorot_normal",
                 prefix="resnet_",
                 channels_index=3,
                 dropout_ratio=None,
                 use_batchnorm=False):
    """
    Build a resnet block
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param activation:
    :param initializer:
    :param prefix:
    :param channels_index:
    :param use_batchnorm:
    :param dropout_ratio:
    :return: Resnet block
    """
    # -------- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    if dropout_ratio is not None:
        if dropout_ratio > 1.0 or dropout_ratio < 0.0:
            raise ValueError("Dropout ration must be [0, 1]")
    strides = (1, 1)
    # -------- build block
    previous_no_filters = K.int_shape(input_layer)[channels_index]

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=activation,
        name=prefix + "conv0",
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.Conv2D(
        filters=filters,
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

    if dropout_ratio is not None:
        if dropout_ratio > 0.0:
            x = keras.layers.Dropout(
                name=prefix + "dropout",
                rate=dropout_ratio)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm")(x)

    return x


# ==============================================================================


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
    previous_no_filters = K.int_shape(input_layer)[3]

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


# ==============================================================================


def gaussian_filter_block(input_layer,
                          kernel_size=3,
                          strides=(1, 1),
                          dilation_rate=(1, 1),
                          padding="same",
                          activation=None,
                          trainable=False,
                          use_bias=False):
    """
    Build a gaussian filter block
    :param input_layer:
    :param kernel_size:
    :param activation:
    :param trainable:
    :param use_bias:
    :param strides:
    :param padding:
    :param dilation_rate:
    :return:
    """

    def _gaussian_kernel(kernlen=[21, 21], nsig=[3, 3]):
        """
        Returns a 2D Gaussian kernel array
        """
        assert len(nsig) == 2
        assert len(kernlen) == 2
        kern1d = []
        for i in range(2):
            interval = (2 * nsig[i] + 1.) / (kernlen[i])
            x = np.linspace(-nsig[i] - interval / 2., nsig[i] + interval / 2.,
                            kernlen[i] + 1)
            kern1d.append(np.diff(st.norm.cdf(x)))

        kernel_raw = np.sqrt(np.outer(kern1d[0], kern1d[1]))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    # Initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        kernel[:, :, 0, 0] = _gaussian_kernel([shape[0], shape[1]])
        return kernel

    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=1,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        trainable=trainable,
        depthwise_initializer=kernel_init,
        kernel_initializer=kernel_init)(input_layer)

# ==============================================================================

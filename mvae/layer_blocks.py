import keras
import numpy as np
import scipy.stats as st
from keras import backend as K
from .custom_logger import logger

# ==============================================================================


def excite_inhibit_block(input_layer,
                         filters=32,
                         kernel_size=[3, 3],
                         channels_index=3,
                         residual=True,
                         prefix="excite_inhibit_"):
    # -------- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    # -------- infer shape
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("works only on 4d tensors")
    channels = shape[channels_index]
    # --------
    kernel_regularizer = keras.regularizers.l1_l2(l1=0.75, l2=0.25)
    initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.1, seed=None)
    # -------- excite
    x_e = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation="relu",
        name=prefix + "conv0_excite",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=initializer)(input_layer)
    # -------- inhibit
    x_i = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation="relu",
        name=prefix + "conv0_inhibit",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=initializer)(input_layer)
    # -------- mix and apply non linearity
    x = x_e - x_i
    x = keras.layers.ReLU()(x)
    # -------- bring back to original channels
    if residual:
        if channels != filters:
            x = keras.layers.Conv2D(
                    filters=channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    activation="linear",
                    name=prefix + "conv1",
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=initializer)(x)
        x = x + input_layer
    return x

# ==============================================================================


def squeeze_excite_block(input_layer,
                         squeeze_units=32,
                         initializer="glorot_normal",
                         regularizer=None,
                         channels_index=3,
                         prefix="squeeze_excite_"):
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
    x = keras.layers.GlobalMaxPool2D(
        name=prefix + "max_pool")(input_layer)
    x = keras.layers.Dense(
        units=squeeze_units,
        name=prefix + "dense0",
        activation="relu",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)
    x = keras.layers.Dense(
        units=channels,
        name=prefix + "dense1",
        activation="hard_sigmoid",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)
    # -------- scale input
    x = keras.layers.Multiply(
        name=prefix + "multiply")([x, input_layer])
    return x

# ==============================================================================


def mobilenetV2_block(input_layer,
                      filters=32,
                      initializer="glorot_normal",
                      regularizer=None,
                      prefix="mobilenetV2_",
                      channels_index=3,
                      dropout_ratio=None,
                      use_batchnorm=False):
    """
    Build a mobilenet V2 bottleneck with residual block
    :param input_layer:
    :param filters:
    :param initializer:
    :param regularizer:
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
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation="relu",
        name=prefix + "conv1",
        kernel_regularizer=regularizer,
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
        kernel_regularizer=regularizer,
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
                      regularizer=None,
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
    :param regularizer:
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
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "conv1",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer,
        depthwise_initializer=initializer,
        depthwise_regularizer=regularizer)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm0")(x)

    x = squeeze_excite_block(x,
                             squeeze_units=squeeze_dim,
                             regularizer=regularizer,
                             initializer=initializer,
                             prefix=prefix + "squeeze_excite_")

    x = keras.layers.Conv2D(
        filters=previous_no_filters,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation="linear",
        name=prefix + "conv2",
        kernel_regularizer=regularizer,
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
                    regularizer=None,
                    prefix="attention_"):
    """
    Builds a attention block
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param activation:
    :param initializer:
    :param regularizer:
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
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)
    phi = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=prefix + "phi",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)
    g = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=prefix + "g",
        kernel_regularizer=regularizer,
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
    theta_x_phi_xg = keras.layers.Reshape(
        (shape[1:3] + (filters,)))(theta_x_phi_xg)

    return theta_x_phi_xg

# ==============================================================================


def self_attention_block(input_layer,
                         filters=32,
                         kernel_size=[1, 1],
                         activation="linear",
                         initializer="glorot_normal",
                         regularizer=None,
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
                                regularizer=regularizer,
                                prefix=prefix)
    # -------- convolve to match output channels
    shape = K.int_shape(input_layer)
    previous_no_filters = shape[channels_index]
    attention = keras.layers.Conv2D(
        filters=previous_no_filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "result",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(attention)
    # -------- residual connection with input
    return keras.layers.Add()([attention, input_layer])


# ==============================================================================


def resnet_block(input_layer,
                 filters=32,
                 kernel_size=[3, 3],
                 activation="relu",
                 initializer="glorot_normal",
                 regularizer=None,
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
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation="linear",
        name=prefix + "conv1",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)

    if previous_no_filters == filters:
        tmp_layer = keras.layers.Layer(name=prefix + "skip")(input_layer)
    else:
        tmp_layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="linear",
            name=prefix + "skip",
            kernel_regularizer=regularizer,
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
                initializer="glorot_normal",
                regularizer=None,
                use_batchnorm=False,
                use_dropout=False,
                prefix="block_",):
    """

    :param input_layer:
    :param block_type:
    :param filters:
    :param kernel_size:
    :param strides:
    :param initializer:
    :param regularizer:
    :param use_batchnorm:
    :param use_dropout:
    :param prefix:
    :return:
    """
    if len(filters) != len(kernel_size) or \
            len(filters) != len(strides) or \
            len(filters) <= 0:
        raise ValueError(
            "len(filters) [{0}] should be equal to "
            "len(kernel_size) [{1}] and len(strides) [{2}]".format(
                len(filters), len(kernel_size), len(strides)))

    if block_type != "encoder" and block_type != "decoder":
        raise ValueError("block_type should be encoder or decoder")

    x = input_layer
    padding = "same"
    activation = "linear"
    previous_no_filters = K.int_shape(input_layer)[3]

    for i in range(len(filters)):
        prefix_i = prefix + str(i) + "_"
        # --- handle subsampling and change in the number of filters
        if strides[i][0] != 1 or strides[i][1] != 1 or \
                filters[i] != previous_no_filters:
            if block_type == "encoder":
                x = keras.layers.Conv2D(
                    filters=filters[i],
                    kernel_size=(3, 3),
                    strides=strides[i],
                    padding=padding,
                    activation=activation,
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer)(x)
            elif block_type == "decoder":
                x = keras.layers.Conv2DTranspose(
                    filters=filters[i],
                    kernel_size=(3, 3),
                    strides=strides[i],
                    padding=padding,
                    activation=activation,
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer)(x)

        x = excite_inhibit_block(x, filters[i], prefix=prefix_i + "excite_inhibit_")
        # x = mobilenetV3_block(
        #     x,
        #     filters=filters[i],
        #     activation="relu",
        #     initializer=initializer,
        #     regularizer=regularizer,
        #     prefix=prefix_i + "mobilenetV3_")

        if use_dropout:
            x = keras.layers.Dropout(rate=0.1)(x)

        if use_batchnorm:
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        previous_no_filters = filters[i]

    return x


# ==============================================================================

def gaussian_kernel(size, nsig):
    """
    Returns a 2D Gaussian kernel array
    """
    assert len(nsig) == 2
    assert len(size) == 2
    kern1d = []
    for i in range(2):
        x = np.linspace(start=-np.abs(nsig[i]),
                        stop=np.abs(nsig[i]),
                        num=size[i],
                        endpoint=True)
        kern1d.append(x)
    x, y = np.meshgrid(kern1d[0], kern1d[1])
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    kernel = g / g.sum()
    return kernel

# ==============================================================================


def gaussian_filter_block(input_layer,
                          kernel_size=3,
                          strides=(1, 1),
                          dilation_rate=(1, 1),
                          padding="same",
                          xy_max=(1.5, 1.5),
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
    :param xy_max:
    :param dilation_rate:
    :return:
    """

    # Initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        kernel[:, :, 0, 0] = gaussian_kernel([shape[0], shape[1]], xy_max)
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

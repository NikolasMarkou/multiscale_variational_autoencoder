import keras
import numpy as np
from keras import backend as K

# ==============================================================================

from .custom_logger import logger

# ==============================================================================

DEFAULT_DROPOUT_RATIO = 0.0
DEFAULT_CHANNEL_INDEX = 3
DEFAULT_ATTENUATION_MULTIPLIER = 4.0
DEFAULT_KERNEL_REGULARIZER = "l1"
DEFAULT_KERNEL_INITIALIZER = "glorot_normal"
DEFAULT_GAUSSIAN_XY_MAX = (1, 1)
DEFAULT_GAUSSIAN_KERNEL_SIZE = (3, 3)


# ==============================================================================


def laplacian_transform_split(
        input_dims,
        levels: int,
        name: str = None,
        min_value: float = 0.0,
        max_value: float = 255.0,
        gaussian_xy_max: tuple = DEFAULT_GAUSSIAN_XY_MAX,
        gaussian_kernel_size: tuple = DEFAULT_GAUSSIAN_KERNEL_SIZE):
    """
    Normalize input values and the compute laplacian pyramid
    """

    def _normalize(args):
        """
        Convert input from [v0, v1] to [-1, +1] range
        """
        y, v0, v1 = args
        return 2.0 * (y - v0) / (v1 - v0) - 1.0

    def _downsample_upsample(
            i0,
            prefix: str = "downsample_upsample"):
        """
        Downsample and upsample the input
        :param i0: input
        :return:
        """

        # gaussian filter
        filtered = \
            gaussian_filter_block(
                i0,
                strides=(1, 1),
                xy_max=gaussian_xy_max,
                kernel_size=gaussian_kernel_size)

        # downsample by order of 2
        filtered_downsampled = \
            keras.layers.MaxPool2D(
                pool_size=(1, 1),
                strides=(2, 2),
                padding="valid",
                name=prefix + "down")(filtered)

        # upsample back
        filtered_downsampled_upsampled = \
            keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(filtered_downsampled)

        diff = keras.layers.Subtract()([i0, filtered_downsampled_upsampled])
        return filtered_downsampled, diff

    # --- prepare input
    input_layer = \
        keras.Input(shape=input_dims)

    input_normalized_layer = \
        keras.layers.Lambda(
            _normalize,
            name="normalize")([input_layer, min_value, max_value])

    # --- split input in levels
    output_multiscale_layers = []
    for i in range(levels):
        if i == levels - 1:
            output_multiscale_layers.append(
                input_normalized_layer)
        else:
            input_normalized_layer, up = \
                _downsample_upsample(
                    input_normalized_layer, prefix=f"du_{i}_")
            output_multiscale_layers.append(up)

    return \
        keras.Model(
            name=name,
            inputs=input_layer,
            outputs=output_multiscale_layers)


# ==============================================================================


def laplacian_transform_merge(
        input_dims,
        levels: int,
        name: str = None,
        min_value: float = 0.0,
        max_value: float = 255.0,
        trainable: bool = False,
        filters: int = 32,
        activation: str = "relu",
        kernel_regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER):
    """
    Merge laplacian pyramid stages and then denormalize
    """

    def _denormalize(args):
        """
        Convert input [-1, +1] to [v0, v1] range
        """
        y, v0, v1 = args
        return K.clip(
            (y + 1.0) * (v1 - v0) / 2.0 + v0,
            min_value=v0,
            max_value=v1)

    # prepare input layers for each level
    input_layers = [
        keras.Input(shape=input_dims[i])
        for i in range(levels)
    ]

    output_layer = None

    for i in range(levels - 1, -1, -1):
        if i == levels - 1:
            output_layer = input_layers[i]
        else:
            x = \
                keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="bilinear")(output_layer)
            if trainable:
                # add conditional
                x = keras.layers.Concatenate()(
                    [x, input_layers[i]])
                # mixing
                x = keras.layers.Conv2D(
                    filters=filters,
                    strides=(1, 1),
                    padding="same",
                    kernel_size=(3, 3),
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer)(x)
                # retargeting
                x = keras.layers.Conv2D(
                    use_bias=False,
                    strides=(1, 1),
                    padding="same",
                    kernel_size=(1, 1),
                    activation="tanh",
                    filters=input_dims[i][-1],
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer)(x)
            output_layer = \
                keras.layers.Add()([x, input_layers[i]])

    # bring bang to initial value range
    output_denormalize_layer = \
        keras.layers.Lambda(
            _denormalize,
            name="denormalize")(
            [output_layer, min_value, max_value])

    return \
        keras.Model(
            name=name,
            inputs=input_layers,
            outputs=output_denormalize_layer)


# ==============================================================================


def attenuate_activation(
        input_layer,
        multiplier: float = DEFAULT_ATTENUATION_MULTIPLIER):
    """
    Transform an input to [0, 1] range
    """
    return (keras.activations.tanh(input_layer * multiplier) + 1.0) / 2.0


# ==============================================================================


def excite_inhibit_spatial_mask_block(
        input_layer,
        filters: int = 32,
        kernel_size=(3, 3),
        flatten: bool = False,
        add_batchnorm: bool = False,
        first_level_activation: str = "relu",
        second_level_activation: str = "sigmoid",
        multiplier: float = DEFAULT_ATTENUATION_MULTIPLIER,
        kernel_regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    """
    Compute a channel a differentiable spatial mask for the input layer
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")

    # --- infer shape
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("works only on 4d tensors")

    if flatten:
        channels = 1
    else:
        channels = shape[channels_index]

    # --- initialize parameters
    first_level_params = dict(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation=first_level_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    second_level_params = dict(
        filters=channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=second_level_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # --- excite
    x_e = keras.layers.Conv2D(**first_level_params)(input_layer)
    if add_batchnorm:
        x_e = keras.layers.BatchNormalization()(x_e)
    x_e = keras.layers.Conv2D(**second_level_params)(x_e)

    # --- inhibit
    x_i = keras.layers.Conv2D(**first_level_params)(input_layer)
    if add_batchnorm:
        x_i = keras.layers.BatchNormalization()(x_i)
    x_i = keras.layers.Conv2D(**second_level_params)(x_i)

    # --- excite vs inhibit, then attenuate
    x = x_e - x_i
    return attenuate_activation(x, multiplier=multiplier)


# ==============================================================================


def excite_inhibit_channel_mask_block(
        input_layer,
        filters: int = 32,
        kernel_size=(3, 3),
        shared: bool = True,
        add_batchnorm: bool = False,
        first_level_activation: str = "linear",
        second_level_activation: str = "sigmoid",
        multiplier: float = DEFAULT_ATTENUATION_MULTIPLIER,
        kernel_regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    """
    Compute a channel a differentiable channel mask for the input layer
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")

    # --- infer shape
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("works only on 4d tensors")

    channels = shape[channels_index]

    # --- initialize parameters
    first_level_params = dict(
        strides=(1, 1),
        padding="same",
        filters=filters,
        kernel_size=kernel_size,
        activation=first_level_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    second_level_params = dict(
        units=channels,
        activation=second_level_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    if shared:
        # --- compute shared
        x = keras.layers.Conv2D(**first_level_params)(input_layer)
        x = keras.layers.GlobalMaxPool2D()(x)
        if add_batchnorm:
            x = keras.layers.BatchNormalization()(x)

        # --- excite
        x_e = keras.layers.Dense(**second_level_params)(x)

        # --- inhibit
        x_i = keras.layers.Dense(**second_level_params)(x)
    else:
        # --- excite
        x_e = keras.layers.Conv2D(**first_level_params)(input_layer)
        x_e = keras.layers.GlobalMaxPool2D()(x_e)
        if add_batchnorm:
            x_e = keras.layers.BatchNormalization()(x_e)
        x_e = keras.layers.Dense(**second_level_params)(x_e)

        # --- inhibit
        x_i = keras.layers.Conv2D(**first_level_params)(input_layer)
        x_i = keras.layers.GlobalMaxPool2D()(x_i)
        if add_batchnorm:
            x_i = keras.layers.BatchNormalization()(x_i)
        x_i = keras.layers.Dense(**second_level_params)(x_i)

    # --- excite vs inhibit, then attenuate
    x = x_e - x_i

    return attenuate_activation(x, multiplier=multiplier)


# ==============================================================================


def excite_inhibit_block(
        input_layer,
        filters: int = 32,
        kernel_size: tuple = (3, 3),
        kernel_regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    # --- infer shape
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("works only on 4d tensors")
    channels = shape[channels_index]

    # --- create spatial and channel mask
    spatial_mask = \
        excite_inhibit_spatial_mask_block(
            input_layer=input_layer,
            filters=filters,
            kernel_size=kernel_size,
            channels_index=channels_index)

    channel_mask = \
        excite_inhibit_channel_mask_block(
            input_layer=input_layer,
            filters=filters,
            kernel_size=kernel_size,
            channels_index=channels_index)

    masked_input = \
        keras.layers.Multiply()([input_layer, spatial_mask])

    masked_input = \
        keras.layers.Multiply()([masked_input, channel_mask])

    # --- operate on input
    x = \
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)(masked_input)

    # --- bring back to original channels
    x = \
        keras.layers.Conv2D(
            filters=channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)(x)

    # --- mask output
    x = keras.layers.Multiply()([spatial_mask, x])

    return x


# ==============================================================================


def squeeze_excite_block(
        input_layer,
        squeeze_units: int = -1,
        use_batchnorm: bool = False,
        prefix="squeeze_excite_",
        initializer=DEFAULT_KERNEL_INITIALIZER,
        regularizer=DEFAULT_KERNEL_REGULARIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("works only on 4d tensors")
    channels = shape[channels_index]
    if squeeze_units is None or squeeze_units <= 0:
        squeeze_units = channels

    # --- squeeze
    x = keras.layers.GlobalAveragePooling2D(
        name=prefix + "avg_pool")(input_layer)

    x = keras.layers.Dense(
        units=squeeze_units,
        name=prefix + "dense0",
        activation="relu",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm0")(x)

    x = keras.layers.Dense(
        units=channels,
        name=prefix + "dense1",
        activation="hard_sigmoid",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)

    # --- mask channels
    x = keras.layers.Multiply(
        name=prefix + "multiply")([x, input_layer])

    return x


# ==============================================================================


def mobilenetV2_block(
        input_layer,
        filters: int = 32,
        dropout_ratio: float = DEFAULT_DROPOUT_RATIO,
        use_batchnorm: bool = False,
        prefix: str = "mobilenetV2_",
        initializer=DEFAULT_KERNEL_INITIALIZER,
        regularizer=DEFAULT_KERNEL_REGULARIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
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
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    if dropout_ratio is not None:
        if dropout_ratio > 1.0 or dropout_ratio < 0.0:
            raise ValueError("Dropout ration must be [0, 1]")

    # --- build block
    previous_no_filters = K.int_shape(input_layer)[channels_index]

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="linear",
        name=prefix + "conv0",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
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
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name=prefix + "conv2",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm1")(x)

    # --- build skip layer and main
    x = keras.layers.Add(name=prefix + "add")([
        x,
        input_layer
    ])

    if dropout_ratio is not None and dropout_ratio > 0.0:
        x = keras.layers.Dropout(
            name=prefix + "dropout",
            rate=dropout_ratio)(x)

    return x


# ==============================================================================


def mobilenetV3_block(
        input_layer,
        filters: int = 32,
        squeeze_units: int = -1,
        activation: str = "relu",
        dropout_ratio: float = None,
        use_batchnorm: bool = False,
        prefix: str = "mobilenetV3_",
        regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        initializer: str = DEFAULT_KERNEL_INITIALIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    """
    Build a mobilenet V3 block is bottleneck with residual + squeeze and excite

    :param input_layer:
    :param filters:
    :param initializer:
    :param regularizer:
    :param activation:
    :param squeeze_units:
    :param prefix:
    :param channels_index:
    :param use_batchnorm:
    :param dropout_ratio:

    :return: mobilenet V3 block
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    if dropout_ratio is not None:
        if dropout_ratio > 1.0 or dropout_ratio < 0.0:
            raise ValueError("Dropout ration must be [0, 1]")
    previous_no_filters = K.int_shape(input_layer)[channels_index]

    # --- build block
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "conv0",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "conv1",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer,
        depthwise_initializer=initializer,
        depthwise_regularizer=regularizer)(x)

    x = \
        squeeze_excite_block(
            x,
            squeeze_units=squeeze_units,
            regularizer=regularizer,
            initializer=initializer,
            use_batchnorm=True,
            prefix=prefix + "squeeze_excite_")

    x = \
        keras.layers.Conv2D(
            filters=previous_no_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="linear",
            name=prefix + "conv2",
            kernel_regularizer=regularizer,
            kernel_initializer=initializer)(x)

    # --- build skip layer and main
    x = \
        keras.layers.Add(name=prefix + "add")([
            x,
            input_layer
        ])

    if dropout_ratio is not None and dropout_ratio > 0.0:
        x = keras.layers.Dropout(
            name=prefix + "dropout",
            rate=dropout_ratio)(x)

    return x


# ==============================================================================


def attention_block(
        input_layer,
        filters=32,
        kernel_size=(1, 1),
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
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    shape = K.int_shape(input_layer)
    if len(shape) != 4:
        raise ValueError("only supports 4d tensors")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    padding = "same"
    strides = (1, 1)

    # --- theta, phi, g
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

    # --- build block
    h_x_w = np.prod(shape[1:3])
    new_shape = (h_x_w, filters)
    theta_flat = keras.layers.Reshape(new_shape)(theta)
    phi_flat = keras.layers.Reshape(new_shape)(phi)
    phi_flat = keras.layers.Permute((2, 1))(phi_flat)
    theta_x_phi = keras.layers.Dot(axes=(1, 2))([theta_flat, phi_flat])
    theta_x_phi = keras.layers.Softmax()(theta_x_phi)
    g_flat = keras.layers.Reshape(new_shape)(g)

    # --- multiply with attention map
    theta_x_phi_xg = keras.layers.Dot(axes=(1, 2))([theta_x_phi, g_flat])
    theta_x_phi_xg = keras.layers.Reshape(
        (shape[1:3] + (filters,)))(theta_x_phi_xg)

    return theta_x_phi_xg


# ==============================================================================


def self_attention_block(
        input_layer,
        filters=32,
        kernel_size=[1, 1],
        activation="linear",
        initializer="glorot_normal",
        regularizer=None,
        prefix="self_attention_",
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    """
    Builds a self-attention block, attention block plus residual skip
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param activation:
    :param initializer:
    :param regularizer:
    :param prefix:
    :param channels_index:
    :return:
    """

    shape = K.int_shape(input_layer)
    previous_no_filters = shape[channels_index]

    # --- build attention block
    attention = \
        attention_block(
            input_layer=input_layer,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            initializer=initializer,
            regularizer=regularizer,
            prefix=prefix)

    # --- convolve to match output channels
    attention = \
        keras.layers.Conv2D(
            filters=previous_no_filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            activation=activation,
            name=prefix + "result",
            kernel_regularizer=regularizer,
            kernel_initializer=initializer)(attention)

    # --- residual connection with input
    return keras.layers.Add()([attention, input_layer])


# ==============================================================================


def resnet_block(
        input_layer,
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        dropout_ratio=DEFAULT_DROPOUT_RATIO,
        use_batchnorm=False,
        prefix="resnet_",
        initializer=DEFAULT_KERNEL_INITIALIZER,
        regularizer=DEFAULT_KERNEL_REGULARIZER,
        channels_index: int = DEFAULT_CHANNEL_INDEX):
    """
    Build a resnet block
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param activation:
    :param initializer:
    :param regularizer:
    :param prefix:
    :param channels_index:
    :param use_batchnorm:
    :param dropout_ratio:
    :return: Resnet block
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be empty")
    if filters <= 0:
        raise ValueError("Filters should be > 0")
    if dropout_ratio is not None:
        if dropout_ratio > 1.0 or dropout_ratio < 0.0:
            raise ValueError("Dropout ration must be [0, 1]")

    # --- build block
    previous_no_filters = K.int_shape(input_layer)[channels_index]

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        activation=activation,
        name=prefix + "conv0",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(input_layer)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation="linear",
        name=prefix + "conv1",
        kernel_regularizer=regularizer,
        kernel_initializer=initializer)(x)

    # adjust strides
    if strides != (1, 1):
        input_layer = \
            keras.layers.MaxPooling2D(
                pool_size=tuple(s + 1 for s in strides),
                padding="same",
                strides=strides)(input_layer)

    # --- build skip layer and main
    if previous_no_filters == filters:
        tmp_layer = \
            keras.layers.Layer(
                name=prefix + "skip")(input_layer)
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

    x = \
        keras.layers.Add(name=prefix + "add")([x, tmp_layer])

    x = keras.layers.Activation(
        activation,
        name=prefix + "activation")(x)

    if dropout_ratio is not None and dropout_ratio > 0.0:
        x = keras.layers.Dropout(
            name=prefix + "dropout",
            rate=dropout_ratio)(x)

    if use_batchnorm:
        x = keras.layers.BatchNormalization(
            name=prefix + "batchnorm")(x)

    return x


# ==============================================================================


def basic_block(
        input_layer,
        block_type="encoder",
        filters=[64],
        kernel_size=[(3, 3)],
        strides=[(1, 1)],
        initializer: str = DEFAULT_KERNEL_INITIALIZER,
        regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        prefix: str = "block_", ):
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
        prefix_i = f"{prefix}_{i}_"
        params = dict(
            padding=padding,
            strides=strides[i],
            filters=filters[i],
            activation=activation,
            kernel_size=kernel_size[i],
            kernel_initializer=initializer,
            kernel_regularizer=regularizer)

        # --- handle subsampling and change in the number of filters
        if strides[i][0] != 1 or strides[i][1] != 1 or \
                filters[i] != previous_no_filters:
            if block_type == "encoder":
                x = keras.layers.Conv2D(**params)(x)
            elif block_type == "decoder":
                x = keras.layers.Conv2DTranspose(**params)(x)
            else:
                raise ValueError("don't know how to parse {0}".format(block_type))

        x = \
            mobilenetV3_block(
                x,
                filters=filters[i],
                initializer=initializer,
                regularizer=regularizer,
                use_batchnorm=use_batchnorm,
                prefix=prefix_i + "mobilenetV3_")

        if use_dropout:
            x = keras.layers.Dropout(
                name=prefix_i + "dropout", rate=0.1)(x)

        if use_batchnorm:
            x = keras.layers.BatchNormalization(
                name=prefix_i + "batchnorm")(x)

        previous_no_filters = filters[i]

    return x


# ==============================================================================


def gaussian_kernel(
        size,
        nsig):
    """
    Returns a 2D Gaussian kernel array
    """
    assert len(nsig) == 2
    assert len(size) == 2
    kern1d = []
    for i in range(2):
        x = \
            np.linspace(
                start=-np.abs(nsig[i]),
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


def gaussian_filter_block(
        input_layer,
        kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        strides=(1, 1),
        dilation_rate=(1, 1),
        padding="same",
        xy_max=DEFAULT_GAUSSIAN_XY_MAX):
    """
    Build a gaussian filter block as non trainable depth wise
    convolution filter with fixed weights

    :param input_layer:
    :param kernel_size:
    :param strides:
    :param padding:
    :param xy_max:
    :param dilation_rate:
    :return:
    """

    # --- initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        single_channel_kernel = \
            gaussian_kernel(
                [shape[0], shape[1]],
                xy_max)
        for i in range(shape[2]):
            kernel[:, :, i, 0] = single_channel_kernel
        return kernel

    return \
        keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=1,
            dilation_rate=dilation_rate,
            activation="linear",
            use_bias=False,
            trainable=False,
            depthwise_initializer=kernel_init,
            kernel_initializer=kernel_init)(input_layer)


# ==============================================================================


def tensor_to_target_encoding_thinning(
        input_layer,
        target_encoding_size: int,
        kernel_regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER,
        prefix: str = "encoding_thinning"):
    # --- argument checking
    input_shape = K.int_shape(input_layer)
    if len(input_shape) != 4:
        raise ValueError("input_layer must be 4d tensor")

    # --- setup variables
    x = input_layer
    _, height, width, channels = input_shape
    iteration = 0

    # --- iteratively thin the tensor
    while height >= 3 and width >= 3:
        x = \
            resnet_block(
                x,
                filters=channels,
                strides=(2, 2),
                kernel_size=(3, 3),
                prefix=f"{prefix}_{iteration}",
                initializer=kernel_initializer,
                regularizer=kernel_regularizer)
        _, height, width, channels = K.int_shape(x)
        iteration += 1

    # target output number of channels
    if target_encoding_size > 0:
        x = keras.layers.Conv2D(
            filters=target_encoding_size,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer)(x)

    shape_before_flattening = K.int_shape(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    return x, shape_before_flattening


# ==============================================================================


def decoding_tensor_to_target_fattening(
        input_layer,
        target_decoding_size: tuple,
        kernel_regularizer: str = DEFAULT_KERNEL_REGULARIZER,
        kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER,
        prefix: str = "decoding_fattening"):
    # --- argument checking
    input_shape = K.int_shape(input_layer)
    if len(input_shape) != 2:
        raise ValueError("input_layer must be 2d tensor")

    # --- setup variables
    _, channels = input_shape
    target_channels = target_decoding_size[2]
    x = keras.layers.Dense(
        units=target_channels,
        activation="linear",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)(input_layer)
    height = 1
    width = 1
    x = keras.layers.Reshape(target_shape=(1, 1, target_channels))(x)

    # --- iteratively thin the tensor
    while height <= 3 and width <= 3:
        channels = channels * 2

        x = \
            resnet_block(
                x,
                filters=channels,
                strides=(2, 2),
                kernel_size=(3, 3),
                prefix=f"{prefix}_{channels}",
                initializer=kernel_initializer,
                regularizer=kernel_regularizer)
        _, height, width, channels = K.int_shape(x)

    return x

# ==============================================================================

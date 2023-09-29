import os
import PIL
import json
import copy
import pathlib
import itertools
import numpy as np
from enum import Enum
import tensorflow as tf
from pathlib import Path
import tensorflow_addons as tfa
from typing import List, Tuple, Union, Dict, Iterable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_layers import Mish
from .custom_logger import logger


# ---------------------------------------------------------------------


def clip_normalized_tensor(
        input_tensor: tf.Tensor) -> tf.Tensor:
    """
    clip an input to [-0.5, +0.5]

    :param input_tensor:
    :return:
    """
    return \
        tf.clip_by_value(
            input_tensor,
            clip_value_min=-0.5,
            clip_value_max=+0.5)


# ---------------------------------------------------------------------


def clip_unnormalized_tensor(
        input_tensor: tf.Tensor) -> tf.Tensor:
    """
    clip an input to [0.0, 255.0]

    :param input_tensor:
    :return:
    """
    return \
        tf.clip_by_value(
            input_tensor,
            clip_value_min=0.0,
            clip_value_max=255.0)


# ---------------------------------------------------------------------


def load_config(
        config: Union[str, Dict, Path]) -> Dict:
    """
    Load configuration from multiple sources

    :param config: dict configuration or path to json configuration
    :return: dictionary configuration
    """
    try:
        if config is None:
            raise ValueError("config should not be empty")
        if isinstance(config, Dict):
            return config
        if isinstance(config, str) or isinstance(config, Path):
            if not os.path.isfile(str(config)):
                return ValueError(
                    "configuration path [{0}] is not valid".format(
                        str(config)
                    ))
            with open(str(config), "r") as f:
                return json.load(f)
        raise ValueError("don't know how to handle config [{0}]".format(config))
    except Exception as e:
        logger.error(e)
        raise ValueError(f"failed to load [{config}]")


# ---------------------------------------------------------------------


def save_config(
        config: Union[str, Dict, Path],
        filename: Union[str, Path]) -> None:
    """
    save configuration to target filename

    :param config: dict configuration or path to json configuration
    :param filename: output filename
    :return: nothing if success, exception if failed
    """
    # --- argument checking
    config = load_config(config)
    if not filename:
        raise ValueError("filename cannot be null or empty")

    # --- log
    logger.info(f"saving configuration pipeline to [{str(filename)}]")

    # --- dump config to filename
    with open(filename, "w") as f:
        return json.dump(obj=config, fp=f, indent=4)

# ---------------------------------------------------------------------


def color_mode_to_channels(
        color_mode: str) -> int:
    color_mode = color_mode.strip().lower()
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )
    return num_channels


# ---------------------------------------------------------------------


def input_shape_fixer(
        input_shape: Iterable):
    for i, shape in enumerate(input_shape):
        if shape == "?" or \
                shape == "" or \
                shape == "-1":
            input_shape[i] = None
    return input_shape


# ---------------------------------------------------------------------


def merge_iterators(
        *iterators):
    """
    Merge different iterators together

    :param iterators:
    """
    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value


# ---------------------------------------------------------------------


def distance_from_center_layer(
        input_layer) -> tf.Tensor:
    """
    add 1 extra channel containing the distance from center

    :param input_layer:
    :return:
    """
    shape = tf.shape(input_layer)
    width = shape[2]
    height = shape[1]

    # ---
    x_grid = tf.linspace(start=0.0, stop=1.0, num=width)
    y_grid = tf.linspace(start=0.0, stop=1.0, num=height)
    xx_grid, yy_grid = \
        tf.meshgrid(x_grid, y_grid)
    dd_grid = \
        tf.sqrt(
            tf.square(xx_grid - 0.5) +
            tf.square(yy_grid - 0.5)
        )
    dd_grid = tf.expand_dims(dd_grid, axis=0)
    dd_grid = tf.expand_dims(dd_grid, axis=3)
    # repeat in batch dim
    dd_grid = tf.repeat(dd_grid, axis=0, repeats=shape[0])
    return dd_grid

# ---------------------------------------------------------------------


class ConvType(Enum):
    CONV2D = 0

    CONV2D_DEPTHWISE = 1

    CONV2D_TRANSPOSE = 2

    CONV2D_SEPARABLE = 3

    @staticmethod
    def from_string(type_str: str) -> "ConvType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ConvType[type_str]

    def to_string(self) -> str:
        return self.name


def conv2d_wrapper(
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None,
        ln_params: Dict = None,
        pre_activation: str = None,
        bn_post_params: Dict = None,
        ln_post_params: Dict = None,
        dropout_params: Dict = None,
        dropout_2d_params: Dict = None,
        conv_type: Union[ConvType, str] = ConvType.CONV2D):
    """
    wraps a conv2d with a preceding normalizer

    if bn_post_params force a conv(linear)->bn->activation setup

    :param input_layer: the layer to operate on
    :param conv_params: conv2d parameters
    :param bn_params: batchnorm parameters before the conv, None to disable bn
    :param ln_params: layer normalization parameters before the conv, None to disable ln
    :param pre_activation: activation after the batchnorm, None to disable
    :param bn_post_params: batchnorm parameters after the conv, None to disable bn
    :param ln_post_params: layer normalization parameters after the conv, None to disable ln
    :param dropout_params: dropout parameters after the conv, None to disable it
    :param dropout_2d_params: dropout parameters after the conv, None to disable it
    :param conv_type: if true use depthwise convolution,

    :return: transformed input
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if conv_params is None:
        raise ValueError("conv_params cannot be None")

    # --- prepare arguments
    use_ln = ln_params is not None
    use_bn = bn_params is not None
    use_bn_post = bn_post_params is not None
    use_ln_post = ln_post_params is not None
    use_dropout = dropout_params is not None
    use_dropout_2d = dropout_2d_params is not None
    use_pre_activation = pre_activation is not None
    conv_params = copy.deepcopy(conv_params)
    conv_activation = conv_params.get("activation", "linear")
    conv_params["activation"] = "linear"

    if conv_params.get("use_bias", True) and \
            (conv_activation == "relu" or conv_activation == "relu6") and \
            not (use_bn_post or use_ln_post):
        conv_params["bias_initializer"] = \
            tf.keras.initializers.Constant(DEFAULT_RELU_BIAS)

    # TODO restructure this
    if isinstance(conv_type, str):
        conv_type = ConvType.from_string(conv_type)
    if "depth_multiplier" in conv_params:
        if conv_type != ConvType.CONV2D_DEPTHWISE:
            logger.info("Changing conv_type to CONV2D_DEPTHWISE because it contains depth_multiplier argument "
                        f"[conv_params[\'depth_multiplier\']={conv_params['depth_multiplier']}]")
        conv_type = ConvType.CONV2D_DEPTHWISE
    if "dilation_rate" in conv_params:
        if conv_type != ConvType.CONV2D_TRANSPOSE:
            logger.info("Changing conv_type to CONV2D_TRANSPOSE because it contains dilation argument "
                        f"[conv_params[\'dilation_rate\']={conv_params['dilation_rate']}]")
        conv_type = ConvType.CONV2D_TRANSPOSE

    # --- set up stack of operation
    x = input_layer

    # --- perform pre convolution normalizations and activation
    if use_bn:
        x = tf.keras.layers.BatchNormalization(**bn_params)(x)
    if use_ln:
        x = tf.keras.layers.LayerNormalization(**ln_params)(x)
    if use_pre_activation:
        x = tf.keras.layers.Activation(pre_activation)(x)

    # --- convolution
    if conv_type == ConvType.CONV2D:
        x = tf.keras.layers.Conv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_DEPTHWISE:
        x = tf.keras.layers.DepthwiseConv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_TRANSPOSE:
        x = tf.keras.layers.Conv2DTranspose(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_SEPARABLE:
        x = tf.keras.layers.SeparableConv2D(**conv_params)(x)
    else:
        raise ValueError(f"don't know how to handle this [{conv_type}]")

    # --- dropout
    if use_dropout:
        x = tf.keras.layers.Dropout(**dropout_params)(x)

    if use_dropout_2d:
        x = tf.keras.layers.SpatialDropout2D(**dropout_2d_params)(x)

    # --- perform post convolution normalizations and activation
    if use_bn_post:
        x = tf.keras.layers.BatchNormalization(**bn_post_params)(x)
    if use_ln_post:
        x = tf.keras.layers.LayerNormalization(**ln_post_params)(x)

    if conv_activation.lower() in ["mish"]:
        # Mish: A Self Regularized Non-Monotonic Activation Function (2020)
        x = Mish()(x)
    elif conv_activation.lower() in ["leaky_relu", "leakyrelu"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    elif conv_activation.lower() in ["leaky_relu_001", "leakyrelu_001"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    elif conv_activation.lower() in ["prelu"]:
        # parametric Rectified Linear Unit
        constraint = \
            tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        x = tf.keras.layers.PReLU(
            alpha_initializer=0.1,
            # very small l1
            alpha_regularizer=tf.keras.regularizers.l1(0.001),
            alpha_constraint=constraint,
            shared_axes=[1, 2])(x)
    elif conv_activation.lower() in ["linear"]:
        # do nothing
        pass
    else:
        x = tf.keras.layers.Activation(conv_activation)(x)

    return x


# ---------------------------------------------------------------------


def layer_denormalize(
        input_layer: tf.Tensor,
        v_min: float,
        v_max: float) -> tf.Tensor:
    """
    Convert input [-0.5, +0.5] to [v0, v1] range
    """
    y_clip = clip_normalized_tensor(input_layer)
    return (y_clip + 0.5) * (v_max - v_min) + v_min


# ---------------------------------------------------------------------


def layer_normalize(
        input_layer: tf.Tensor,
        v_min: float,
        v_max: float) -> tf.Tensor:
    """
    Convert input from [v0, v1] to [-0.5, +0.5] range
    """
    y_clip = \
        tf.clip_by_value(
            t=input_layer,
            clip_value_min=v_min,
            clip_value_max=v_max)
    return (y_clip - v_min) / (v_max - v_min) - 0.5


# ---------------------------------------------------------------------


def build_normalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "normalize") -> keras.Model:
    """
    Wrap a normalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model

    :return: normalization model
    """
    model_input = tf.keras.Input(shape=input_dims)

    # --- normalize input
    # from [min_value, max_value] to [-0.5, +0.5]
    model_output = \
        layer_normalize(
            input_layer=model_input,
            v_min=float(min_value),
            v_max=float(max_value))

    # --- wrap model
    return tf.keras.Model(
        name=name,
        trainable=False,
        inputs=model_input,
        outputs=model_output)


# ---------------------------------------------------------------------


def build_denormalize_model(
        input_dims,
        min_value: float = 0.0,
        max_value: float = 255.0,
        name: str = "denormalize") -> tf.keras.Model:
    """
    Wrap a denormalize layer in a model

    :param input_dims: Models input dimensions
    :param min_value: Minimum value
    :param max_value: Maximum value
    :param name: name of the model

    :return: denormalization model
    """
    model_input = tf.keras.Input(shape=input_dims)

    # --- denormalize input
    # from [-0.5, +0.5] to [v0, v1] range
    model_output = \
        layer_denormalize(
            input_layer=model_input,
            v_min=float(min_value),
            v_max=float(max_value))

    # --- wrap model
    return \
        tf.keras.Model(
            name=name,
            trainable=False,
            inputs=model_input,
            outputs=model_output)


# ---------------------------------------------------------------------


def crop_from_middle(
        input_batch: tf.Tensor,
        crop_size: Tuple[int, int] = (128, 128)) -> tf.Tensor:
    # TODO rewrite this to be more robust
    shape = tf.shape(input_batch)
    width = shape[1]
    height = shape[2]
    start_width = tf.cast((width - crop_size[0]) / 2, tf.int32)
    start_height = tf.cast((height - crop_size[1]) / 2, tf.int32)
    if width > crop_size[0]:
        input_batch = input_batch[:, start_width:start_width + crop_size[0], :, :]
    if height > crop_size[1]:
        input_batch = input_batch[:, :, start_height:start_height + crop_size[1], :]
    return input_batch


# ---------------------------------------------------------------------


def downsample(
        input_batch: tf.Tensor,
        sigma: float = 1.0,
        filter_shape: Tuple[int, int] = (5, 5),
        strides: Tuple[int, int] = (2, 2)) -> tf.Tensor:
    x = \
        tfa.image.gaussian_filter2d(
            sigma=sigma,
            image=input_batch,
            filter_shape=filter_shape)
    return \
        tf.nn.max_pool2d(
            input=x,
            ksize=(1, 1),
            strides=strides,
            padding="SAME")


# ---------------------------------------------------------------------


def clean_image(
        t: tf.Tensor,
        threshold_band: Tuple[float, float] = (247.0, 255.0),
        filter_shape: Tuple[int, int] = (7, 7),
        iterations: int = 3) -> tf.Tensor:
    """
    inpainting of values based on diffusion process
    """
    # --- assertions
    tf.assert_rank(x=t,
                   rank=4,
                   message=f"t [{tf.shape(t)}] must be 4d")

    t_f = tf.cast(t, dtype=tf.float32)
    mask = \
        tf.cast(
            tf.math.logical_and(
                t_f >= threshold_band[0],
                t_f <= threshold_band[1]),
            dtype=tf.float32)

    mask = \
        tf.nn.max_pool2d(
            mask,
            ksize=(3, 3),
            strides=(1, 1),
            padding="SAME")
    result = tf.identity(t_f)

    # diffuse surroundings into the mask
    # without affecting anything outside the mask
    for i in range(iterations):
        result = \
            tf.nn.avg_pool2d(
                result,
                ksize=filter_shape,
                strides=(1, 1),
                padding="SAME")
        result = \
            (result * mask) + \
            (t_f * (1.0 - mask))

    return tf.cast(result, dtype=t.dtype)


# ---------------------------------------------------------------------


def fix_broken_png(
        input_filename: Union[str, pathlib.Path],
        output_filename: Union[str, pathlib.Path] = None,
        verbose: bool = False) -> bool:
    """
    load a png file and fix its structure so it can be read by tensorflow

    :param input_filename: input filename of image
    :param output_filename: optional output filename, if None save to original
    :param verbose: if True show extra messages
    :return: True if operation success, False otherwise
    """
    try:
        # --- argument check
        if input_filename is None:
            raise ValueError("input_filename cannot be None")

        if verbose:
            logger.info(f"processing [{str(input_filename)}]")

        # open the file
        x = PIL.Image.open(str(input_filename))

        # convert to np array
        x = np.array(x)

        # convert back to Image
        x = PIL.Image.fromarray(np.uint8(x))

        # save
        if output_filename is None:
            x.save(input_filename)
        else:
            if verbose:
                logger.info(f"saving to [{str(output_filename)}]")
            # create directory if required
            directory = os.path.split(str(output_filename))[0]
            if not os.path.isdir(directory):
                p = pathlib.Path(directory)
                p.mkdir(parents=True, exist_ok=True)
            # save file
            x.save(str(output_filename))
        return True
    except Exception as e:
        logger.error(f"failed to fix [{input_filename}] -> {e}")
    return False


# ---------------------------------------------------------------------


def one_hot_to_color(
        t: tf.Tensor,
        normalize_probabilities: bool = True,
        normalize: bool = False) -> tf.Tensor:
    """
    convert one hot tensor to colored tensor image for better visualization
    assumes that all numbers are positive

    :param t: [B, H, W, Classes]
    :param normalize_probabilities: if True normalize probabilities for each xy so it adds up to 1
    :param normalize: if True scale from 0..255 to 0..1
    :return:  [B, H, W, 3]
    """
    # --- assertions
    tf.assert_rank(x=t,
                   rank=4,
                   message=f"t [{tf.shape(t)}] must be 4d")

    # tf.assert_equal(
    #     x=tf.shape(t)[-1],
    #     y=len(MATERIAL_COLORS2LABEL),
    #     message=f"t [{tf.shape(t)}] must have as many channels as materials")

    # --- normalize probabilities
    if normalize_probabilities:
        t_sum = tf.reduce_sum(t, axis=-1, keepdims=True)
        t = tf.divide(t, t_sum + DEFAULT_EPSILON)

    # --- set variables
    shape = t.shape[0:3] + (3,)
    result = tf.zeros(shape=shape, dtype=tf.float32)

    # --- iterate input
    items = tf.unstack(t, axis=3)
    for i, item in enumerate(items):
        x = tf.stack([item, item, item], axis=3)
        result += x * MATERIAL_LABEL2COLORS[i]
    if normalize:
        result = result / 255
    return result


# ---------------------------------------------------------------------


def colorize_tensor_hard(
        t: tf.Tensor,
        normalize: bool = False) -> tf.Tensor:
    """
    convert one hot tensor to colored tensor image for better visualization

    :param t: tensor of shape [B, H, W, Classes]
    :param normalize: if True scale from 0..255 to 0..1
    :return:  [B, H, W, 3]
    """
    # --- assertions
    # tf.assert_equal(
    #     x=tf.shape(t)[-1],
    #     y=len(MATERIAL_COLORS2LABEL),
    #     message=f"t [{tf.shape(t)}] must have as many channels as materials")

    t = class_prob_to_one_hot(t)
    return one_hot_to_color(t, normalize=normalize)


# ---------------------------------------------------------------------


def class_prob_to_one_hot(
        t: tf.Tensor) -> tf.Tensor:
    """
    replaces the top value in last dim with 1 and the rest with 0

    :param t: one hot tensor to operator on
    :return: tensor same shape x but with only zeros and ones
    """
    # --- get maximum value on the channels axis
    max_val = tf.math.reduce_max(t, axis=-1, keepdims=True)

    # --- set to True and the cast to float
    return tf.cast(tf.greater_equal(t, max_val), tf.float32)


# ---------------------------------------------------------------------


def gray_to_one_hot(
        t: tf.Tensor,
        round_vales: bool = True,
        out_dtype: tf.dtypes = tf.float32,
        class_values: Iterable[int] = MATERIAL_COLORS2LABEL.values()) -> tf.Tensor:
    """
    converts a 4d [B,W,H,1] color tensor
    to a 4d [B,W,H,classes] one hot tensor,

    :param t: categorical tensor to operate on
    :param round_vales:
    :param out_dtype:
    :param class_values: list of available classes (defaults to all)

    :return: one hot tensor
    """
    # --- assertions
    tf.assert_rank(x=t,
                   rank=4,
                   message=f"t [{tf.shape(t)}] must be 4d")
    tf.assert_equal(
        x=tf.shape(t)[-1],
        y=1,
        message=f"t [{tf.shape(t)}] must have a single channel")

    if round_vales:
        # round so they become integers
        t = tf.round(t)

    results = [
        tf.cast(tf.equal(t, tf.constant(i, dtype=t.dtype)), dtype=out_dtype)
        for i in class_values
    ]

    return tf.concat(values=results, axis=3)


# ---------------------------------------------------------------------


def one_hot_to_gray(
        t: tf.Tensor,
        out_dtype: tf.dtypes = tf.float32) -> tf.Tensor:
    """
    converts a 4d [B,W,H,classes] one hot tensor
    to a 4d [B,W,H,1] image segmentation,

    :param t: one hot tensor to operator on
    :param out_dtype:
    :return: single channel tensor
    """
    # --- assertions
    tf.assert_rank(x=t,
                   rank=4,
                   message="categorical tensor must be 4d")
    tf.assert_equal(
        x=tf.shape(t)[-1],
        y=len(MATERIAL_COLORS2LABEL),
        message=f"tensor must have as many channels [{tf.shape(t)[-1]}] "
                f"as materials [{len(MATERIAL_COLORS2LABEL)}]")

    result = \
        tf.math.argmax(
            t,
            axis=3,
            output_type=tf.dtypes.int32,
            name=None
        )

    return tf.cast(result, dtype=out_dtype)


# ---------------------------------------------------------------------


def pixel_differences(
        y_true_one_hot: tf.Tensor,
        y_pred_one_hot: tf.Tensor,
        keepdims: bool = True,
        output_type: tf.DType = tf.dtypes.float32) -> tf.Tensor:
    """
    highlight differences in the 2 two tensors

    :param y_true_one_hot: 4d one hot tensor
    :param y_pred_one_hot: 4d one hot tensor
    :param keepdims: if True keep dims in the reduction
    :param output_type: output data type
    :return: tensor with the differences
    """
    # --- argument checking
    tf.assert_rank(x=y_true_one_hot,
                   rank=4,
                   message="y_true_one_hot tensor must be 4d")
    tf.assert_rank(x=y_pred_one_hot,
                   rank=4,
                   message="y_pred_one_hot tensor must be 4d")

    # --- keep only highest prob and set it to one
    y_true_one_hot = class_prob_to_one_hot(y_true_one_hot)
    y_pred_one_hot = class_prob_to_one_hot(y_pred_one_hot)

    # --- highlight pixels that differ
    return tf.reduce_max(
        tf.cast(
            tf.math.logical_xor(
                tf.cast(y_true_one_hot, dtype=tf.bool),
                tf.cast(y_pred_one_hot, dtype=tf.bool)
            ),
            dtype=output_type),
        axis=3,
        keepdims=keepdims
    )


# ---------------------------------------------------------------------


def reduce_mean_above_threshold(
        input_tensor: tf.Tensor,
        threshold: float = 0.0,
        axis: List[int] = [1, 2]) -> tf.Tensor:
    """
    similar to reduce mean, but count only everything above threshold

    :param input_tensor: input tensor
    :param threshold: cutoff value for computing mean
    :param axis:
    :return: tensor with the differences

    """
    mask = \
        tf.cast(
            tf.greater(input_tensor,
                       tf.constant(threshold, dtype=tf.float32)),
            dtype=tf.float32)
    non_zero = \
        tf.math.count_nonzero(
            mask,
            axis=axis,
            keepdims=False,
            dtype=tf.float32)
    # sum over axis and divide over non zero elements
    mean_non_zero = \
        tf.reduce_sum(
            tf.multiply(input_tensor, mask),
            keepdims=False,
            axis=axis) / \
        (non_zero + DEFAULT_EPSILON)
    return mean_non_zero

# ---------------------------------------------------------------------


def compute_border(
        input_tensor: tf.Tensor) -> tf.Tensor:
    """
    computes the hard borders for each channel

    :param input_tensor: 4d dim tensor, one hot encoding of image segmentation
    :return: 4d tensor, with border/contour on each channel, tf.bool
    """
    # --- argument checking
    tf.assert_rank(x=input_tensor,
                   rank=4,
                   message="input_tensor tensor must be 4d")

    input_tensor = \
        tf.cast(input_tensor, dtype=tf.float16)

    # 4-D with shape
    # [filter_height, filter_width, in_channels, channel_multiplier]
    kernel = \
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float16)
    kernel = \
        tf.repeat(
            input=np.reshape(kernel, newshape=(3, 3, 1, 1)),
            repeats=tf.shape(input_tensor)[3],
            axis=2)
    result = \
        tf.nn.depthwise_conv2d(
            input=input_tensor,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME")
    result = \
        tf.logical_and(
            tf.greater(input_tensor, 0),
            tf.less(result, 4))
    return result


# ---------------------------------------------------------------------


def soft_border(
        input_tensor: tf.Tensor,
        pool_size: Tuple[int, int] = (3, 3),
        multiplier: float = 1.0) -> tf.Tensor:
    """
    computes the borders for each channel

    :param input_tensor: 4d dim tensor, one hot encoding of image segmentation
    :param pool_size: integer, pool size
    :param multiplier:
    :return: 4d tensor, one hot encoding
    """
    # --- argument checking
    if len(pool_size) != 2:
        raise ValueError(f"pool_size [{pool_size}] should be > 0")

    # --- build
    x = input_tensor
    x_avg = tf.nn.avg_pool2d(
        input=x,
        padding="SAME",
        ksize=pool_size,
        strides=(1, 1))
    return \
        tf.clip_by_value(
            tf.abs(x - x_avg) * float(multiplier),
            clip_value_min=0.0,
            clip_value_max=1.0)

# ---------------------------------------------------------------------


def internal_border_fast(
        input_tensor: tf.Tensor,
        pool_size: Tuple[int, int] = (3, 3)) -> tf.Tensor:
    """
    computes the borders for each channel

    :param input_tensor: 4d dim tensor, one hot encoding of image segmentation
    :param pool_size: integer, pool size
    :return: 4d tensor, one hot encoding
    """
    x = \
        soft_border(
            input_tensor=input_tensor,
            pool_size=pool_size,
            multiplier=float(pool_size[0] * pool_size[1]))
    return x * input_tensor

# ---------------------------------------------------------------------


def smooth_borders(
        input_tensor: tf.Tensor,
        iterations: int = 3,
        kernel_size: Tuple[int, int] = (3, 3)) -> tf.Tensor:
    x = input_tensor

    for i in range(iterations):
        x = \
            tf.keras.layers.AveragePooling2D(
                pool_size=kernel_size,
                strides=(2, 2),
                padding="same")(x)
        x = \
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear")(x)
        x = x / (tf.reduce_sum(x, axis=-1, keepdims=True) + DEFAULT_EPSILON)

    return x

# ---------------------------------------------------------------------


def edge_magnitude(
        input_tensor: tf.Tensor) -> tf.Tensor:
    y_pred_max = \
        tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    y_pred_max = \
        tf.cast(tf.greater_equal(input_tensor, y_pred_max), tf.float32)
    dx, dy = \
        tf.image.image_gradients(y_pred_max)
    return \
        tf.sqrt(
            tf.math.square(dx) +
            tf.math.square(dy) +
            DEFAULT_EPSILON)

# ---------------------------------------------------------------------


def normalize_local(
        input_tensor: tf.Tensor,
        kernel_size: Tuple[int, int] = (9, 9)) -> tf.Tensor:
    """
    calculate window mean per channel and window variance per channel

    :param input_tensor: the layer to operate on
    :param kernel_size: size of the kernel (window)
    :return: normalized input_tensor
    """

    # ---
    local_mean = (
        tf.nn.avg_pool2d(
            input=input_tensor,
            ksize=kernel_size,
            strides=(1, 1),
            padding="SAME"))
    local_diff = tf.square(input_tensor - local_mean)
    local_variance = (
        tf.nn.avg_pool2d(
            input=local_diff,
            ksize=kernel_size,
            strides=(1, 1),
            padding="SAME"))
    local_std = tf.sqrt(local_variance + DEFAULT_EPSILON)
    return input_tensor / (local_std + 1.0)

# ---------------------------------------------------------------------


def logit_norm(
        input_tensor: tf.Tensor,
        t: tf.Tensor = tf.constant(1.0),
        axis: Union[int, Tuple[int, int]] = -1) -> tf.Tensor:
    """
    implementation of logit_norm based on

    Mitigating Neural Network Overconfidence with Logit Normalization
    """
    x = input_tensor
    x_denominator = tf.square(x)
    x_denominator = tf.reduce_sum(x_denominator, axis=axis, keepdims=True)
    x_denominator = tf.sqrt(x_denominator + DEFAULT_EPSILON) + DEFAULT_EPSILON
    return x / (x_denominator * t)

# ---------------------------------------------------------------------


def norm_softmax(
        input_tensor: tf.Tensor) -> tf.Tensor:
    """
    implementation of normsoftmax

    NORM SOFTMAX : NORMALIZE THE INPUT
    OF SOFTMAX TO ACCELERATE AND STABILIZE TRAINING
    """
    x = input_tensor
    x_ones = x * 0.0 + 1.0
    x_sum_dims = tf.reduce_sum(x_ones, axis=-1, keepdims=True)
    x_sum_dims = tf.pow(x_sum_dims, -0.5)
    x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    return logit_norm(x - x_mean, x_sum_dims)

# ---------------------------------------------------------------------

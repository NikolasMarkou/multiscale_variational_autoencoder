import os
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
        name: str = "normalize") -> tf.keras.Model:
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


def random_crops(
        input_batch: tf.Tensor,
        no_crops_per_image: int = 16,
        crop_size: Tuple[int, int] = (64, 64),
        x_range: Tuple[float, float] = None,
        y_range: Tuple[float, float] = None,
        extrapolation_value: float = 0.0,
        interpolation_method: str = "bilinear") -> tf.Tensor:
    """
    random crop from each image in the batch

    :param input_batch: 4D tensor
    :param no_crops_per_image: number of crops per image in batch
    :param crop_size: final crop size output
    :param x_range: manually set x_range
    :param y_range: manually set y_range
    :param extrapolation_value: value set to beyond the image crop
    :param interpolation_method: interpolation method
    :return: tensor with shape
        [input_batch[0] * no_crops_per_image,
         crop_size[0],
         crop_size[1],
         input_batch[3]]
    """
    shape = tf.shape(input_batch)
    original_dtype = input_batch.dtype
    batch_size = shape[0]

    if shape[1] <= 0 or shape[2] <= 0:
        return \
            tf.zeros(
                shape=(no_crops_per_image, crop_size[0], crop_size[1], shape[3]),
                dtype=original_dtype)

    # computer the total number of crops
    total_crops = no_crops_per_image * batch_size

    # fill y_range, x_range based on crop size and input batch size
    if y_range is None:
        y_range = (float(crop_size[0] / shape[1]),
                   float(crop_size[0] / shape[1]))

    if x_range is None:
        x_range = (float(crop_size[1] / shape[2]),
                   float(crop_size[1] / shape[2]))

    #
    y1 = tf.random.uniform(
        shape=(total_crops, 1), minval=0.0, maxval=1.0 - y_range[0])
    y2 = y1 + y_range[1]
    #
    x1 = tf.random.uniform(
        shape=(total_crops, 1), minval=0.0, maxval=1.0 - x_range[0])
    x2 = x1 + x_range[1]
    # limit the crops to the end of image
    y1 = tf.maximum(y1, 0.0)
    y2 = tf.minimum(y2, 1.0)
    x1 = tf.maximum(x1, 0.0)
    x2 = tf.minimum(x2, 1.0)
    # concat the dimensions to create [total_crops, 4] boxes
    boxes = tf.concat([y1, x1, y2, x2], axis=1)

    # --- randomly choose the image to crop inside the batch
    box_indices = \
        tf.random.uniform(
            shape=(total_crops,),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32)

    result = \
        tf.image.crop_and_resize(
            image=input_batch,
            boxes=boxes,
            box_indices=box_indices,
            crop_size=crop_size,
            method=interpolation_method,
            extrapolation_value=extrapolation_value)

    del boxes
    del box_indices
    del x1, y1, x2, y2
    del y_range, x_range

    # --- cast to original img dtype (no surprises principle)
    return tf.cast(result, dtype=original_dtype)

# ---------------------------------------------------------------------


def subsample(
        input_batch: tf.Tensor) -> tf.Tensor:
    return \
        tf.nn.max_pool2d(
            input=input_batch,
            ksize=(1, 1),
            strides=(2, 2),
            padding="SAME")

# ---------------------------------------------------------------------


def downsample(
        input_batch: tf.Tensor,
        kernel_size: Tuple[int, int] = (3, 3)) -> tf.Tensor:
    x = \
        tfa.image.gaussian_filter2d(
            sigma=1,
            image=input_batch,
            filter_shape=kernel_size)
    return subsample(input_batch=x)

# ---------------------------------------------------------------------

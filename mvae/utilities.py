import os
import PIL
import json
import copy
import pathlib
import itertools
import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Union, Dict, Iterable

# ==============================================================================
# local imports
# ==============================================================================


from .custom_logger import logger


# ==============================================================================

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


# ---------------------------------------------------------------------


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
        squeeze_and_excite_params: Dict = None,
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
    :param squeeze_and_excite_params: squeeze and excitation parameters, if None do disable
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
    use_squeeze_and_excite = squeeze_and_excite_params is not None
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

    if conv_activation.lower() in ["leaky_relu", "leakyrelu"]:
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

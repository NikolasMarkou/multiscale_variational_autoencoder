import copy
import tensorflow as tf
from collections import namedtuple
from typing import Dict, Tuple, List

# ---------------------------------------------------------------------

from .constants import *
from .utilities import conv2d_wrapper, ConvType

# ---------------------------------------------------------------------


def builder_scales_fusion(
        levels: int = 5,
        filters: int = 16,
        filters_level_multiplier: float = 2.0) -> tf.keras.Model:
    filters_per_level = [
        int(round(filters * max(1, filters_level_multiplier ** i)))
        for i in range(levels)
    ]

    input_layers = [
        tf.keras.Input(
            name=f"input_tensor_{i}",
            shape=(None, None, filters_per_level[i]))
        for i in range(levels)
    ]

    output_layer = None

    for i in reversed(range(levels)):
        input_layer = input_layers[i]
        if output_layer is None:
            output_layer = input_layer
        else:
            output_layer = (
                tf.keras.layers.UpSampling2D(
                    size=(2, 2), interpolation="nearest")(output_layer))
            output_layer = (
                tf.keras.layers.Concatenate(axis=-1)([input_layer, output_layer]))

    return (
        tf.keras.Model(
            name=f"scales_fusion",
            trainable=False,
            inputs=input_layers,
            outputs=[output_layer]))

# ---------------------------------------------------------------------


def builder_scales_splitter(
        levels: int = 5,
        filters: int = 16,
        filters_level_multiplier: float = 2.0) -> tf.keras.Model:
    filters_per_level = [
        int(round(filters * max(1, filters_level_multiplier ** i)))
        for i in range(levels)
    ]

    input_layer = tf.keras.Input(
            name=f"input_tensor",
            shape=(None, None, sum(filters_per_level)))
    output_layers = tf.split(input_layer, filters_per_level, axis=-1)

    for i in range(levels):
        s = int(i ** 2)
        if s == 1:
            continue
        output_layers[i] = (
            tf.keras.layers.MaxPooling2D(
                pool_size=(1, 1), strides=(s, s))(output_layers[i]))

    return (
        tf.keras.Model(
            name=f"scales_splitter",
            trainable=False,
            inputs=input_layer,
            outputs=output_layers))

# ---------------------------------------------------------------------

def builder_vae(
        levels: int = 5,
        filters: int = 16,
        filters_level_multiplier: float = 2.0,
        activation: str = "gelu",
        use_bn: bool = True,
        use_ln: bool = False,
        use_bias: bool = False,
        kernel_regularizer="l2",
        kernel_initializer="glorot_normal"):
    filters_per_level = [
        int(round(filters * max(1, filters_level_multiplier ** i)))
        for i in range(levels)
    ]
    bn_params = None
    if use_bn:
        bn_params = \
            dict(
                scale=True,
                center=use_bias,
                momentum=DEFAULT_BN_MOMENTUM,
                epsilon=DEFAULT_BN_EPSILON
            )

    ln_params = None
    if use_ln:
        ln_params = \
            dict(
                scale=True,
                center=use_bias,
                epsilon=DEFAULT_LN_EPSILON
            )
    base_conv_params = dict(
        kernel_size=(2, 2),
        filters=filters,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    # ----
    input_layer = tf.keras.Input(
            name=INPUT_TENSOR_STR,
            shape=(None, None, None))
    x = input_layer
    for i in range(levels):
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=bn_params,
                ln_post_params=ln_params,
                conv_params=base_conv_params)
    x = tf


# ---------------------------------------------------------------------

"""
vanilla unet backbone
"""

import copy
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import conv2d_wrapper, ConvType

# ---------------------------------------------------------------------


def builder(
        input_dims,
        levels: int = 5,
        kernel_size: int = 3,
        filters: int = 32,
        filters_level_multiplier: float = 2.0,
        activation: str = "relu",
        use_bn: bool = True,
        use_ln: bool = False,
        use_bias: bool = False,
        use_noise_regularization: bool = False,
        kernel_regularizer="l1",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        spatial_dropout_rate: float = -1,
        cheap_upsample: bool = False,
        multiple_scale_outputs: bool = False,
        output_layer_name: str = "intermediate_output",
        name="unet",
        **kwargs) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    builds a unet vanilla model as described in

    U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)

    :param input_dims: Models input dimensions
    :param levels: number of levels to go down
    :param kernel_size: kernel size of base convolutional layer
    :param filters_level_multiplier: every down level increase the number of filters by a factor of
    :param filters: filters of base convolutional layer
    :param activation: activation of the convolutional layers
    :param dropout_rate: probability of dropout, negative to turn off
    :param spatial_dropout_rate: probability of dropout, negative to turn off
    :param use_bn: use batch normalization
    :param use_ln: use layer normalization
    :param use_bias: use bias (bias free means this should be off)
    :param use_noise_regularization: if True add gaussian noise
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param output_layer_name: the output layer name
    :param cheap_upsample: if True upsample using nearest neighbors
    :param multiple_scale_outputs:
    :param name: name of the model

    :return: unet encoder/decoder model
    """
    # --- logging
    logger.info("building unet backbone")
    if len(kwargs) > 0:
        logger.info(f"parameters not used: {kwargs}")

    # --- setup parameters
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
                epsilon=DEFAULT_BN_EPSILON
            )

    pool_params = dict(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="SAME"
    )

    upsample_params = dict(
        size=(2, 2),
        interpolation="nearest"
    )

    base_conv_params = dict(
        kernel_size=kernel_size,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    conv_params = []
    up_level_conv_params = []
    for i in range(levels):
        params = copy.deepcopy(base_conv_params)
        params["filters"] = int(round(filters * (filters_level_multiplier ** i)))
        conv_params.append(params)

        params = copy.deepcopy(base_conv_params)
        params["filters"] = int(round(filters * (filters_level_multiplier ** i)))
        params["kernel_size"] = (2, 2)
        params["strides"] = (2, 2)
        up_level_conv_params.append(params)

    # --- build model
    # set input
    input_layer = \
        keras.Input(
            name="input_tensor",
            shape=input_dims)
    x = input_layer

    x_down = []
    x_level = x

    # --- down
    for i in range(levels):
        x_level = \
            conv2d_wrapper(
                input_layer=x_level,
                bn_post_params=None,
                ln_post_params=None,
                conv_params=conv_params[i])

        if use_noise_regularization:
            x_level = \
                tf.keras.layers.GaussianNoise(
                    stddev=0.01)(x_level)

        x_level = \
            conv2d_wrapper(
                input_layer=x_level,
                bn_post_params=bn_params,
                ln_post_params=ln_params,
                conv_params=conv_params[i])

        x_down.append(x_level)

        x_level = \
            keras.layers.MaxPooling2D(
                **pool_params)(x_level)

    # --- add optional dropout between encoders and decoders
    if dropout_rate > 0 or spatial_dropout_rate > 0:
        for i in range(len(x_down)):
            if dropout_rate > 0:
                x_down[i] = \
                    keras.layers.Dropout(
                        rate=dropout_rate)(x_down[i])

            if spatial_dropout_rate > 0:
                x_down[i] = \
                    keras.layers.SpatialDropout2D(
                        rate=spatial_dropout_rate)(x_down[i])

    # --- create encoder
    model_encoder = tf.keras.Model(
        name=f"{name}_encoder",
        trainable=True,
        inputs=input_layer,
        outputs=x_down)

    # --- up
    x_level = x_down[-1]
    x_levels = [x_level]
    for i in reversed(range(levels-1)):
        x_down_i = x_down[i]

        if cheap_upsample:
            x_level = \
                tf.keras.layers.UpSampling2D(
                    **upsample_params)(x_level)
        else:
            x_level = \
                conv2d_wrapper(
                    input_layer=x_level,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=up_level_conv_params[i],
                    conv_type=ConvType.CONV2D_TRANSPOSE)
        x_level = \
            tf.keras.layers.Concatenate()([x_level, x_down_i])

        x_level = conv2d_wrapper(
            input_layer=x_level,
            ln_post_params=None,
            bn_post_params=None,
            conv_params=conv_params[i])

        x_level = conv2d_wrapper(
            input_layer=x_level,
            ln_post_params=ln_params,
            bn_post_params=bn_params,
            conv_params=conv_params[i])

        x_levels.append(x_level)

    # --- output layer here
    output_layers = []
    if multiple_scale_outputs:
        for i in range(0, levels-1):
            # i = 0, 1, 2, 3
            depth = levels - i - 1
            x = x_levels[i]
            x = tf.keras.layers.Layer(name=f"{output_layer_name}_{depth}")(x)
            output_layers.append(x)

    output_layers += [
        tf.keras.layers.Layer(name=f"{output_layer_name}_{0}")(x_level)
    ]

    # IMPORTANT
    # reverse it so the deepest output is first
    # otherwise we will get the most shallow output
    output_layers = output_layers[::-1]

    # --- create decoder
    model_decoder = tf.keras.Model(
        name=f"{name}_decoder",
        trainable=True,
        inputs=[
            keras.Input(
                name=f"input_tensor_{i}",
                shape=(None, None, conv_params[i]["filters"]))
            for i in range(levels)
        ],
        outputs=output_layers)

    return model_encoder, model_decoder

# ---------------------------------------------------------------------

"""
unet+ backbone
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
from .utilities import conv2d_wrapper

# ---------------------------------------------------------------------


def builder(
        input_dims,
        width: int = 1,
        levels: int = 5,
        backbone_kernel_size: int = 7,
        kernel_size: int = 3,
        filters: int = 32,
        filters_level_multiplier: float = 2.0,
        activation: str = "gelu",
        use_bn: bool = True,
        use_ln: bool = False,
        use_bias: bool = False,
        use_scale_diffs: bool = False,
        kernel_regularizer="l2",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        spatial_dropout_rate: float = -1,
        multiple_scale_outputs: bool = False,
        output_layer_name: str = "intermediate_output",
        name="unet_p",
        **kwargs) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    builds a unet model that uses convnext blocks

    :param input_dims: Models input dimensions
    :param width:
    :param levels: number of levels to go down
    :param backbone_kernel_size: kernel size of backbone encoder convolutional layer
    :param kernel_size: kernel size of decoder convolutional layer
    :param filters_level_multiplier: every down level increase the number of filters by a factor of
    :param filters: filters of base convolutional layer
    :param activation: activation of the convolutional layers
    :param dropout_rate: probability of dropout, negative to turn off
    :param spatial_dropout_rate: probability of dropout, negative to turn off
    :param use_bn: use batch normalization
    :param use_ln: use layer normalization
    :param use_bias: use bias (bias free means this should be off)
    :param use_scale_diffs: remove per scale diffs
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param multiple_scale_outputs:
    :param output_layer_name: the output layer name
    :param name: name of the model

    :return: unet+ encoder/decoder model
    """
    # --- argument checking
    logger.info("building unet_pp backbone")
    if len(kwargs) > 0:
        logger.info(f"parameters not used: {kwargs}")

    if backbone_kernel_size <= 0:
        raise ValueError("backbone_kernel_size must be > 0")

    if kernel_size <= 0:
        kernel_size = backbone_kernel_size

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
                epsilon=DEFAULT_LN_EPSILON
            )

    dropout_params = None
    if dropout_rate > 0.0:
        dropout_params = {"rate": dropout_rate}

    dropout_2d_params = None
    if spatial_dropout_rate > 0.0:
        dropout_2d_params = {"rate": spatial_dropout_rate}

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
    conv_params_up = []
    conv_params_res_1 = []
    conv_params_res_2 = []
    conv_params_res_3 = []

    for i in range(levels):
        filters_level = \
            int(round(filters * max(1, filters_level_multiplier ** i)))

        # conv2d params when moving horizontally the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        conv_params.append(params)

        # 1st residual conv
        params = \
            dict(
                kernel_size=kernel_size,
                depth_multiplier=1,
                strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                activation="linear",
                depthwise_regularizer=kernel_regularizer,
                depthwise_initializer=kernel_initializer
            )
        conv_params_res_1.append(params)

        # 2nd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = activation
        params["filters"] = filters_level * 4
        conv_params_res_2.append(params)

        # 3rd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = "linear"
        params["filters"] = filters_level
        conv_params_res_3.append(params)

        # conv2d params when moving up the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = (3, 3)
        params["strides"] = (1, 1)
        params["activation"] = conv_params_res_3[-1]["activation"]
        conv_params_up.append(params)

    # --- book keeping
    nodes_dependencies = {}

    # default unet
    for j in range(0, levels-1, 1):
        # bottom and left dependencies
        nodes_dependencies[(j, 1)] = [
            (j, 0),
            (j + 1, 1)
        ]

    # add the deepest level
    nodes_dependencies[(levels-1, 1)] = [
        (levels-1, 0)
    ]

    nodes_output = {}
    nodes_to_visit = list(nodes_dependencies.keys())
    nodes_visited = set()

    # --- build model
    # set input
    encoder_input_layer = \
        keras.Input(
            name=INPUT_TENSOR_STR,
            shape=input_dims)
    x = encoder_input_layer

    # all the down sampling, backbone
    for i in range(levels):
        x_skip = None
        for j in range(width):
            if i == 0 and j == 0:
                # first ever
                params = copy.deepcopy(base_conv_params)
                params["filters"] = max(32, filters)
                params["kernel_size"] = (backbone_kernel_size, backbone_kernel_size)
                x = \
                    conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=bn_params,
                        ln_post_params=ln_params,
                        conv_params=params)
            elif j == 0:
                # new level
                if use_scale_diffs:
                    node_level = (i-1, 0)
                    x_down_up = \
                        tf.keras.layers.AveragePooling2D(
                            pool_size=(3, 3), padding="same", strides=(2, 2))(x)
                    x_down_up = tf.keras.layers.UpSampling2D(size=(2, 2))(x_down_up)
                    nodes_output[node_level] = \
                        nodes_output[node_level] - tf.no_gradient(x_down_up)
                x = \
                    tf.keras.layers.MaxPooling2D(
                        pool_size=(2, 2), padding="same", strides=(2, 2))(x)
                params = copy.deepcopy(conv_params_res_1[i])
                params["kernel_size"] = (backbone_kernel_size, backbone_kernel_size)
                x = \
                    conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=bn_params,
                        ln_post_params=ln_params,
                        conv_params=params)
            else:
                params = copy.deepcopy(conv_params_res_1[i])
                params["kernel_size"] = (kernel_size, kernel_size)
                x = \
                    conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=bn_params,
                        ln_post_params=ln_params,
                        conv_params=params)
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=conv_params_res_2[i])
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    conv_params=conv_params_res_3[i])
            if j > 0 and x_skip is not None:
                x = x + x_skip
            x_skip = x

        node_level = (i, 0)
        nodes_visited.add(node_level)
        nodes_output[node_level] = x

    i = None
    x = None
    nodes_visited.add((levels - 1, 1))
    nodes_output[(levels - 1, 1)] = nodes_output[(levels - 1, 0)]

    for k in nodes_output.keys():
        if dropout_params is not None:
            nodes_output[k] = (
                tf.keras.layers.Dropout(rate=dropout_params["rate"])(nodes_output[k]))
        if dropout_2d_params is not None:
            nodes_output[k] = (
                tf.keras.layers.SpatialDropout2D(rate=dropout_2d_params["rate"])(nodes_output[k]))

    # --- create encoder
    model_encoder = tf.keras.Model(
        name=f"{name}_encoder",
        trainable=True,
        inputs=encoder_input_layer,
        outputs=[
            nodes_output[(i, 0)]
            for i in range(levels)
        ])

    decoder_inputs = [
        tf.keras.Input(
            name=f"input_tensor_{i}",
            shape=(None, None, conv_params_res_3[i]["filters"]))
        for i in range(levels)
    ]

    for i in range(levels):
        nodes_output[(i, 0)] = decoder_inputs[i]
    nodes_output[(levels - 1, 1)] = nodes_output[(levels - 1, 0)]

    # --- move up
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"node: [{node}, nodes_visited: {nodes_visited}, nodes_to_visit: {nodes_to_visit}")
        logger.info(f"dependencies: {nodes_dependencies[node]}")
        # make sure a node is not visited twice
        if node in nodes_visited:
            continue
        # make sure that all the dependencies for a node are matched
        dependencies = nodes_dependencies[node]
        dependencies_matched = \
            all([
                (d in nodes_output) and (d in nodes_visited or d == node)
                for d in dependencies
            ])
        if not dependencies_matched:
            nodes_to_visit.append(node)
            continue
        # sort it so all same level dependencies are first and added
        # as residual before finally concatenating the previous scale
        dependencies = \
            sorted(list(dependencies),
                   key=lambda d: d[0],
                   reverse=False)
        logger.debug(f"processing node: {node}, "
                     f"dependencies: {dependencies}, "
                     f"nodes_output: {list(nodes_output.keys())}")

        x_input = []

        logger.debug(f"node: [{node}], dependencies: {dependencies}")
        for d in dependencies:
            logger.debug(f"processing dependency: {d}")
            x = nodes_output[d]
            if d[0] == node[0]:
                # same level
                pass
            elif d[0] > node[0]:
                # lower level, upscale
                x = tf.keras.layers.UpSampling2D(
                    size=(2, 2), interpolation="nearest")(x)
                x = conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    conv_params=conv_params_up[node[0]])
            else:
                raise ValueError(f"node: {node}, dependencies: {dependencies}, "
                                 f"should not supposed to be here")
            x_input.append(x)

        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 0:
            x = tf.keras.layers.Concatenate()(x_input)
        else:
            raise ValueError("this must never happen")

        # --- convnext block
        x_skip = None
        for j in range(width):
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=conv_params_res_1[node[0]])
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=conv_params_res_2[node[0]])
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    conv_params=conv_params_res_3[node[0]])
            if j > 0 and x_skip is not None:
                x = x + x_skip
            x_skip = x

        nodes_output[node] = x
        nodes_visited.add(node)

    # --- output layer here
    output_layers = []

    # depth outputs
    if multiple_scale_outputs:
        tmp_output_layers = []
        for i in range(1, levels, 1):
            depth = i
            width = 1
            x = nodes_output[(depth, width)]
            x = tf.keras.layers.Layer(
                name=f"{output_layer_name}_{depth}_{width}")(x)
            tmp_output_layers.append(x)
        # reverse here so deeper levels come on top
        output_layers += tmp_output_layers[::-1]

    # add as last the best output
    depth = 0
    width = 1
    output_layers += [
        tf.keras.layers.Layer(name=f"{output_layer_name}_{depth}_{width}")(
            nodes_output[(depth, width)])
    ]

    # IMPORTANT
    # reverse it so the deepest output is first
    # otherwise we will get the most shallow output
    output_layers = output_layers[::-1]

    # --- create decoder
    model_decoder = tf.keras.Model(
        name=f"{name}_decoder",
        trainable=True,
        inputs=decoder_inputs,
        outputs=output_layers)

    return model_encoder, model_decoder

# ---------------------------------------------------------------------

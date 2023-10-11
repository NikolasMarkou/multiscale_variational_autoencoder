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
from .custom_layers import GaussianFilter
from .utilities import conv2d_wrapper, ConvType
from .layer_blocks import (
    RandomOnOff,
    skip_squeeze_and_excite_block,
    self_attention_block)
from .regularizers import (
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer)

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
        use_laplacian: bool = False,
        use_self_attention: bool = False,
        use_squeeze_excite: bool = False,
        use_random_on_off: bool = True,
        use_orthonormal_projections: bool = False,
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
    :param use_laplacian: remove per scale diffs
    :param use_self_attention:
    :param use_squeeze_excite:
    :param use_orthonormal_projections: if True use orthonormal projections on the 1x1 kernels
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
        # # 96 is the max for convnext
        # filters_level = min(96, filters_level)
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
        params["kernel_size"] = (1, 1)
        params["activation"] = activation
        params["filters"] = filters_level * 4
        if use_orthonormal_projections:
            logger.info("added SoftOrthonormalConstraintRegularizer")
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01,
                    l1_coefficient=0.0,
                    l2_coefficient=0.00001)
        conv_params_res_2.append(params)

        # 3rd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = (1, 1)
        params["activation"] = "linear"
        params["filters"] = filters_level
        if use_orthonormal_projections:
            logger.info("added SoftOrthonormalConstraintRegularizer")
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.01,
                    l1_coefficient=0.0,
                    l2_coefficient=0.00001)
        conv_params_res_3.append(params)

        # conv2d params when moving up the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = (2, 2)
        params["strides"] = (2, 2)
        params["activation"] = conv_params_res_3[-1]["activation"]
        params["kernel_regularizer"] = tf.keras.regularizers.L2(l2=0.00001)
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

    # first ever
    params = copy.deepcopy(base_conv_params)
    params["filters"] = max(32, filters)
    params["activation"] = activation
    params["kernel_size"] = (backbone_kernel_size, backbone_kernel_size)
    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_post_params=None,
            ln_post_params=None,
            conv_params=params)

    # all the down sampling, backbone
    for i in range(levels):
        x_skip = None
        for j in range(width):
            if i == 0:
                params = copy.deepcopy(conv_params_res_1[i])
                params["kernel_size"] = (backbone_kernel_size, backbone_kernel_size)
                x = \
                    conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=bn_params,
                        ln_post_params=ln_params,
                        conv_params=params)
            elif i > 0 and j == 0:
                # new level
                if use_laplacian:
                    node_level = (i-1, 0)
                    # gaussian blur
                    x_blurred = \
                        GaussianFilter(
                            kernel_size=(5, 5),
                            strides=(1, 1))(x)
                    # laplacian
                    nodes_output[node_level] = \
                        nodes_output[node_level] - tf.stop_gradient(x_blurred)
                    # half resolution
                    x = x_blurred[:, ::2, ::2, :]
                else:
                    # max pooling, half resolution
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

    for k in nodes_output.keys():
        if dropout_params is not None:
            nodes_output[k] = (
                tf.keras.layers.Dropout(rate=dropout_params["rate"])(nodes_output[k]))
        if dropout_2d_params is not None:
            nodes_output[k] = (
                tf.keras.layers.SpatialDropout2D(rate=dropout_2d_params["rate"])(nodes_output[k]))
        if use_random_on_off:
            # last level does not get an on off
            if k != (levels-1, 0):
                nodes_output[k] = RandomOnOff(rate=0.5)(nodes_output[k])

    nodes_visited.add((levels - 1, 1))
    nodes_output[(levels - 1, 1)] = nodes_output[(levels - 1, 0)]

    for k in nodes_output.keys():
        depth = k[0]
        nodes_output[k] = \
            tf.keras.layers.Layer(name=f"backbone_level_{depth}")(nodes_output[k])

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

    # ---  self attention block
    if use_self_attention:
        nodes_output[(levels - 1, 1)] = \
            self_attention_block(
                input_layer=nodes_output[(levels - 1, 1)],
                conv_params=conv_params_res_3[-1])

    # --- squeeze and excite preparation
    control_layer = None

    # --- move up
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"node: [{node}, "
                    f"nodes_visited: {nodes_visited}, "
                    f"nodes_to_visit: {nodes_to_visit}")
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

        logger.debug(f"node: [{node}], "
                     f"dependencies: {dependencies}")

        for d in dependencies:
            logger.debug(f"processing dependency: {d}")
            x = nodes_output[d]
            if d[0] == node[0]:
                # same level
                pass
            elif d[0] > node[0]:
                # lower level, upscale
                if dropout_params is not None:
                    x = tf.keras.layers.Dropout(rate=dropout_params["rate"])(x)
                if dropout_2d_params is not None:
                    x = tf.keras.layers.SpatialDropout2D(rate=dropout_2d_params["rate"])(x)
                x = conv2d_wrapper(
                    input_layer=x,
                    bn_params=None,
                    ln_params=None,
                    ln_post_params=None,
                    bn_post_params=None,
                    conv_params=conv_params_up[node[0]],
                    conv_type=ConvType.CONV2D_TRANSPOSE)
            else:
                raise ValueError(f"node: {node}, "
                                 f"dependencies: {dependencies}, "
                                 f"should not supposed to be here")
            x_input.append(x)

        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 0:
            if use_laplacian:
                x = tf.keras.layers.Add()(x_input)
            else:
                x = tf.keras.layers.Concatenate()(x_input)
        else:
            raise ValueError("this must never happen")

        # --- convnext block
        x_skip = None

        if use_squeeze_excite:
            control_layer_tmp = (
                nodes_output)[(node[0]+1, 1)]
            control_layer_tmp = (
                conv2d_wrapper(
                    input_layer=control_layer_tmp,
                    bn_params=bn_params,
                    ln_params=ln_params,
                    conv_params=conv_params_res_3[0]))
            control_layer_tmp = \
                tf.keras.layers.GlobalAvgPool2D(keepdims=True)(
                    control_layer_tmp)

            if control_layer is None:
                control_layer = control_layer_tmp
            else:
                control_layer = \
                    tf.keras.layers.Concatenate(axis=-1)([
                        control_layer,
                        control_layer_tmp
                    ])

        for j in range(width):
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=conv_params_res_1[node[0]])
            # pass global information here
            if j == 0 and use_squeeze_excite:
                x = \
                    skip_squeeze_and_excite_block(
                        control_layer=control_layer,
                        signal_layer=x,
                        flatten=False,
                        hard_sigmoid_version=True,
                        learn_to_turn_off=False,
                        bn_params=None,
                        ln_params=None)
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

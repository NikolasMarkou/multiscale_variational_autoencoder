"""
modified unet backbone
"""

import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import \
    ConvType, \
    conv2d_wrapper
from .upsampling import upsample
from .downsampling import downsample
from .regularizers import (
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer)
from .custom_layers import (
    AttentionGate,
    StochasticDepth,
    SqueezeExcitation,
    ConvolutionalSelfAttention)


# ---------------------------------------------------------------------


def builder(
        input_dims,
        depth: int = 5,
        width: int = 1,
        backbone_kernel_size: int = 7,
        kernel_size: int = -1,
        filters: int = 32,
        max_filters: int = -1,
        filters_level_multiplier: float = 2.0,
        activation: str = "gelu",
        second_activation: str = "linear",
        upsample_type: str = "conv2d_transpose",
        downsample_type: str = "maxpool",
        use_bn: bool = True,
        use_ln: bool = False,
        use_bias: bool = False,
        use_concat: bool = True,
        use_attention_gates: bool = False,
        use_squeeze_excitation: bool = False,
        use_noise_regularization: bool = False,
        use_soft_orthogonal_regularization: bool = False,
        use_soft_orthonormal_regularization: bool = False,
        kernel_regularizer="l2",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        depth_drop_rate: float = 0.5,
        spatial_dropout_rate: float = -1,
        dropout_skip_connection_rate: float = -1,
        align_scale_outputs: bool = False,
        multiple_scale_outputs: bool = False,
        attention_block_passes: List[Tuple[int, int]] = [],
        r_ratio: float = 0.25,
        conv_constant: float = 0.0,
        output_layer_name: str = "intermediate_output",
        name="unet_p",
        **kwargs) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    builds a modified unet model that uses convnext blocks

    :param input_dims: Models input dimensions
    :param depth: number of levels to go down
    :param width: number of horizontals nodes, if -1 it gets set to depth
    :param kernel_size: kernel size of the rest of convolutional layers
    :param backbone_kernel_size: kernel size of backbone convolutional layer
    :param filters_level_multiplier: every down level increase the number of filters by a factor of
    :param filters: filters of base convolutional layer
    :param max_filters: max number of filters
    :param activation: activation of the first 1x1 kernel
    :param second_activation: activation of the second 1x1 kernel
    :param upsample_type:
    :param downsample_type:
    :param use_bn: use batch normalization
    :param use_ln: use layer normalization
    :param use_bias: use bias (bias free means this should be off)
    :param use_noise_regularization: if true add a gaussian noise layer on each scale
    :param use_concat: if true concatenate otherwise add skip layers
    :param use_attention_gates: if true use attention gates between scales
    :param use_squeeze_excitation: if true add squeeze and excitation layers
    :param use_soft_orthogonal_regularization: if True use soft orthogonal regularization on the 1x1 kernels
    :param use_soft_orthonormal_regularization: if true use soft orthonormal regularization on the 1x1 middle kernels
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param attention_block_passes: List of tuples (depth, passes),
        which depth of the encoder to add the attention block and how many passes, example [(3,1),(4,2)]
    :param dropout_rate: probability of dropout, negative to turn off
    :param spatial_dropout_rate: probability of spatial dropout, negative to turn off
    :param depth_drop_rate: probability of residual block dropout, negative to turn off
    :param dropout_skip_connection_rate: probability of dropping a skip connection between encoder decoder
    :param align_scale_outputs: if True align multiple scales to the same output units
    :param multiple_scale_outputs: if True for each scale give an output
    :param r_ratio: ratio of squeeze and excitation
    :param conv_constant: if different than zero add to each convolution (makes training more stable)
    :param output_layer_name: the output layer's name
    :param name: model's name
    :return: unet p model
    """
    # --- argument checking
    logger.info("building unet_p backbone")
    if len(kwargs) > 0:
        logger.info(f"parameters not used: {kwargs}")

    if width is None or width <= 0:
        width = 1

    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be > 0")

    if kernel_size is None or kernel_size <= 0:
        kernel_size = backbone_kernel_size

    if kernel_size <= 0 or backbone_kernel_size <= 0:
        raise ValueError(
            f"kernel_size: [{kernel_size}] and "
            f"backbone_kernel_size: [{backbone_kernel_size}] must be > 0")
    if second_activation is None:
        second_activation = ""
    second_activation = second_activation.strip().lower()
    if len(second_activation) <= 0:
        second_activation = activation
    if use_soft_orthonormal_regularization and use_soft_orthogonal_regularization:
        raise ValueError(
            "only one use_soft_orthonormal_regularization or "
            "use_soft_orthogonal_regularization must be turned on")
    if attention_block_passes is None or \
            not isinstance(attention_block_passes, List):
        attention_block_passes = []

    upsample_type = upsample_type.strip().lower()
    downsample_type = downsample_type.strip().lower()

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

    depth_drop_rates = None
    use_depth_drop = False
    if depth_drop_rate > 0.0:
        depth_drop_rates = [
            float(x) for x in np.linspace(0.0, depth_drop_rate, width)
        ]
        use_depth_drop = True

    base_conv_params = dict(
        kernel_size=backbone_kernel_size,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    conv_params_up = []
    conv_params_down = []
    conv_params_res_1 = []
    conv_params_res_2 = []
    conv_params_res_3 = []

    for d in range(depth):
        filters_level = \
            int(round(filters * max(1, filters_level_multiplier ** d)))
        if max_filters > 0:
            filters_level = min(max_filters, filters_level)
        filters_level_next = \
            int(round(filters * max(1, filters_level_multiplier ** (d+1))))
        if max_filters > 0:
            filters_level_next = min(max_filters, filters_level_next)

        # 1st residual conv
        params = \
            dict(
                kernel_size=backbone_kernel_size,
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
        if use_soft_orthogonal_regularization:
            logger.info("added SoftOrthogonalConstraintRegularizer")
            params["kernel_regularizer"] = \
                SoftOrthogonalConstraintRegularizer(
                    lambda_coefficient=0.1, l1_coefficient=0.0, l2_coefficient=1e-3)
        if use_soft_orthonormal_regularization:
            logger.info("added SoftOrthonormalConstraintRegularizer")
            params["kernel_regularizer"] = \
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=0.1, l1_coefficient=0.0, l2_coefficient=1e-3)
        conv_params_res_2.append(params)

        # 3rd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["activation"] = second_activation
        params["filters"] = filters_level
        conv_params_res_3.append(params)

        # conv2d params when moving up the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = (2, 2)
        params["strides"] = (2, 2)
        # this needs to be the activation of the 3rd convnext so the outputs are aligned
        params["activation"] = conv_params_res_3[-1]["activation"]
        conv_params_up.append(params)

        # conv2d params when moving down the scale
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level_next
        params["kernel_size"] = (2, 2)
        params["strides"] = (2, 2)
        # this needs to be the activation of the 3rd convnext so the outputs are aligned
        params["activation"] = conv_params_res_3[-1]["activation"]
        conv_params_down.append(params)

    # --- book keeping
    nodes_dependencies = {}
    for d in range(0, depth, 1):
        if d == (depth - 1):
            nodes_dependencies[(d, 1)] = [(d, 0)]
        else:
            # add only left and bottom dependency
            nodes_dependencies[(d, 1)] = [(d, 0), (d + 1, 1)]

    nodes_output = {}
    nodes_to_visit = list(nodes_dependencies.keys())
    nodes_visited = set([(depth - 1, 0), (depth - 1, 1)])

    # --- build model
    # set input layer
    encoder_input_layer = \
        keras.Input(
            name=INPUT_TENSOR_STR,
            shape=input_dims)
    x = encoder_input_layer

    # build backbone
    for d in range(depth):
        for w in range(width):
            # get skip for residual
            x_skip = x

            if w == 0 and d == 0:
                # stem similar to ConvNext
                params = copy.deepcopy(base_conv_params)
                params["filters"] = max(32, filters)
                params["kernel_size"] = (5, 5)
                x = \
                    conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=bn_params,
                        ln_post_params=ln_params,
                        conv_params=params)
            else:
                x = \
                    conv2d_wrapper(
                        input_layer=x,
                        bn_post_params=bn_params,
                        ln_post_params=ln_params,
                        conv_params=copy.deepcopy(conv_params_res_1[d]),
                        conv_constant=conv_constant)

            if use_noise_regularization:
                x = tf.keras.layers.GaussianNoise(stddev=0.05, seed=0)(x)

            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    dropout_params=dropout_params,
                    dropout_2d_params=dropout_2d_params,
                    conv_constant=conv_constant,
                    conv_params=conv_params_res_2[d])

            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    conv_constant=conv_constant,
                    conv_params=conv_params_res_3[d])

            if w > 0:
                if use_depth_drop:
                    x = StochasticDepth(drop_path_rate=depth_drop_rates[w])(x)
                x = x_skip + x

        node_level = (d, 0)
        nodes_visited.add(node_level)
        nodes_output[node_level] = x

        x = (
            downsample(x,
                       downsample_type=downsample_type,
                       ln_params=ln_params,
                       bn_params=bn_params,
                       conv_params=conv_params_down[d]))

    # --- add attention passes
    for abp in attention_block_passes:
        # depth to apply attention
        d = abp[0]
        # passes to apply attention
        p = abp[1]
        attention_channels = conv_params_res_3[d]["filters"] // 4
        x = nodes_output[(d, 0)]
        for _ in range(p):
            x = \
                ConvolutionalSelfAttention(
                    attention_channels=attention_channels,
                    use_scale=True,
                    use_residual=True,
                    ln_params=ln_params,
                    bn_params=bn_params)(x)
        nodes_output[(d, 0)] = x

    # --- add dropout of all the skip connection
    # to encourage drawing from the previous depth
    if dropout_skip_connection_rate > 0:
        for d in range(depth-1):
            nodes_output[(d, 0)] = (
                tf.keras.layers.Dropout(
                    rate=dropout_skip_connection_rate/(d+1),
                    noise_shape=(1,))(nodes_output[(d, 0)]))

    # --- VERY IMPORTANT
    # add this, so it works correctly
    nodes_output[(depth-1, 1)] = nodes_output[(depth-1, 0)]

    # --- create encoder
    model_encoder = tf.keras.Model(
        name=f"{name}_encoder",
        trainable=True,
        inputs=encoder_input_layer,
        outputs=[
            nodes_output[(d, 0)]
            for d in range(depth)
        ])

    decoder_inputs = [
        tf.keras.Input(
            name=f"input_tensor_{d}",
            shape=(None, None, conv_params_res_3[d]["filters"]))
        for d in range(depth)
    ]

    # --- build the encoder side based on dependencies
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"node: [{node}, "
                    f"nodes_visited: {nodes_visited}, "
                    f"nodes_to_visit: {nodes_to_visit}")
        logger.info(f"dependencies: {nodes_dependencies[node]}")
        # make sure a node is not visited twice
        if node in nodes_visited:
            logger.info(f"node: [{node}] already processed")
            continue
        # make sure that all the dependencies for a node are matched
        dependencies = nodes_dependencies[node]
        dependencies_matched = \
            all([
                (d in nodes_output) and (d in nodes_visited or d == node)
                for d in dependencies
            ])
        if not dependencies_matched:
            logger.info(f"node: [{node}] dependencies not matches, continuing")
            nodes_to_visit.append(node)
            continue
        # sort it so all same level dependencies are first and added
        # as residual before finally concatenating the previous scale
        dependencies = \
            sorted(list(dependencies),
                   key=lambda d: d[0],
                   reverse=False)
        logger.info(f"processing node: {node}, "
                     f"dependencies: {dependencies}, "
                     f"nodes_output: {list(nodes_output.keys())}")

        x_input = []

        logger.debug(f"node: [{node}], dependencies: {dependencies}")
        for dependency in dependencies:
            logger.debug(f"processing dependency: {dependency}")
            x = nodes_output[dependency]

            # --- add squeeze and excite
            if use_squeeze_excitation:
                x = SqueezeExcitation(r_ratio=r_ratio)(x)

            if dependency[0] == node[0]:
                pass
            elif dependency[0] > node[0]:
                x = \
                    upsample(
                        input_layer=x,
                        upsample_type=upsample_type,
                        conv_params=conv_params_up[node[0]])
            else:
                raise ValueError(f"node: {node}, dependencies: {dependencies}, "
                                 f"should not supposed to be here")

            x_input.append(x)

        if use_attention_gates and len(x_input) == 2:
            logger.debug(f"adding AttentionGate at depth: [{node[0]}]")

            x_input[0] = (
                AttentionGate(
                    attention_channels=conv_params_res_3[node[0]]["filters"],
                    use_bias=use_bias,
                    use_ln=use_ln,
                    use_bn=use_bn)(x_input))

        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 0:
            if use_concat:
                x = tf.keras.layers.Concatenate()(x_input)
            else:
                x = tf.keras.layers.Add()(x_input)
        else:
            raise ValueError("this must never happen")

        # --- convnext block
        for w in range(width):
            x_skip = x

            params = copy.deepcopy(conv_params_res_1[node[0]])
            params["kernel_size"] = (kernel_size, kernel_size)
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=bn_params,
                    ln_post_params=ln_params,
                    conv_params=params,
                    conv_constant=conv_constant)
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    dropout_params=dropout_params,
                    dropout_2d_params=dropout_2d_params,
                    conv_params=conv_params_res_2[node[0]],
                    conv_constant=conv_constant)
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    bn_post_params=None,
                    ln_post_params=None,
                    conv_params=conv_params_res_3[node[0]],
                    conv_constant=conv_constant)

            if w > 0:
                if use_depth_drop:
                    x = StochasticDepth(drop_path_rate=depth_drop_rates[w])(x)
                x = x_skip + x

        nodes_output[node] = x
        nodes_visited.add(node)

    # --- output layer here
    output_layers = []

    # depth outputs
    if multiple_scale_outputs:
        tmp_output_layers = []
        for d in range(1, depth, 1):
            d = d
            w = 1

            if d < 0 or w < 0:
                logger.error(f"there is no node[{d},{w}] "
                             f"please check your assumptions")
                continue
            x = nodes_output[(d, w)]
            tmp_output_layers.append(x)
        # reverse here so deeper levels come on top
        output_layers += tmp_output_layers[::-1]

        if align_scale_outputs:
            # align all scales below the top one
            for d in range(len(output_layers)):
                output_layers[d] = \
                    conv2d_wrapper(
                        input_layer=output_layers[d],
                        bn_params=bn_params,
                        ln_params=ln_params,
                        conv_params=conv_params_res_3[0],
                        conv_constant=conv_constant)

    # add as last the best output
    output_layers += [
        nodes_output[(0, 1)]
    ]

    # IMPORTANT
    # reverse it so the deepest output is first
    # otherwise we will get the most shallow output
    output_layers = output_layers[::-1]

    # add names to the final layers
    for d in range(len(output_layers)):
        output_layers[d] = (
            tf.keras.layers.Layer(
                name=f"{output_layer_name}_{d}")(output_layers[d]))

    # --- create decoder
    model_decoder = tf.keras.Model(
        name=f"{name}_decoder",
        trainable=True,
        inputs=decoder_inputs,
        outputs=output_layers)

    return model_encoder, model_decoder

# ---------------------------------------------------------------------

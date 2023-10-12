import copy
import tensorflow as tf
from collections import namedtuple
from typing import Dict, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import \
    conv2d_wrapper, \
    input_shape_fixer, \
    build_normalize_model, \
    build_denormalize_model
from .backbone_unet import builder as builder_unet
from .backbone_unet_p import builder as builder_unet_p

# ---------------------------------------------------------------------


BuilderResults = namedtuple(
    "BuilderResults",
    {
        # this is the end to end model with multiple heads
        # (backbone, ssl and segmentation)
        HYDRA_STR,
        # converts input values to correct range
        NORMALIZER_STR,
        # converts output values to correct range
        DENORMALIZER_STR,
        # encoder part of the backbone
        ENCODER_STR,
        # decoder part of the backbone
        DECODER_STR,
        # this is the self-supervised denoiser task heads
        # (clips on the output of decoder)
        DENOISER_STR,
        # # this is the variational autoencoder task
        # # (clips between encoder and decoder)
        # VARIATIONAL_AUTOENCODER_STR,
    })

BackboneBuilderResults = namedtuple(
    "BackboneBuilderResults",
    {
        # converts input values to correct range
        NORMALIZER_STR,
        # converts output values to correct range
        DENORMALIZER_STR,
        # encoder part of the backbone
        ENCODER_STR,
        # decoder part of the backbone
        DECODER_STR
    }
)

# ---------------------------------------------------------------------


def model_builder(
        config: Dict) -> BuilderResults:
    # --- get configs
    batch_size = config.get(BATCH_SIZE_STR, None)
    config_denoiser = config[DENOISER_STR]
    config_backbone = config[BACKBONE_STR]

    # --- build backbone
    backbone_results = \
        model_backbone_builder(config=config_backbone)

    decoder_no_outputs = len(backbone_results.decoder.outputs)
    logger.warning(
        f"Decoder model has [{decoder_no_outputs}] outputs, "
        f"probably of different scale or depth")

    denoiser_input_channels = \
        tf.keras.backend.int_shape(
            backbone_results.decoder.outputs[0])[-1]
    denoiser_shape = copy.deepcopy(config_backbone[INPUT_SHAPE_STR])
    denoiser_shape[-1] = denoiser_input_channels
    config_denoiser[INPUT_SHAPE_STR] = denoiser_shape

    # --- build denoiser and other networks
    model_denoiser = model_denoiser_builder(config=config_denoiser)

    input_shape = tf.keras.backend.int_shape(backbone_results.encoder.inputs[0])[1:]
    logger.info("input_shape: [{0}]".format(input_shape))

    # --- build hydra combined model
    input_layer = \
        tf.keras.Input(
            shape=input_shape,
            dtype="float32",
            sparse=False,
            ragged=False,
            batch_size=batch_size,
            name=INPUT_TENSOR_STR)

    logger.info(f"input_layer: {input_layer}")
    input_normalized_layer = \
        backbone_results.normalizer(
            input_layer, training=False)

    logger.info(f"input_normalized_layer: {input_normalized_layer}")

    # common backbone low level
    encoding_results = \
        backbone_results.encoder(input_normalized_layer)
    decoding_results = \
        backbone_results.decoder(encoding_results)

    config_denoisers = []

    for i in range(decoder_no_outputs):
        input_channels = \
            tf.keras.backend.int_shape(decoding_results[i])[-1]
        shape = copy.deepcopy(config_backbone[INPUT_SHAPE_STR])
        shape[-1] = input_channels

        tmp_config_denoiser = copy.deepcopy(config_denoiser)
        tmp_config_denoiser[INPUT_SHAPE_STR] = copy.deepcopy(shape)
        tmp_config_denoiser["output_channels"] = input_shape[-1]
        config_denoisers.append(tmp_config_denoiser)

    # --- denoiser heads
    # check whether all the denoisers have the same input channels
    def is_true_fn(c) -> bool:
        return c[INPUT_SHAPE_STR][-1] == config_denoisers[0][INPUT_SHAPE_STR][-1]

    common_model = all([is_true_fn(c) for c in config_denoisers])

    if common_model:
        logger.info("creating a common denoiser for each output")
        denoiser = (
            model_denoiser_builder(
                config=config_denoisers[0],
                name=f"denoiser_head_{0}"))
        model_denoisers = [
            denoiser
            for _ in range(decoder_no_outputs)
        ]
    else:
        logger.info("creating a different denoiser for each output")
        model_denoisers = [
            model_denoiser_builder(
                config=config_denoisers[i],
                name=f"denoiser_head_{i}")
            for i in range(decoder_no_outputs)
        ]

    denoisers_mid = [
        backbone_results.denormalizer(
            model_denoisers[i](decoding_results[i]), training=False)
        for i in range(decoder_no_outputs)
    ]

    output_layers = (
            encoding_results +
            decoding_results +
            denoisers_mid +
            [tf.zeros_like(denoisers_mid[0])]) # TODO replace this with autoencoder results

    # create model
    model_hydra = \
        tf.keras.Model(
            inputs=[
                input_layer
            ],
            outputs=output_layers,
            name=f"hydra")

    # --- pack results
    return \
        BuilderResults(
            hydra=model_hydra,
            normalizer=backbone_results.normalizer,
            denormalizer=backbone_results.denormalizer,
            encoder=backbone_results.encoder,
            decoder=backbone_results.decoder,
            denoiser=model_denoiser
        )


# ---------------------------------------------------------------------


def model_backbone_builder(
        config: Dict,
        name_str: str = None) -> BackboneBuilderResults:
    """
    reads a configuration a model backbone

    :param config: configuration dictionary
    :param name_str: custom name, leave None to get the default

    :return: backbone, normalizer, denormalizer
    """
    logger.info("building backbone model with config [{0}]".format(config))

    # --- argument parsing
    model_type = config[TYPE_STR].strip().lower()
    value_range = config.get("value_range", (0, 255))
    input_shape = config.get(INPUT_SHAPE_STR, (None, None, 1))
    min_value = value_range[0]
    max_value = value_range[1]
    input_shape = input_shape_fixer(input_shape)
    if name_str is None or len(name_str) <= 0:
        name_str = f"{model_type}_backbone"

    # --- build normalize denormalize models
    model_normalize = \
        build_normalize_model(
            input_dims=input_shape,
            min_value=min_value,
            max_value=max_value)

    model_denormalize = \
        build_denormalize_model(
            input_dims=input_shape,
            min_value=min_value,
            max_value=max_value)

    if model_type == "unet":
        backbone_builder = builder_unet
    elif model_type in ["unet_plus", "unet+", "unet_p"]:
        backbone_builder = builder_unet_p
    else:
        raise ValueError(
            "don't know how to build model [{0}]".format(model_type))

    model_params = config[PARAMETERS_STR]

    model_encoder, model_decoder = \
        backbone_builder(
            input_dims=input_shape, **model_params)

    return (
        BackboneBuilderResults(
            encoder=model_encoder,
            decoder=model_decoder,
            normalizer=model_normalize,
            denormalizer=model_denormalize))

# ---------------------------------------------------------------------


def model_denoiser_builder(
        config: Dict,
        name: str = "denoiser_head") -> tf.keras.Model:
    """
    builds the denoiser model on top of the backbone layer

    :param config: dictionary with the denoiser configuration
    :param name: name of the model

    :return: denoiser head model
    """
    """
    builds the denoiser model on top of the backbone layer

    :param config: dictionary with the denoiser configuration

    :return: denoiser head model
    """
    # --- argument checking
    logger.info(f"building denoiser model with [{config}]")

    # --- set configuration
    output_channels = config.get("output_channels", 1)
    input_shape = input_shape_fixer(config.get(INPUT_SHAPE_STR))
    # use bias
    use_bias = config.get("use_bias", True)
    # output kernel size
    output_kernel = config.get("output_kernel", 1)

    conv_params = \
        dict(
            kernel_size=output_kernel,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            filters=output_channels,
            activation="linear",
            kernel_regularizer="l2",
            kernel_initializer="glorot_normal")

    # --- define denoiser network here
    model_input_layer = \
        tf.keras.Input(
            shape=input_shape,
            name=INPUT_TENSOR_STR)

    x = model_input_layer

    x = \
        conv2d_wrapper(
            input_layer=x,
            conv_params=conv_params)
    x = tf.nn.tanh(x) * 0.51

    model_output_layer = \
        tf.keras.layers.Layer(
            name="output_tensor")(x)

    model_head = \
        tf.keras.Model(
            inputs=model_input_layer,
            outputs=model_output_layer,
            name=name)

    return model_head

# ---------------------------------------------------------------------


def model_output_indices(no_outputs: int) -> Dict[str, List[int]]:
    """
    computes the indices for each head

    :param no_outputs:
    :return:
    """
    # --- argument checking
    if no_outputs <= 0:
        raise ValueError("no_outputs must be > 0")

    # --- compute the indices
    no_outputs_tmp = no_outputs - 1

    return {
        ENCODER_STR: [
            i for i in range(0, int(no_outputs_tmp / 3))
        ],
        DECODER_STR: [
            i for i in range(int(no_outputs_tmp / 3), 2 * int(no_outputs_tmp / 3))
        ],
        DENOISER_STR: [
            i for i in range(2 * int(no_outputs_tmp / 3), no_outputs_tmp)
        ],
        VARIATIONAL_AUTOENCODER_STR: [no_outputs-1]
    }

# ---------------------------------------------------------------------

import copy
import tensorflow as tf
from collections import namedtuple
from typing import Dict, Tuple, List

# ---------------------------------------------------------------------


def builder_scales_fusion(
        levels: int = 5,
        filters: int = 16,
        filters_level_multiplier: float = 2.0):
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
        filters_level_multiplier: float = 2.0):
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

import numpy as np
import tensorflow as tf
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


# ---------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable()
class Sampling(tf.layers.Layer):
    """
    use (z_mean, z_log_var) to sample z,
    the vector encoding a digit.
    """

    def __init__(self,
                 name="sampling",
                 **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ---------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    """
    Maps input to a triplet (z_mean, z_log_var, z).
    """

    def __init__(self,
                 latent_dim: int,
                 name: str = "encoder",
                 intermediate_activation: str = "relu",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._dense_proj = (
            tf.keras.layers.Dense(
                units=latent_dim * 2,
                activation=intermediate_activation))
        self._dense_mean = \
            tf.keras.layers.Dense(
                units=latent_dim)
        self._dense_log_var = \
            tf.keras.layers.Dense(units=latent_dim)
        self._sampling = Sampling()

    def call(self, inputs):
        x = self._dense_proj(inputs)
        z_mean = self._dense_mean(x)
        z_log_var = self._dense_log_var(x)
        z = self._sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


# ---------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    """
    Converts z, the encoded input vector, back into a readable input.
    """

    def __init__(self,
                 latent_dim: int,
                 target_shape: Tuple[int, int, int],
                 name="decoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._dense_proj = \
            tf.keras.layers.Dense(units=latent_dim, activation="linear")
        self._dense_output = \
            tf.keras.layers.Dense(units=np.prod(target_shape), activation="relu")
        self._reshape_layer = \
            tf.keras.layers.Reshape(target_shape=target_shape)

    def call(self, inputs):
        x = self._dense_proj(inputs)
        x = self._dense_output(x)
        x = self._reshape_layer(x)
        return x

# ---------------------------------------------------------------------

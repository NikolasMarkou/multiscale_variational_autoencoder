import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Iterable, Dict

# ==============================================================================
# local imports
# ==============================================================================

from .constants import *
from .custom_logger import logger
from . import schedule, layer_blocks, callbacks
from .utilities import conv2d_wrapper, ConvType


# ==============================================================================

@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """
    use (z_mean, z_log_var) to sample z,
    the vector encoding a digit.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ==============================================================================

@keras.saving.register_keras_serializable()
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
            layers.Dense(latent_dim * 2,
                         activation=intermediate_activation))
        self._dense_mean = layers.Dense(latent_dim)
        self._dense_log_var = layers.Dense(latent_dim)
        self._sampling = Sampling()

    def call(self, inputs):
        x = self._dense_proj(inputs)
        z_mean = self._dense_mean(x)
        z_log_var = self._dense_log_var(x)
        z = self._sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


# ==============================================================================

@keras.saving.register_keras_serializable()
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
        self._dense_proj = layers.Dense(latent_dim, activation="linear")
        self._dense_output = layers.Dense(np.prod(target_shape), activation="relu")
        self._reshape_layer = layers.Reshape(target_shape=target_shape, activation="sigmoid")

    def call(self, inputs):
        x = self._dense_proj(inputs)
        x = self._dense_output(x)
        x = self._reshape_layer(x)
        return x


# ==============================================================================

@keras.saving.register_keras_serializable()
class MultiscaleVae(tf.keras.Model):
    """
    Combines the encoder and decoder into an end-to-end model for training.
    """

    def __init__(
            self,
            input_dims,
            levels: int = 1,
            use_bias: bool = False,
            use_bn: bool = True,
            use_ln: bool = False,
            latent_dim: int = 32,
            name="autoencoder",
            **kwargs
    ):
        """
        builds a multiscale variational autoencoder model

        :param input_dims: Models input dimensions
        """
        super().__init__(name=name)

        logger.info("building vae")
        if len(kwargs) > 0:
            logger.info(f"parameters not used: {kwargs}")

        self._input_dims = input_dims

        # initialize loss trackers
        self._total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self._reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self._kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        # initialize models
        self._levels = levels
        self._encoders = []
        self._decoders = []
        self._models = []

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

        # for each level initialize encoder/decoder
        kernel_size = 16
        filters = 32
        activation = "relu"
        kernel_regularizer = "l2",
        kernel_initializer = "glorot_normal",

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
        same_level_conv_params = []
        up_level_conv_params = []
        for i in range(levels):
            params = copy.deepcopy(base_conv_params)
            params["filters"] = int(round(filters * (2.0 ** i)))
            conv_params.append(params)

            params = copy.deepcopy(base_conv_params)
            params["filters"] = int(round(filters * (2.0 ** i)))
            params["kernel_size"] = 1
            same_level_conv_params.append(params)

            params = copy.deepcopy(base_conv_params)
            params["filters"] = int(round(filters * (2.0 ** i)))
            params["kernel_size"] = kernel_size
            params["strides"] = (2, 2)
            up_level_conv_params.append(params)
        #
        self._encoders = []
        self._decoders = []
        self._shapes_encoded_decoder = []
        self._shapes_output = []
        input_layer = layers.InputLayer(input_shape=input_dims)
        x = input_layer
        x_up = []
        x_down = []
        x_level = x
        # down models
        for i in range(levels):
            x_level = \
                conv2d_wrapper(
                    input_layer=x_level,
                    bn_post_params=None,
                    ln_post_params=None,
                    conv_params=conv_params[i])

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
            encoder_decoder_shape = \
                (input_dims[0] // (2 ** i),
                 input_dims[1] // (2 ** i),
                 conv_params[i]["filters"])
            output_shape = \
                (input_dims[0] // (2 ** i),
                 input_dims[1] // (2 ** i),
                 input_dims[2])
            self._encoders.append(
                Encoder(latent_dim=latent_dim,
                        name=f"encoder_{i}"))
            self._decoders.append(
                Decoder(latent_dim=latent_dim,
                        target_shape=encoder_decoder_shape,
                        name=f"decoder_{i}"))
            self._shapes_encoded_decoder.append(encoder_decoder_shape)
            self._shapes_output.append(output_shape)

        self._model_down = tf.keras.Model(inputs=x, outputs=x_down, trainable=True)

        # up and merge
        for i in reversed(range(levels - 1)):
            x_down_i = x_down[i]

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
            # output multiscale
            x_level_output = conv2d_wrapper(
                input_layer=x_level,
                ln_post_params=ln_params,
                bn_post_params=bn_params,
                conv_params=conv_params[i])

            x_up.append(x_level_output)

        self._model_up = (
            tf.keras.Model(inputs=x_down, outputs=x_up[::-1], trainable=True))

        # compute different scales
        x = input_layer
        input_scales = [x]
        tmp_scale = x
        for i in range(self._levels - 1):
            tmp_scale = tf.nn.avg_pool2d(
                input=tmp_scale,
                ksize=(2, 2),
                strides=(2, 2),
                padding="SAME")
            input_scales.append(tmp_scale)
        self._model_scales = (
            tf.keras.Model(inputs=input_layer, outputs=input_scales, trainable=False))

    def call(self, inputs, trainable: bool = True):
        x = inputs

        x_down = self._model_down(x, trainable=trainable)
        x_enc_dec = []

        if trainable:
            for i in range(self._levels):
                encoder = self._encoders[i]
                decoder = self._decoder[i]
                z_mean, z_log_var, z = encoder(x_down[i])
                x_enc_dec.append(decoder(z))
                # Add KL divergence regularization loss.
                kl_loss = -0.5 * tf.reduce_mean(
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
                )
                self.add_loss(kl_loss)

            return self._model_up(x_enc_dec)

        for i in range(self._levels):
            encoder = self._encoders[i]
            decoder = self._decoder[i]
            _, _, z = encoder(x_down[i])
            x_enc_dec.append(decoder(z))

        return self._model_up(x_enc_dec)[0]

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._reconstruction_loss_tracker,
            self._kl_loss_tracker,
        ]

    def train_step(self, data):
        data_scales = self._model_scales(data, trainable=False)

        with tf.GradientTape() as tape:
            kl_loss = 0.0
            reconstruction_loss = 0.0
            x_down = self._model_down(data, trainable=True)
            x_enc_dec = []
            for i in range(self._levels):
                encoder = self._encoders[i]
                decoder = self._decoder[i]
                z_mean, z_log_var, z = encoder(x_down[i])
                x_enc_dec.append(decoder(z))
                kl_loss += tf.reduce_mean(
                    tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
            reconstructions = self._model_up(x_enc_dec, trainable=True)
            for i in range(self._levels):
                reconstruction_loss += tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.MAE(data_scales[i], reconstructions[i]), axis=(1, 2)
                    )
                )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._total_loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self._total_loss_tracker.result(),
            "reconstruction_loss": self._reconstruction_loss_tracker.result(),
            "kl_loss": self._kl_loss_tracker.result(),
        }

    def encode(self, inputs):
        pass

    def decode(self, inputs):
        pass

# ==============================================================================

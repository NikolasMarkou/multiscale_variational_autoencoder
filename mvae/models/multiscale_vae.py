import os
import keras
import logging
import numpy as np
from ..utils import callbacks
from ..utils import layer_blocks

# ===============================================================================
# setup logger
# ===============================================================================


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s")
logging.getLogger("multiscale-vae").setLevel(logging.INFO)
logger = logging.getLogger("multiscale-vae")


# ===============================================================================


class MultiscaleVariationalAutoencoder:
    def __init__(
            self,
            input_dims,
            levels,
            z_dims,
            channels_index=2,
            compress_output=True,
            encoder={
                "filters": [64, 64, 64, 64, 32],
                "kernel_size": [(3, 3), (3, 3), (3, 3,), (1, 1), (1, 1)],
                "strides": [(1, 1), (2, 2), (1, 1), (1, 1), (1, 1)]
            },
            decoder={
                "filters": [64, 64, 64, 64, 32],
                "kernel_size": [(3, 3), (3, 3), (3, 3,), (1, 1), (1, 1)],
                "strides": [(1, 1), (2, 2), (1, 1), (1, 1), (1, 1)]
            }):
        """
        Encoder that compresses a signal
        into a latent space of normally distributed variables
        :param input_dims:
        :param levels:
        :param z_dims:
        """
        # --------- Argument checking
        if levels <= 0:
            raise ValueError("levels should be integer and > 0")
        if len(z_dims) != levels:
            raise ValueError("z_dims should be a list of length levels")
        if not all(i > 0 for i in z_dims):
            raise ValueError("z_dims elements should be > 0")
        # --------- Variable initialization
        self._name = "mvae"
        self._levels = levels
        self._z_latent_dims = z_dims
        self._inputs_dims = input_dims
        self._encoder_config = encoder
        self._decoder_config = decoder
        self._compress_output = compress_output
        self._initialization_scheme = "glorot_uniform"
        self._output_channels = input_dims[channels_index]
        self._build()

    # ===============================================================================

    def _build(self):
        """
        Creates then multiscale VAE networks
        :return:
        """
        # --------- Build multiscale input
        self._scales = []
        self._input_layer = keras.Input(shape=self._inputs_dims, name="input_layer")
        layer = self._input_layer

        for i in range(self._levels):
            layer, up = self._downsample_upsample(layer, prefix="du_" + str(i) + "_")
            self._scales.append(up)

        # --------- Create Encoder / Decoder
        self._mu_log = []
        self._encoders = []
        self._decoder_input = []

        # --------- all encoders are the same
        shapes_before = []
        for i in range(self._levels):
            encoder, mu_log, shape_before_flattening = \
                self._build_encoder(self._scales[i],
                                    z_dim=self._z_latent_dims[i],
                                    prefix="encoder_" + str(i) + "_")
            self._mu_log.append(mu_log)
            self._encoders.append(encoder)
            shapes_before.append(shape_before_flattening)

        # --------- concat all z-latent layers
        logger.info("self._scales={0}".format(self._scales))
        logger.info("self._encoders={0}".format(self._encoders))
        self._z_latent_concat = keras.layers.Concatenate(axis=-1)(self._encoders)

        # --------- build decoder that uses all the combined latent variables
        self._decoder = \
            self._build_decoder(
                self._z_latent_concat,
                shape=shapes_before[0],
                prefix="decoder_")

        if self._compress_output:
            # -------- Cap output to [0, 1]
            self._result = keras.layers.Activation("sigmoid", name="output")(self._decoder)
        else:
            self._result = keras.layers.Layer(name="output")(self._decoder)

        # --------- The end-to-end trainable model
        self._model_trainable = keras.Model(self._input_layer,
                                            self._result)

        # # --------- The encoding model
        # self._model_encode = keras.Model(self._input_layer,
        #                                  self._z_latent_concat)
        #
        # # --------- The sample model
        # self._model_sample = keras.Model(self._z_latent_concat,
        #                                  self._result)

    # ===============================================================================

    @staticmethod
    def _downsample_upsample(i0, prefix="downsample_upsample"):
        """
        Downsample and upsample the input
        :param i0: input
        :return:
        """
        # --------- filter
        # TODO: Run a gaussian filter

        # --------- downsample
        d0 = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                       strides=None,
                                       padding="valid",
                                       name=prefix + "down")(i0)

        # --------- upsample
        u0 = keras.layers.UpSampling2D(size=(2, 2),
                                       interpolation="nearest",
                                       name=prefix + "up")(d0)
        # --------- diff
        diff = keras.layers.Subtract()([i0, u0])
        return d0, diff

    # ===============================================================================

    def _build_encoder(self,
                       encoder_input,
                       z_dim,
                       prefix="encoder_"):
        """
        Creates an encoder block
        :param encoder_input:
        :param z_dim:
        :param prefix:
        :return:
        """
        x = encoder_input

        # --------- Transforming here
        x = layer_blocks.basic_block(x,
                                     block_type="encoder",
                                     filters=self._encoder_config["filters"],
                                     kernel_size=self._encoder_config["kernel_size"],
                                     strides=self._encoder_config["strides"],
                                     prefix=prefix)
        # --------- Keep shape before flattening
        shape_before_flattening = keras.backend.int_shape(x)[1:]

        # --------- Global pool, flatten and convert to z_dim dimensions
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        mu = keras.layers.Dense(
            z_dim,
            kernel_initializer=self._initialization_scheme,
            name=prefix + "mu")(x)
        log_var = keras.layers.Dense(
            z_dim,
            kernel_initializer=self._initialization_scheme,
            name=prefix + "log_var")(x)

        def sample(args):
            tmp_mu, tmp_log_var = args
            epsilon = keras.backend.random_normal(
                shape=keras.backend.shape(tmp_mu),
                mean=0.,
                stddev=0.0001)
            return tmp_mu + keras.backend.exp(tmp_log_var / 2.0) * epsilon

        return keras.layers.Lambda(sample, name=prefix + "output")([mu, log_var]), \
               [mu, log_var], shape_before_flattening

    # ===============================================================================

    def _build_decoder(self,
                       decoder_input,
                       shape,
                       prefix="decoder_"):
        """
        Creates a decoder block
        :param decoder_input:
        :param shape:
        :param prefix:
        :return:
        """
        x = decoder_input
        # --------- Decoding here
        x = keras.layers.Dense(np.prod(shape))(x)
        x = keras.layers.Reshape(shape)(x)
        # --------- Transforming here
        x = layer_blocks.basic_block(input_layer=x,
                                     block_type="decoder",
                                     filters=self._decoder_config["filters"],
                                     kernel_size=self._decoder_config["kernel_size"],
                                     strides=self._decoder_config["strides"],
                                     prefix=prefix)
        # -------- Match target output channels
        x = keras.layers.Conv2D(filters=self._output_channels,
                                strides=(1, 1),
                                kernel_size=(1, 1),
                                kernel_initializer=self._initialization_scheme,
                                activation="linear")(x)
        return x

    # ===============================================================================

    def compile(
            self,
            learning_rate,
            r_loss_factor=1.0,
            kl_loss_factor=1.0,
            clipnorm=1.0):
        """

        :param learning_rate:
        :param r_loss_factor:
        :param kl_loss_factor:
        :param clipnorm:
        :return:
        """
        self._learning_rate = learning_rate
        # --------- Define VAE recreation loss
        def vae_r_loss(y_true, y_pred):
            r_loss = keras.backend.mean(
                keras.backend.abs(y_true - y_pred),
                axis=[1, 2, 3])
            # loss_ratio = float(np.prod(keras.backend.int_shape(y_pred)[1:])) / float(np.prod(self.inputs_dims))
            loss_ratio = 1.0
            return r_loss * r_loss_factor * loss_ratio

        # --------- Define KL loss for the latent space (difference from normally distributed m=0, var=1)
        def vae_kl_loss():
            # TODO
            return 0.0

        # --------- Define combined loss
        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss()
            return r_loss + kl_loss

        optimizer = keras.optimizers.Adagrad(
            lr=self._learning_rate,
            clipnorm=clipnorm
        )

        self._model_trainable.compile(
            optimizer=optimizer,
            loss=vae_loss,
            metrics=[vae_r_loss])

    # ===============================================================================

    def train(
            self,
            x_train,
            batch_size,
            epochs,
            run_folder,
            print_every_n_batches=100,
            initial_epoch=0,
            step_size=1,
            lr_decay=1,
            save_checkpoint_weights=False):
        custom_callback = callbacks.CustomCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            x_train[0:1, :, :, :],
            self)
        lr_schedule = callbacks.step_decay_schedule(
            initial_lr=self._learning_rate,
            decay_factor=lr_decay,
            step_size=step_size)
        checkpoint_filepath = os.path.join(
            run_folder,
            "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            save_weights_only=True,
            verbose=1)
        checkpoint2 = keras.callbacks.ModelCheckpoint(
            os.path.join(
                run_folder,
                "weights/weights.h5"),
            save_weights_only=True,
            verbose=1)

        callbacks_fns = [lr_schedule, custom_callback]

        if save_checkpoint_weights:
            callbacks_fns += [checkpoint1, checkpoint2]

        self._model_trainable.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_fns
        )

    # ===============================================================================

    def load_weights(self, filename):
        return

import os
import keras
import logging
import numpy as np
import scipy.stats as st
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
                "filters": [32, 32, 32],
                "kernel_size": [(3, 3), (3, 3), (1, 1)],
                "strides": [(1, 1), (1, 1), (1, 1)]
            },
            decoder={
                "filters": [32, 32, 32],
                "kernel_size": [(3, 3), (3, 3), (1, 1)],
                "strides": [(1, 1), (1, 1), (1, 1)]
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
        self._kernel_regularizer = None
        self._compress_output = compress_output
        self._initialization_scheme = "glorot_uniform"
        self._output_channels = input_dims[channels_index]
        self._build()

    # ==========================================================================

    def _build(self):
        """
        Creates then multiscale VAE networks
        :return:
        """
        # --------- Build multiscale input
        self._scales = []
        self._input_layer = keras.Input(shape=self._inputs_dims,
                                        name="input_layer")
        layer = self._input_layer

        for i in range(self._levels):
            if i == self._levels - 1:
                self._scales.append(layer)
            else:
                layer, up = self._downsample_upsample(layer,
                                                      prefix="du_" + str(i) + "_")
                self._scales.append(up)

        # --------- Create Encoder / Decoder
        self._mu_log = []
        self._decoders = []
        self._encoders = []
        self._decoder_input = []
        self._mu = None
        self._log_var = None
        # --------- all encoders are the same
        mu = []
        log_var = []
        for i in range(self._levels):
            encoder, mu_log, shape_before_flattening = \
                self._build_encoder(self._scales[i],
                                    z_dim=self._z_latent_dims[i],
                                    prefix="encoder_" + str(i) + "_")
            mu.append(mu_log[0])
            log_var.append(mu_log[1])
            self._mu_log.append(mu_log)
            self._encoders.append(encoder)

            decoder = \
                self._build_decoder(
                    encoder,
                    shape=shape_before_flattening,
                    prefix="decoder_" + str(i) + "_")

            self._decoders.append(decoder)

        # --------- upsample and merge decoders
        results = []
        for i in range(self._levels):
            layer = self._decoders[i]
            size = (2 ** i, 2 ** i)
            layer = keras.layers.UpSampling2D(size=size,
                                              interpolation="bilinear")(layer)
            results.append(layer)
        x = keras.layers.Add()(results)

        # --------- concat all z-latent layers
        self._z_latent_concat = \
            keras.layers.Concatenate(axis=-1)(self._encoders)
        self._mu = \
            keras.layers.Concatenate(axis=-1, name="concat_mu")(mu)
        self._log_var = \
            keras.layers.Concatenate(axis=-1, name="concat_log_var")(log_var)

        if self._compress_output:
            # -------- Cap output to [0, 1]
            x = keras.layers.Activation("sigmoid")(x)

        self._result = keras.layers.Layer(name="output")(x)

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
        # TODO : fix for different tensor schemes
        shape = keras.backend.int_shape(i0)[1:3]
        logger.info("shape={0}".format(shape))
        shape = (int(shape[0] / 4), int(shape[1] / 4))
        f0 = MultiscaleVariationalAutoencoder._gaussian_filter(shape)(i0)

        # --------- downsample
        d0 = keras.layers.MaxPool2D(pool_size=(1, 1),
                                    strides=(2, 2),
                                    padding="valid",
                                    name=prefix + "down")(f0)

        # --------- upsample
        u0 = keras.layers.UpSampling2D(size=(2, 2),
                                       interpolation="bilinear",
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
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=None,
            kernel_initializer="glorot_normal")(x)

        # --------- Transforming here
        x = layer_blocks.basic_block(x,
                                     block_type="encoder",
                                     filters=self._encoder_config["filters"],
                                     kernel_size=self._encoder_config["kernel_size"],
                                     strides=self._encoder_config["strides"],
                                     prefix=prefix,
                                     use_dropout=False,
                                     use_batchnorm=False)
        # --------- Keep shape before flattening
        shape_before_flattening = keras.backend.int_shape(x)[1:]

        # --------- flatten and convert to z_dim dimensions
        initializer = keras.initializers.orthogonal()
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        mu = keras.layers.Dense(
            z_dim,
            kernel_regularizer=None,
            kernel_initializer=initializer,
            name=prefix + "mu")(x)
        log_var = keras.layers.Dense(
            z_dim,
            kernel_regularizer=None,
            kernel_initializer=initializer,
            name=prefix + "log_var")(x)

        def sample(args):
            tmp_mu, tmp_log_var = args
            epsilon = keras.backend.random_normal(
                shape=keras.backend.shape(tmp_mu),
                mean=0.,
                stddev=0.01)
            return tmp_mu + keras.backend.exp(tmp_log_var) * epsilon

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
        # -------- Add batchnorm to boost signal
        x = keras.layers.BatchNormalization()(x)

        # -------- Match target output channels
        x = keras.layers.Conv2D(filters=self._output_channels,
                                strides=(1, 1),
                                kernel_size=(1, 1),
                                kernel_initializer="glorot_normal",
                                activation="linear")(x)
        return x

    # ===============================================================================

    def compile(self,
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

        # --------- Define VAE reconstruction loss
        def vae_r_loss(y_true, y_pred):
            tmp0 = keras.backend.abs(y_true - y_pred)
            return keras.backend.mean(tmp0, axis=[1, 2, 3])

        # --------- Define KL loss for the latent space
        # (difference from normally distributed m=0, var=1)
        def vae_kl_loss(y_true, y_pred):
            tmp = 1 + self._log_var[1] - \
                  keras.backend.square(self._mu[0]) - \
                  keras.backend.exp(self._log_var[1])
            tmp = keras.backend.sum(tmp, axis=-1)
            tmp *= -0.5
            return keras.backend.mean(tmp)

        # --------- Define combined loss
        def vae_loss(y_true, y_pred):
            kl_loss = vae_kl_loss(y_true, y_pred)
            r_loss = vae_r_loss(y_true, y_pred)
            return (r_loss * r_loss_factor) + (kl_loss * kl_loss_factor)

        optimizer = keras.optimizers.Adagrad(
            lr=self._learning_rate,
            clipnorm=clipnorm)

        self._model_trainable.compile(
            optimizer=optimizer,
            loss=vae_loss,
            metrics=[vae_r_loss, vae_kl_loss])

    # ===============================================================================

    def train(self,
              x_train,
              batch_size,
              epochs,
              run_folder,
              print_every_n_batches=100,
              initial_epoch=0,
              step_size=1,
              lr_decay=1,
              save_checkpoint_weights=False):
        """

        """
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
        weights_path = os.path.join(run_folder, "weights")
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
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
            callbacks=callbacks_fns)

    # ===============================================================================

    def load_weights(self, filename):
        return

    @staticmethod
    def _gaussian_kernel(kernlen=[21, 21], nsig=[3, 3]):
        """
        Returns a 2D Gaussian kernel array
        """
        assert len(nsig) == 2
        assert len(kernlen) == 2
        kern1d = []
        for i in range(2):
            interval = (2 * nsig[i] + 1.) / (kernlen[i])
            x = np.linspace(-nsig[i] - interval / 2., nsig[i] + interval / 2.,
                            kernlen[i] + 1)
            kern1d.append(np.diff(st.norm.cdf(x)))

        kernel_raw = np.sqrt(np.outer(kern1d[0], kern1d[1]))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    @staticmethod
    def _gaussian_filter(kernel_size):
        # Initialise to set kernel to required value
        def kernel_init(shape, dtype):
            kernel = np.zeros(shape)
            kernel[:, :, 0, 0] = \
                MultiscaleVariationalAutoencoder._gaussian_kernel(
                    [shape[0], shape[1]])
            return kernel

        return keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            depth_multiplier=1,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=False,
            trainable=False,
            depthwise_initializer=kernel_init,
            kernel_initializer=kernel_init)

    # ===============================================================================


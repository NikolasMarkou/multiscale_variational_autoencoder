import os
import keras
import numpy as np
from keras import backend as K
from .custom_logger import logger
from . import schedule, layer_blocks, callbacks

# ==============================================================================


class MultiscaleVAE:
    def __init__(
            self,
            input_dims,
            levels,
            z_dims,
            channels_index=2,
            compress_output=True,
            encoder={
                "filters": [32],
                "kernel_size": [(3, 3)]
            },
            decoder={
                "filters": [32],
                "kernel_size": [(3, 3)]
            }):
        """
        Encoder that compresses a signal
        into a latent space of normally distributed variables
        :param input_dims: HxWxC
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
        self._share_z_space = False
        self._conv_base_filters = 32
        self._conv_activation = "relu"
        self._z_latent_dims = z_dims
        self._inputs_dims = input_dims
        self._encoder_config = encoder
        self._decoder_config = decoder
        self._gaussian_kernel = [5, 5]
        self._clip_min_value = (1.0 / 255.0)
        self._clip_max_value = 255.0
        self._training_noise_std = (1.0 / 255.0) / 2.0
        self._compress_output = compress_output
        self._initialization_scheme = "glorot_normal"
        self._output_channels = input_dims[channels_index]
        self._kernel_regularizer = "l2"
        self._dense_regularizer = None
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
        if self._training_noise_std is not None:
            if self._training_noise_std > 0.0:
                layer = keras.layers.GaussianNoise(
                    self._training_noise_std)(layer)

        for i in range(self._levels):
            if i == self._levels - 1:
                self._scales.append(layer)
            else:
                layer, up = self._downsample_upsample(
                    layer, prefix="du_" + str(i) + "_")
                self._scales.append(up)

        # --------- Create Encoder / Decoder
        self._mu = None
        self._log_var = None
        self._decoders = []
        self._encoders = []
        self._encoders_no_random = []
        # --------- encoders
        mu = []
        log_var = []
        shapes_before_flattening = []
        for i in range(self._levels):
            encoder, encoder_no_random, mu_log, shape_before_flattening = \
                self._build_encoder(self._scales[i],
                                    z_dim=self._z_latent_dims[i],
                                    prefix="encoder_" + str(i) + "_")
            mu.append(mu_log[0])
            log_var.append(mu_log[1])
            shapes_before_flattening.append(shape_before_flattening)
            self._encoders.append(encoder)
            self._encoders_no_random.append(encoder_no_random)
        # --------- concat all z-latent layers
        self._z_latent_concat = \
            keras.layers.Concatenate(axis=-1)(self._encoders)
        self._z_latent_no_random_concat = \
            keras.layers.Concatenate(axis=-1)(self._encoders_no_random)
        self._mu = \
            keras.layers.Concatenate(axis=-1, name="concat_mu")(mu)
        self._log_var = \
            keras.layers.Concatenate(axis=-1, name="concat_log_var")(log_var)
        # --------- decoders
        if self._share_z_space:
            z_no_gradients = [K.stop_gradient(z)
                              for z in self._encoders]
            for i in range(self._levels):
                # --------- get encodings of all
                # except this one and stop gradients
                # this way they can all share information on the z-space
                z_tmp = z_no_gradients[:]
                z_tmp[i] = self._encoders[i]
                decoder = \
                    self._build_decoder(
                        keras.layers.Concatenate()(z_tmp),
                        target_shape=shapes_before_flattening[i],
                        prefix="decoder_" + str(i) + "_")
                self._decoders.append(decoder)
        else:
            for i in range(self._levels):
                decoder = \
                    self._build_decoder(
                        self._encoders[i],
                        target_shape=shapes_before_flattening[i],
                        prefix="decoder_" + str(i) + "_")
                self._decoders.append(decoder)
        # --------- upsample and merge decoders
        layer = None
        for i in range(self._levels-1, -1, -1):
            if i == self._levels - 1:
                layer = self._decoders[i]
            else:
                x = keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="bilinear")(layer)
                x = keras.layers.Add()(
                    [x, self._decoders[i]])
                x = keras.layers.ReLU()(x)
                x = keras.layers.DepthwiseConv2D(
                    depth_multiplier=1,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding="same",
                    activation=self._conv_activation,
                    depthwise_regularizer=self._kernel_regularizer,
                    depthwise_initializer=self._initialization_scheme,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._initialization_scheme)(x)
                layer = x
        x = keras.layers.Conv2D(
            filters=self._output_channels,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation="linear",
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._initialization_scheme)(x)

        if self._compress_output:
            # -------- Cap output to [0, 1]
            x = keras.layers.Activation("sigmoid")(x)

        self._result = keras.layers.Layer(name="output")(x)

        # --------- The end-to-end trainable model
        self._model_trainable = keras.Model(self._input_layer,
                                            self._result)

        # --------- The encoding model
        self._model_encode = keras.Model(self._input_layer,
                                         self._z_latent_no_random_concat)

        # # --------- The sample model
        # self._model_sample = keras.Model(self._z_latent_no_random_concat,
        #                                  self._result)

    # ==========================================================================

    def _downsample_upsample(self,
                             i0,
                             prefix="downsample_upsample"):
        """
        Downsample and upsample the input
        :param i0: input
        :return:
        """
        # --------- filter and downsample
        f0 = layer_blocks.gaussian_filter_block(
            i0, kernel_size=self._gaussian_kernel, strides=(1, 1))

        # --------- downsample
        d0 = keras.layers.MaxPool2D(pool_size=(1, 1),
                                    strides=(2, 2),
                                    padding="valid",
                                    name=prefix + "down")(f0)

        # --------- diff
        diff = keras.layers.Subtract()([i0, f0])
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
            filters=self._conv_base_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=self._conv_activation,
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._initialization_scheme)(x)

        # --------- Transforming here
        x = layer_blocks.basic_block(
            x,
            block_type="encoder",
            filters=self._encoder_config["filters"],
            kernel_size=self._encoder_config["kernel_size"],
            strides=self._encoder_config["strides"],
            prefix=prefix,
            use_dropout=False,
            use_batchnorm=False)
        # --------- Keep shape before flattening
        shape_before_flattening = K.int_shape(x)[1:]

        # --------- flatten and convert to z_dim dimensions
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        mu = keras.layers.Dense(
            z_dim,
            kernel_regularizer=self._dense_regularizer,
            kernel_initializer=self._initialization_scheme,
            name=prefix + "mu")(x)
        log_var = keras.layers.Dense(
            z_dim,
            activation="relu",
            kernel_regularizer=self._dense_regularizer,
            kernel_initializer=self._initialization_scheme,
            name=prefix + "log_var")(x)

        def sample(args):
            tmp_mu, tmp_log_var = args
            epsilon = K.random_normal(
                shape=K.shape(tmp_mu),
                mean=0.,
                stddev=0.001)
            return tmp_mu + K.exp(tmp_log_var) * epsilon

        def dont_sample(args):
            tmp_mu, tmp_log_var = args
            return tmp_mu + K.exp(tmp_log_var)

        return \
            keras.layers.Lambda(
                sample, name=prefix + "_sample_output")([mu, log_var]), \
            keras.layers.Lambda(
                dont_sample, name=prefix + "_dont_sample_output")([mu, log_var]), \
            [mu, log_var], \
            shape_before_flattening

    # ==========================================================================

    def _build_decoder(self,
                       input_layer,
                       target_shape,
                       prefix="decoder_"):
        """
        Creates a decoder block
        :param input_layer:
        :param target_shape: HxWxC
        :param prefix:
        :return:
        """
        # --------- Decoding here
        x = keras.layers.Dense(
            units=np.prod(target_shape),
            activation="linear",
            kernel_initializer=self._initialization_scheme,
            kernel_regularizer=self._dense_regularizer)(input_layer)

        x = keras.layers.Reshape(target_shape)(x)

        # --------- Transforming here
        x = layer_blocks.basic_block(
            input_layer=x,
            block_type="decoder",
            filters=self._decoder_config["filters"],
            kernel_size=self._decoder_config["kernel_size"],
            strides=self._decoder_config["strides"],
            prefix=prefix)

        # -------- Add batchnorm to boost signal
        x = keras.layers.BatchNormalization()(x)

        # -------- Match target channels
        x = keras.layers.Conv2D(
            filters=self._conv_base_filters,
            strides=(1, 1),
            kernel_size=(1, 1),
            padding="same",
            activation="linear",
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._initialization_scheme)(x)

        return x

    # ==========================================================================

    def compile(self,
                learning_rate,
                r_loss_factor=1.0,
                kl_loss_factor=1.0,
                clip_norm=1.0):
        """

        :param learning_rate:
        :param r_loss_factor:
        :param kl_loss_factor:
        :param clip_norm:
        :return:
        """
        self.learning_rate = learning_rate

        # --------- Define VAE reconstruction loss
        def vae_r_loss(y_true, y_pred):
            tmp0 = K.abs(y_true - y_pred)
            return K.mean(tmp0, axis=[1, 2, 3])

        def vae_r_experimental(y_true, y_pred):
            x = K.clip(x=K.abs(y_true - y_pred),
                       min_value=self._clip_min_value,
                       max_value=self._clip_max_value)
            return K.mean(x, axis=[1, 2, 3])

        # --------- Define KL loss for the latent space
        # (difference from normally distributed m=0, var=1)
        def vae_kl_loss(y_true, y_pred):
            tmp = 1 + self._log_var[1] - \
                  K.square(self._mu[0]) - \
                  K.exp(self._log_var[1])
            tmp = K.sum(tmp, axis=-1)
            tmp *= -0.5
            return K.mean(tmp)

        # --------- Define combined loss
        def vae_loss(y_true, y_pred):
            kl_loss = vae_kl_loss(y_true, y_pred)
            r_loss = vae_r_experimental(y_true, y_pred)
            return (r_loss * r_loss_factor) + (kl_loss * kl_loss_factor)

        optimizer = keras.optimizers.Adagrad(
            lr=self._learning_rate,
            clipnorm=clip_norm)

        self._model_trainable.compile(
            optimizer=optimizer,
            loss=vae_loss,
            metrics=[vae_r_loss, vae_kl_loss])

    # ==========================================================================

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
        custom_callback = callbacks.SaveIntermediateResultsCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            x_train[0:16, :, :, :],
            self)
        lr_schedule = schedule.step_decay_schedule(
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

    # ==========================================================================

    def load_weights(self, filename):
        return

    # ==========================================================================

    @property
    def model_trainable(self):
        return self._model_trainable

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    # ==========================================================================
    @property
    def model_encode(self):
        return self._model_encode

    @property
    def model_sample(self):
        return self._model_sample

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
            z_dims,
            compress_output=False,
            encoder={
                "filters": [32],
                "kernel_size": [(3, 3)],
                "strides": [(1, 1)]
            },
            decoder=None,
            min_value=0.0,
            max_value=255.0,
            sample_std=0.01,
            channels_index=2):
        """
        Encoder that compresses a signal
        into a latent space of normally distributed variables
        :param input_dims: HxWxC
        :param levels:
        :param z_dims:
        """
        # --- argument checking
        if encoder is None:
            raise ValueError("encoder cannot be None")
        if not all(i > 0 for i in z_dims):
            raise ValueError("z_dims elements should be > 0")

        # --- decoder is reverse encoder
        if decoder is None:
            decoder = {
                "filters": encoder["filters"][::-1],
                "strides": encoder["strides"][::-1],
                "kernel_size": encoder["kernel_size"][::-1]
            }
        levels = len(z_dims)

        # --- Variable initialization
        self._name = "mvae"
        self._levels = levels
        self._conv_base_filters = 32
        self._conv_activation = "relu"
        self._z_latent_dims = z_dims
        self._inputs_dims = input_dims
        self._encoder_config = encoder
        self._decoder_config = decoder
        self._gaussian_kernel = (3, 3)
        self._gaussian_nsig = (2, 2)
        self._training_dropout = 0.1
        self._training_noise_std = 1.0 / (max_value - min_value)
        self._compress_output = compress_output
        self._initialization_scheme = "glorot_normal"
        self._output_channels = input_dims[channels_index]
        self._kernel_regularizer = "l2"
        self._dense_regularizer = "l2"
        self._min_value = min_value
        self._max_value = max_value
        self._sample_std = sample_std
        self._channels_index = channels_index
        self._build()

    # ==========================================================================

    def _build(self):
        """
        Creates then multiscale VAE networks
        :return:
        """

        def normalize(args):
            """
            Convert input from [v0, v1] to [-1, +1] range
            """
            y, v0, v1 = args
            return 2.0 * (y - v0) / (v1 - v0) - 1.0

        def denormalize(args):
            """
            Convert input [-1, +1] to [v0, v1] range
            """
            y, v0, v1 = args
            return K.clip(
                (y + 1.0) * (v1 - v0) / 2.0 + v0,
                min_value=v0,
                max_value=v1)

        def split(args):
            """
            Split one dimensional input to set parts
            """
            y, z_dims = args
            start = 0
            result = []
            for z in z_dims:
                result.append(y[:, start:(start + z)])
                start += z
            return result

        # --- Build multi-scale input
        logger.info("Building multi-scale input")

        def compute_scales(
                input_dims=self._inputs_dims,
                levels=self._inputs_dims):
            result = []
            for i in range(levels):
                if i == 0:
                    result.append(input_dims)
                else:
                    s = tuple()
                    for j in range(len(result[i - 1])):
                        if j == self._channels_index:
                            s += (result[i - 1][j],)
                        else:
                            s += (int(result[i - 1][j] / 2),)
                    result.append(s)
            return result

        def input_transform(
                input_dims=self._inputs_dims,
                levels=self._levels,
                min_value=self._min_value,
                max_value=self._max_value,
                training_noise=self._training_noise_std,
                training_dropout=self._training_dropout):
            input_layer = keras.Input(shape=input_dims)
            layer = keras.layers.Lambda(normalize, name="normalize")([
                input_layer, min_value, max_value])

            if training_noise is not None:
                if training_noise > 0.0:
                    layer = keras.layers.GaussianNoise(
                        training_noise)(layer)

            if training_dropout is not None:
                if training_dropout > 0.0:
                    layer = keras.layers.SpatialDropout2D(
                        training_dropout)(layer)

            result = []
            for i in range(levels):
                if i == levels - 1:
                    result.append(layer)
                else:
                    layer, up = self._downsample_upsample(
                        layer, prefix="du_" + str(i) + "_")
                    result.append(up)

            return keras.Model(inputs=input_layer,
                               outputs=result,
                               name="multiscale")

        # --- Create Encoder / Decoder
        self._mu = None
        self._log_var = None

        # --- encoders
        logger.info("Building encoder")
        encoders = []
        shapes_before_flattening = []
        scales = compute_scales(self._inputs_dims, self._levels)

        for i in range(self._levels):
            logger.info("Encoder scale [{0}]".format(i))
            encoder_input = keras.Input(shape=scales[i],
                                        name="encoder_" + str(i) + "_input")
            encoder_output, mu_log, shape = \
                self._build_encoder(encoder_input,
                                    sample_stddev=self._sample_std,
                                    z_dim=self._z_latent_dims[i],
                                    prefix="encoder_" + str(i) + "_")
            shapes_before_flattening.append(shape)
            encoders.append(
                keras.Model(inputs=encoder_input, outputs=[encoder_output,
                                                           mu_log[0],
                                                           mu_log[1]]))

        # --- decoders
        logger.info("Building decoder")
        decoders = []
        for i in range(self._levels):
            logger.info("Decoder scale [{0}]".format(i))
            decoder_input = keras.Input(shape=(self._z_latent_dims[i],),
                                        name="decoder_" + str(i) + "_input")
            decoder_output = \
                self._build_decoder(
                    decoder_input,
                    target_shape=shapes_before_flattening[i],
                    prefix="decoder_" + str(i) + "_")
            decoders.append(
                keras.Model(inputs=decoder_input, outputs=decoder_output))

        # --- upsample and merge decoders
        logger.info("Building merge decoder")
        output_merge_inputs = [
            keras.Input(shape=scales[i])
            for i in range(self._levels)
        ]
        layer = None

        for i in range(self._levels - 1, -1, -1):
            if i == self._levels - 1:
                layer = output_merge_inputs[i]
            else:
                x = keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="bilinear")(layer)
                x = keras.layers.Add()(
                    [x, output_merge_inputs[i]])
                layer = x
        # --- Bring bang to initial value range
        x = keras.layers.Lambda(denormalize, name="denormalize")(
            [x, self._min_value, self._max_value])
        output_merge_inputs = keras.Model(inputs=output_merge_inputs,
                                          outputs=x)

        # --- build encoder model
        logger.info("Building encoder model")
        model_encoder_input = keras.Input(shape=self._inputs_dims,
                                          name="input")
        input_transform_encoder = input_transform(input_dims=self._inputs_dims,
                                                  levels=self._levels,
                                                  min_value=self._min_value,
                                                  max_value=self._max_value,
                                                  training_noise=None,
                                                  training_dropout=None)
        model_encoder_input_multiscale = input_transform_encoder(model_encoder_input)
        model_encoder_scale = [
            encoders[i](model_encoder_input_multiscale[i])[0]
            for i in range(self._levels)
        ]
        model_encoder_output = keras.layers.Concatenate()(model_encoder_scale)
        self._model_encoder = keras.Model(inputs=model_encoder_input,
                                          outputs=model_encoder_output)

        # --- build decoder model
        logger.info("Building decoder model")
        model_decoder_input = keras.Input(shape=(np.sum(self._z_latent_dims),),
                                          name="input")
        model_decoder_split = keras.layers.Lambda(split, name="split")(
            [model_decoder_input, self._z_latent_dims])
        model_decoder_decode = []
        for i in range(self._levels):
            decode = decoders[i](model_decoder_split[i])
            model_decoder_decode.append(decode)
        model_decoder_output = output_merge_inputs(model_decoder_decode)
        self._model_decoder = keras.Model(inputs=model_decoder_input,
                                          outputs=model_decoder_output)

        # --- build end-to-end trainable model
        logger.info("Building end-to-end trainable model")
        mu = []
        log_var = []
        model_encode_decode = []
        model_input = keras.Input(shape=self._inputs_dims, name="input")
        model_input_transform = input_transform(input_dims=self._inputs_dims,
                                                levels=self._levels,
                                                min_value=self._min_value,
                                                max_value=self._max_value,
                                                training_noise=self._training_noise_std,
                                                training_dropout=self._training_dropout)
        model_input_multiscale = model_input_transform(model_input)

        for i in range(self._levels):
            enc = encoders[i](model_input_multiscale[i])
            encode = enc[0]
            mu.append(enc[1])
            log_var.append(enc[2])
            decode = decoders[i](encode)
            model_encode_decode.append(decode)
        model_output = output_merge_inputs(model_encode_decode)
        self._model_trainable = keras.Model(inputs=model_input,
                                            outputs=model_output)

        # --- concat all z-latent layers
        self._mu = \
            keras.layers.Concatenate(axis=-1, name="mu")(mu)
        self._log_var = \
            keras.layers.Concatenate(axis=-1, name="log_var")(log_var)

    # ==========================================================================

    def _downsample_upsample(self,
                             i0,
                             prefix="downsample_upsample"):
        """
        Downsample and upsample the input
        :param i0: input
        :return:
        """

        # --- filter and downsample
        f0 = layer_blocks.gaussian_filter_block(
            i0,
            strides=(1, 1),
            xy_max=self._gaussian_nsig,
            kernel_size=self._gaussian_kernel)

        # --- downsample
        d0 = keras.layers.MaxPool2D(pool_size=(1, 1),
                                    strides=(2, 2),
                                    padding="valid",
                                    name=prefix + "down")(f0)

        # --- diff
        diff = keras.layers.Subtract()([i0, f0])
        return d0, diff

    # ===============================================================================

    def _build_encoder(self,
                       encoder_input,
                       z_dim,
                       sample_stddev=0.01,
                       prefix="encoder_"):
        """
        Creates an encoder block

        :param encoder_input:
        :param z_dim:
        :param prefix:
        :return:
        """
        # --- Transforming here
        x = keras.layers.Conv2D(
            filters=self._conv_base_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name=prefix + "conv_base",
            activation=self._conv_activation,
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._initialization_scheme)(encoder_input)

        # --- Transforming here
        x = layer_blocks.basic_block(
            x,
            block_type="encoder",
            filters=self._encoder_config["filters"],
            kernel_size=self._encoder_config["kernel_size"],
            strides=self._encoder_config["strides"],
            prefix=prefix,
            use_dropout=False,
            use_batchnorm=True)

        # --- Keep shape before flattening
        shape_before_flattening = K.int_shape(x)[1:]

        # --- flatten and convert to z_dim dimensions
        x = keras.layers.Flatten()(x)
        mu = keras.layers.Dense(
            z_dim,
            activation="linear",
            kernel_regularizer=self._dense_regularizer,
            kernel_initializer=self._initialization_scheme,
            name=prefix + "mu")(x)
        log_var = keras.layers.Dense(
            z_dim,
            activation="linear",
            kernel_regularizer=self._dense_regularizer,
            kernel_initializer=self._initialization_scheme,
            name=prefix + "log_var")(x)

        def sample(args):
            tmp_mu, tmp_log_var, tmp_stddev = args
            epsilon = K.random_normal(
                shape=K.shape(tmp_mu),
                mean=0.,
                stddev=tmp_stddev)
            return tmp_mu + K.exp(tmp_log_var) * epsilon

        return \
            keras.layers.Lambda(
                sample, name=prefix + "_sample_output")(
                [mu, log_var, sample_stddev]), \
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
        # --- Decoding here
        x = keras.layers.Dense(
            units=np.prod(target_shape),
            activation="linear",
            kernel_initializer=self._initialization_scheme,
            kernel_regularizer=self._dense_regularizer)(input_layer)

        x = keras.layers.Reshape(target_shape)(x)

        # --- Transforming here
        x = layer_blocks.basic_block(
            input_layer=x,
            block_type="decoder",
            use_batchnorm=True,
            filters=self._decoder_config["filters"],
            kernel_size=self._decoder_config["kernel_size"],
            strides=self._decoder_config["strides"],
            prefix=prefix)

        # --- Match target channels
        x = keras.layers.Conv2D(
            filters=self._output_channels,
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

        # --- Define VAE reconstruction loss
        def vae_r_loss(y_true, y_pred):
            # focus on matching all pixels
            tmp_pixels = K.abs(y_true - y_pred)
            return K.mean(tmp_pixels)

        # --- Define KL loss for the latent space
        # (difference from normally distributed m=0, var=1)
        def vae_kl_loss(y_true, y_pred):
            x = 1.0 + self._log_var - K.square(self._mu) - K.exp(self._log_var)
            x = -0.5 * K.sum(x, axis=-1)
            return x

        # --- Define combined loss
        def vae_loss(y_true, y_pred):
            kl_loss = vae_kl_loss(y_true, y_pred)
            r_loss = vae_r_loss(y_true, y_pred)
            return (r_loss * r_loss_factor) + \
                   (kl_loss * kl_loss_factor)

        optimizer = keras.optimizers.Adagrad(
            lr=self.learning_rate,
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
    def encoder(self):
        return self._model_encoder

    @property
    def decoder(self):
        return self._model_decoder

    @property
    def model_trainable(self):
        return self._model_trainable

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def normalize(self, v):
        return (v - self._min_value) / (self._max_value - self._min_value)

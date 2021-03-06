import os
import math
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

# ==============================================================================

from .custom_logger import logger
from . import schedule, layer_blocks, callbacks, utilities

# ==============================================================================


class MultiscaleVAE:
    def __init__(
            self,
            input_dims,
            z_dims: list,
            compress_output: bool = False,
            encoder={
                "filters": [32],
                "kernel_size": [(3, 3)],
                "strides": [(1, 1)]
            },
            decoder=None,
            min_value: float = 0.0,
            max_value: float = 255.0,
            sample_std: float = 1.0,
            channels_index: int = 2):
        """
        Encoder that compresses a signal
        into a latent space of normally distributed variables
        :param input_dims: HxWxC
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
        self._dense_encoding = False
        self._conv_base_filters = 32
        self._conv_activation = "relu"
        self._z_latent_dims = z_dims
        self._inputs_dims = input_dims
        self._encoder_config = encoder
        self._decoder_config = decoder
        self._gaussian_kernel = (3, 3)
        self._gaussian_nsig = (0.5, 0.5)
        self._training_dropout = 0.0
        self._training_noise_std = 0.0
        self._compress_output = compress_output
        self._initialization_scheme = "glorot_normal"
        self._output_channels = input_dims[channels_index]
        self._kernel_regularizer = "l1"
        self._dense_regularizer = "l1"
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
            encoder_input = \
                keras.Input(
                    shape=scales[i],
                    name=f"encoder_{i}_input")
            encoder_output, mu_log, shape = \
                self._build_encoder(
                    encoder_input,
                    sample_stddev=self._sample_std,
                    z_dim=self._z_latent_dims[i],
                    prefix=f"encoder_{i}_")
            shapes_before_flattening.append(shape)
            encoders.append(
                keras.Model(
                    name=f"encoder_{i}",
                    inputs=encoder_input,
                    outputs=[
                        encoder_output,
                        mu_log[0],
                        mu_log[1]
                    ]
                )
            )

        # --- decoders
        logger.info("Building decoder")
        decoders = []
        for i in range(self._levels):
            logger.info(f"Decoder scale [{i}]")
            decoder_input = \
                keras.Input(
                    shape=(self._z_latent_dims[i],),
                    name=f"decoder_{i}_input")
            decoder_output = \
                self._build_decoder(
                    decoder_input,
                    target_shape=shapes_before_flattening[i],
                    prefix=f"decoder_{i}_")
            decoders.append(
                keras.Model(
                    name=f"decoder_{i}",
                    inputs=decoder_input,
                    outputs=decoder_output))

        # --- upsample and merge decoder outputs
        logger.info("Building decoder merge")

        model_decoder_merge = \
            layer_blocks.laplacian_transform_merge(
                trainable=True,
                input_dims=scales,
                levels=self._levels,
                min_value=self._min_value,
                max_value=self._max_value,
                filters=self._conv_base_filters,
                activation=self._conv_activation,
                name="laplacian_merge")

        # --- build encoder model
        logger.info("Building encoder model")
        encoder_input = \
            keras.Input(
                shape=self._inputs_dims,
                name="input")

        model_laplacian_split = \
            layer_blocks.laplacian_transform_split(
                levels=self._levels,
                min_value=self._min_value,
                max_value=self._max_value,
                input_dims=self._inputs_dims,
                gaussian_xy_max=self._gaussian_nsig,
                gaussian_kernel_size=self._gaussian_kernel,
                name="laplacian_split")

        encoder_input_multiscale = \
            model_laplacian_split(encoder_input)
        encoder_per_level = [
            encoders[i](encoder_input_multiscale[i])[0]
            for i in range(self._levels)
        ]
        encoder_output = \
            keras.layers.Concatenate()(encoder_per_level)
        self._model_encoder = \
            keras.Model(
                name="encoder",
                inputs=encoder_input,
                outputs=encoder_output)

        # --- build decoder model
        logger.info("Building decoder model")
        decoder_input = \
            keras.Input(
                shape=(np.sum(self._z_latent_dims),),
                name="input")
        decoder_input_split = \
            keras.layers.Lambda(split, name="split")(
                [decoder_input, self._z_latent_dims])
        decoder_decode = [
            decoders[i](decoder_input_split[i])
            for i in range(self._levels)
        ]
        decoder_output = \
            model_decoder_merge(decoder_decode)
        self._model_decoder = \
            keras.Model(
                name="decoder",
                inputs=decoder_input,
                outputs=decoder_output)

        # --- build end-to-end trainable model
        logger.info("Building end-to-end trainable model")
        mu_s = []
        encode_s = []
        log_var_s = []
        vae_encode_decode = []
        vae_input = \
            keras.Input(
                shape=self._inputs_dims,
                name="input")
        vae_input_multiscale = \
            model_laplacian_split(vae_input)

        for i in range(self._levels):
            encode, mu, log_var = \
                encoders[i](vae_input_multiscale[i])
            mu_s.append(mu)
            log_var_s.append(log_var)
            encode_s.append(encode)
            decode = decoders[i](encode)
            vae_encode_decode.append(decode)

        # merge multiple outputs in a single output
        vae_output = \
            model_decoder_merge(vae_encode_decode)

        self._model_trainable = \
            keras.Model(
                inputs=vae_input,
                outputs=vae_output,
                name="trainable_vae")

        # --- concat all z-latent layers
        self._mu = \
            keras.layers.Concatenate(axis=-1, name="mu")(mu_s)
        self._log_var = \
            keras.layers.Concatenate(axis=-1, name="log_var")(log_var_s)

        # --- concat z-dim
        self._encode = \
            keras.layers.Concatenate(axis=-1, name="z_dim")(encode_s)

        # --- save intermediate levels
        self._output_multiscale = vae_encode_decode
        self._input_multiscale = vae_input_multiscale

    # ===============================================================================

    def _build_encoder(
            self,
            encoder_input,
            z_dim: int,
            sample_stddev: float = 1.0,
            prefix: str = "encoder_"):
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
            input_layer=x,
            prefix=prefix,
            use_dropout=False,
            use_batchnorm=True,
            block_type="encoder",
            regularizer=self._kernel_regularizer,
            initializer=self._initialization_scheme,
            strides=self._encoder_config["strides"],
            filters=self._encoder_config["filters"],
            kernel_size=self._encoder_config["kernel_size"])

        # --- keep shape before flattening
        shape = K.int_shape(x)
        shape_before_flattening = shape[1:]

        # --- flatten and convert to z_dim dimensions
        x = keras.layers.Flatten()(x)

        mu = keras.layers.Dense(
            units=z_dim,
            use_bias=False,
            activation="linear",
            name=f"{prefix}mu",
            kernel_regularizer=self._dense_regularizer,
            kernel_initializer=self._initialization_scheme)(x)
        log_var = keras.layers.Dense(
            units=z_dim,
            use_bias=False,
            activation="linear",
            name=f"{prefix}log_var",
            kernel_regularizer=self._dense_regularizer,
            kernel_initializer=self._initialization_scheme)(x)

        def sample(args):
            tmp_mu, tmp_log_var, tmp_stddev = args
            epsilon = K.random_normal(
                mean=0.0,
                stddev=tmp_stddev,
                shape=K.shape(tmp_mu))
            return tmp_mu + K.exp(0.5 * tmp_log_var) * epsilon

        return \
            keras.layers.Lambda(
                sample, name=prefix + "_sample_output")(
                [mu, log_var, sample_stddev]), \
            [mu, log_var], \
            shape_before_flattening

    # ==========================================================================

    def _build_decoder(
            self,
            input_layer,
            target_shape,
            prefix: str = "decoder_"):
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

        # --- transforming here
        x = layer_blocks.basic_block(
            input_layer=x,
            prefix=prefix,
            use_batchnorm=True,
            block_type="decoder",
            filters=self._decoder_config["filters"],
            strides=self._decoder_config["strides"],
            kernel_size=self._decoder_config["kernel_size"])

        # --- match target channels to [-1, +1]
        x = keras.layers.Conv2D(
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_size=(1, 1),
            activation="tanh",
            filters=self._output_channels,
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._initialization_scheme)(x)

        return x

    # ==========================================================================

    def compile(
            self,
            learning_rate,
            r_loss_factor: float = 1.0,
            kl_loss_factor: float = 1.0,
            z_dim_encoding_factor: float = 0.001,
            clip_norm: float = 1.0):
        """

        :param learning_rate:
        :param r_loss_factor:
        :param kl_loss_factor:
        :param z_dim_encoding_factor:
        :param clip_norm:
        :return:
        """
        self.learning_rate = learning_rate

        # --- define encoding distance
        def z_dim_pairwise_distance_loss(y_true, y_pred):
            batch = K.shape(self._encode)[0]
            x = tf.expand_dims(self._encode, 1)
            x_t = tf.expand_dims(self._encode, 0)
            x_tile = tf.tile(x, [1, batch, 1])
            x_t_tile = tf.tile(x_t, [batch, 1, 1])
            delta_x = x_tile - x_t_tile
            delta_x_2 = delta_x * delta_x
            d2x = tf.reduce_sum(delta_x_2, 2)
            return d2x

        # pulls encodings towards the origin 0
        def z_dim_encoding_loss(y_true, y_pred):
            return K.mean(K.square(K.sum(K.pow(self._encode, 2.0), axis=1)), axis=0)

        # --- define VAE reconstruction loss
        def r_loss(y_true, y_pred):
            tmp_diff = K.abs(y_true - y_pred)
            tmp_pixels = K.mean(tmp_diff, axis=[1, 2, 3])
            return K.mean(tmp_pixels, axis=-1)

        # --- define VAE reconstruction loss per scale
        def multi_r_loss(y_true, y_pred):
            result = 0.0
            for i in range(self._levels):
                tmp_pixels = \
                    K.abs(self._input_multiscale[i] - self._output_multiscale[i])
                tmp_pixels = K.mean(tmp_pixels, axis=[1, 2, 3])
                result += K.mean(tmp_pixels, axis=-1)
            return result / float(self._levels)

        # --- define KL divergence loss for the latent space
        # (difference from isotropic normal distribution m=0, var=1)
        def kl_loss(y_true, y_pred):
            x = K.sum(
                -0.5 * (1 + self._log_var - K.square(self._mu) - K.exp(self._log_var)),
                axis=1)
            return K.mean(x)

        # --- define combined loss
        def loss(y_true, y_pred):
            tmp_r_loss = r_loss(y_true, y_pred)
            tmp_kl_loss = kl_loss(y_true, y_pred)
            tmp_multi_r_loss = multi_r_loss(y_true, y_pred)
            tmp_z_encoding = z_dim_encoding_loss(y_true, y_pred)
            tmp_z_encoding_pairwise = z_dim_pairwise_distance_loss(y_true, y_pred)
            return \
                tmp_kl_loss * kl_loss_factor + \
                tmp_r_loss * r_loss_factor + \
                K.abs(tmp_z_encoding - tmp_z_encoding_pairwise) * z_dim_encoding_factor + \
                tmp_multi_r_loss * r_loss_factor * (self._max_value - self._min_value)

        optimizer = \
            keras.optimizers.Adagrad(
                lr=self.learning_rate,
                clipnorm=clip_norm)

        self._model_trainable.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                r_loss,
                kl_loss,
                multi_r_loss,
                z_dim_encoding_loss,
                z_dim_pairwise_distance_loss
            ])

    # ==========================================================================

    def train(
            self,
            x_train,
            epochs: int,
            batch_size: int,
            run_folder: str,
            step_size: int = 1,
            lr_decay: float = 1.0,
            initial_epoch: int = 0,
            print_every_n_batches: int = 100,
            save_checkpoint_weights: bool = False):

        custom_callback = callbacks.SaveIntermediateResultsCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            x_train[0:64, :, :, :],
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
        tb_callback = \
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(run_folder, "logs"),
                update_freq=100,
                histogram_freq=100)

        callbacks_fns = [
            lr_schedule,
            tb_callback,
            custom_callback
        ]

        if save_checkpoint_weights:
            callbacks_fns += [checkpoint1, checkpoint2]

        batches_per_epoch = 4 * int(math.ceil(len(x_train) / batch_size))

        self._model_trainable.fit(
            utilities.noisy_image_data_generator(
                dataset=x_train,
                batch_size=batch_size,
                min_value=0.0,
                max_value=255.0,
                min_noise_std=0.0,
                max_noise_std=1.0,
                random_invert=False,
                vertical_flip=True,
                horizontal_flip=True),
            epochs=epochs,
            callbacks=callbacks_fns,
            initial_epoch=initial_epoch,
            steps_per_epoch=batches_per_epoch)

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
    def z_dim(self) -> int:
        return sum(self._z_latent_dims)

    @property
    def model_encode(self):
        return self._model_encoder

    @property
    def model_decode(self):
        return self._model_decoder

    @property
    def model_sample(self):
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

    # ==========================================================================

import os
import json
import keras
import pickle
import logging
import numpy as np

import utils
import utils.transform
from utils.callbacks import CustomCallback, step_decay_schedule

# --------------------------------------------------------------------------------
# setup logger
# --------------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s")
logging.getLogger("multiscale-vae").setLevel(logging.INFO)
logger = logging.getLogger("multiscale-vae")

# --------------------------------------------------------------------------------


class MultiscaleVariationalAutoencoder():
    def __init__(
            self,
            input_dims,
            levels,
            z_dims):
        """

        :param input_dims:
        :param levels:
        :param z_dims:
        """
        self.name = "multiscale_variational_autoencoder"
        self.inputs_dims = input_dims

        if levels <= 0:
            raise ValueError("levels should be integer and > 0")

        if len(z_dims) != levels:
            raise ValueError("z_dims should be a list of length levels")

        if not all(i > 0 for i in z_dims):
            raise ValueError("z_dims elements should be > 0")

        self.levels = levels
        self.z_dims = z_dims

        self._build()

    # --------------------------------------------------------------------------------

    def _build(self):
        """

        :return:
        """
        # --------- Build multiscale input
        self.scales = []

        for i in range(self.levels):
            if i == 0:
                self.scales.append(
                    keras.Input(
                        shape=self.inputs_dims,
                        name="scale_" + str(i)))
            else:
                self.scales.append(
                    keras.layers.MaxPool2D(
                        pool_size=(2, 2),
                        strides=None,
                        padding="valid",
                        name="scale_" + str(i))(self.scales[i-1]))

        # --------- Reverse scales
        self.scales = self.scales[::-1]

        # --------- Create Encoder / Decoder
        self.encoders = []
        self.decoders = []
        self.z_domains = []

        for i in range(self.levels):
            if i == 0:
                # --------- Base Encoder Decoder is special
                encoder, z_domain, shape_before_flattening = \
                    self._encoder(
                        self.scales[i],
                        z_dim=self.z_dims[i],
                        prefix="encoder_" + str(i) + "_")

                decoder = self._decoder(
                    encoder,
                    shape=shape_before_flattening,
                    prefix="decoder_" + str(i) + "_")
            else:
                # --------- Upper scale Encoder Decoders are the same
                previous_scale_upscaled = \
                    self._upscale(
                        self.decoders[i-1],
                        shape=keras.backend.int_shape(self.scales[i]))

                diff = keras.layers.Subtract()([
                    self.scales[i],
                    previous_scale_upscaled
                ])

                encoder, z_domain, shape_before_flattening = \
                    self._encoder(
                        diff,
                        z_dim=self.z_dims[i],
                        prefix="encoder_" + str(i) + "_")

                decoder = keras.layers.Add()([
                    self._decoder(
                        encoder,
                        shape=shape_before_flattening,
                        prefix="decoder_" + str(i) + "_"),
                    previous_scale_upscaled])

            self.z_domains.append(z_domain)
            self.encoders.append(encoder)
            self.decoders.append(decoder)

        self.model = keras.Model(
            self.scales[-1],
            self.decoders)

        return self.model

    # --------------------------------------------------------------------------------

    def load_weights(self, filename):
        return

    # --------------------------------------------------------------------------------

    @staticmethod
    def _upscale(input_layer, shape):
        """
        Upscales the input_layer to match the given shape
        :param input_layer:
        :param shape:
        :return:
        """
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear")(input_layer)
        #current_shape = keras.backend.int_shape(x)
        #while current_shape != shape:
        #    logger.info(">>>> {0}-{1}".format(current_shape, shape))
        #    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        #    current_shape = keras.backend.int_shape(x)
        return x

    # --------------------------------------------------------------------------------

    def _encoder(self,
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
        x = self._basic_block(x,
                              block_type="encoder",
                              filters=[16, 16, 16],
                              kernel_size=[(3, 3), (3, 3), (3, 3)],
                              strides=[(2, 2), (2, 2), (1, 1)],
                              prefix=prefix)
        # --------- Keep shape before flattening
        shape_before_flattening = keras.backend.int_shape(x)[1:]

        # --------- Flatten and convert to z_dim dimensions
        x = keras.layers.Flatten()(x)
        mu = keras.layers.Dense(z_dim, name=prefix + "mu")(x)
        log_var = keras.layers.Dense(z_dim, name=prefix + "log_var")(x)

        def sampling(args):
            tmp_mu, tmp_log_var = args
            epsilon = keras.backend.random_normal(
                shape=keras.backend.shape(tmp_mu),
                mean=0.,
                stddev=1.)
            return tmp_mu + keras.backend.exp(tmp_log_var / 2.0) * epsilon

        return keras.layers.Lambda(sampling, name=prefix + "output")([mu, log_var]), \
               [mu, log_var], \
               shape_before_flattening

    # --------------------------------------------------------------------------------

    def _decoder(self,
                 decoder_input,
                 shape,
                 prefix="decoder_"):
        x = decoder_input
        # --------- Decoding here
        x = keras.layers.Dense(np.prod(shape))(x)
        x = keras.layers.Reshape(shape)(x)
        # --------- Transforming here
        x = self._basic_block(x,
                              block_type="decoder",
                              filters=[16, 16, 3],
                              kernel_size=[(3, 3), (3, 3), (3, 3)],
                              strides=[(2, 2), (2, 2), (1, 1)],
                              prefix=prefix)
        return x

    # --------------------------------------------------------------------------------

    @staticmethod
    def _basic_block(input_layer,
                     block_type="encoder",
                     filters=[64],
                     kernel_size=[(3, 3)],
                     strides=[(1, 1)],
                     use_batchnorm=True,
                     use_dropout=True,
                     prefix="block_", ):

        if len(filters) != len(kernel_size) or \
                len(filters) != len(strides) or \
                len(filters) <= 0:
            raise ValueError("len(filters) should be equal to len(kernel_size) and len(strides)")

        if block_type != "encoder" and block_type != "decoder":
            raise ValueError("block_type should be encoder or decoder")

        x = input_layer

        for i in range(len(filters)):
            if block_type == "encoder":
                x = keras.layers.Conv2D(
                    filters=filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding="same",
                    name=prefix + "conv_0_" + str(i))(x)
            if block_type == "decoder":
                x = keras.layers.Conv2DTranspose(
                    filters=filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding="same",
                    name=prefix + "conv_0_" + str(i))(x)

            if use_batchnorm:
                x = keras.layers.BatchNormalization()(x)

            x = keras.layers.LeakyReLU()(x)

            if use_dropout:
                x = keras.layers.Dropout(rate=0.25)(x)

            if block_type == "encoder" or \
                    i < len(filters) - 1:
                if use_batchnorm:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.LeakyReLU()(x)
                if use_dropout:
                    x = keras.layers.Dropout(rate=0.25)(x)
            else:
                x = keras.layers.Activation("sigmoid")(x)

        return x

    # --------------------------------------------------------------------------------

    def compile(
            self,
            learning_rate,
            r_loss_factor=1.0,
            kl_loss_factor=1.0):
        self.learning_rate = learning_rate

        # --------- Define VAE recreation loss
        def vae_r_loss(y_true, y_pred):
            r_loss = keras.backend.mean(
                keras.backend.square(y_true - y_pred),
                axis=[1, 2, 3])
            return r_loss_factor * r_loss

        # --------- Define KL parameters loss
        def vae_kl_loss(y_true, y_pred):
            for i in range(len(self.z_domains)):
                z = self.z_domains[i]
                if i == 0:
                    kl_loss = -0.5 * keras.backend.sum(
                        1.0 +
                        z[1] - keras.backend.square(z[0]) - keras.backend.exp(z[1]),
                        axis=1)
                else:
                    kl_loss += -0.5 * keras.backend.sum(
                        1.0 +
                        z[1] - keras.backend.square(z[0]) - keras.backend.exp(z[1]),
                        axis=1)
            return kl_loss_factor * kl_loss / len(self.z_domains)

        # --------- Define combined loss
        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = keras.optimizers.Adam(lr=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=vae_loss,
            metrics=[vae_r_loss, vae_kl_loss])

    # --------------------------------------------------------------------------------

    def train(
            self,
            x_train,
            batch_size,
            epochs,
            run_folder,
            print_every_n_batches=100,
            initial_epoch=0,
            lr_decay=1):

        target = []

        for i in range(self.levels):
            if i == 0:
                target.append(x_train)
            else:
                target.append(
                    utils.transform.pool4d(
                        target[i-1],
                        kernel_size=2,
                        stride=2,
                        padding=0))

        target = target[::-1]

        #custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)
        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = keras.callbacks.ModelCheckpoint(os.path.join(run_folder, "weights/weights.h5"), save_weights_only=True, verbose=1)

        self.model.fit(
            x_train,
            target,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=[
                checkpoint1,
                checkpoint2,
                lr_sched
            ]
        )

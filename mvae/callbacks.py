import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.callbacks import Callback

# ==============================================================================


def collage(images_batch):
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))
    height = no_images / width

    for i in range(no_images):
        images.append(images_batch[i, :, :, :])

        if len(images) % width == 0:
            if result is None:
                result = np.hstack(images)
            else:
                tmp = np.hstack(images)
                result = np.vstack([result, tmp])
            images.clear()
    return result


class SaveIntermediateResultsCallback(Callback):
    
    def __init__(self,
                 run_folder,
                 print_every_n_batches,
                 initial_epoch,
                 images,
                 vae,
                 resize_shape=(256, 256)):
        """
        Callback for saving the intermediate result image
        :param run_folder:
        :param print_every_n_batches:
        :param initial_epoch:
        :param images:
        :param vae:
        :param resize_shape:
        """
        self._vae = vae
        self._images = images
        self._epoch = initial_epoch
        self._run_folder = run_folder
        self._resize_shape = resize_shape
        self._print_every_n_batches = print_every_n_batches
        self._noise = \
            np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(images.shape[0], self._vae.z_dim))
        self._images_path = os.path.join(self._run_folder, "images")
        if not os.path.exists(self._images_path):
            os.mkdir(self._images_path)

    def save_collage(
            self,
            samples: np.ndarray,
            batch: int,
            prefix: str):
        # normalize to [0, 1]
        x = self._vae.normalize(samples)
        # create collage
        x = collage(x)
        # resize to output size
        x = resize(x, self._resize_shape, order=0)
        filepath_x = os.path.join(
            self._images_path,
            f"{prefix}_" + str(self._epoch).zfill(3) +
            "_" + str(batch) + ".png")
        if len(x.shape) == 2:
            plt.imsave(filepath_x, x, cmap="gray_r")
        else:
            plt.imsave(filepath_x, x)

    def on_batch_end(self, batch, logs={}):
        # --- do this only so many batches
        if batch % self._print_every_n_batches != 0:
            return

        # --- setup parameters
        no_samples = self._images.shape[0]

        # --- encode decode
        encodings = self._vae.model_encode.predict(self._images)
        decodings = self._vae.model_decode.predict(encodings)
        # create and save collage of the reconstructions
        self.save_collage(
            samples=decodings,
            batch=batch,
            prefix="img")

        # --- random z-dim decoding
        samples = self._vae.model_decode.predict(self._noise)
        # create and save collage of the samples
        self.save_collage(
            samples=samples,
            batch=batch,
            prefix="samples")

        # --- interpolation z-dim decoding
        start_sample = encodings[0, :]
        end_sample = encodings[1, :]
        interpolations = np.zeros_like(encodings)
        for i in range(no_samples):
            mix_coeff = float(i) / float(no_samples)
            interpolations[i, :] = \
                start_sample * (1.0 - mix_coeff) + \
                end_sample * mix_coeff
        interpolations[0, :] = start_sample
        interpolations[-1, :] = end_sample
        interpolations = self._vae.model_decode.predict(interpolations)
        # create and save collage of the samples
        self.save_collage(
            samples=interpolations,
            batch=batch,
            prefix="interpolations")

    def on_epoch_begin(self, epoch, logs={}):
        self._epoch += 1


# ==============================================================================

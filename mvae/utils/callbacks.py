import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.callbacks import Callback

# ==============================================================================


class SaveIntermediateResultsCallback(Callback):
    
    def __init__(self,
                 run_folder,
                 print_every_n_batches,
                 initial_epoch,
                 image,
                 vae,
                 resize_shape=(128, 128)):
        """
        Callback for saving the intermediate result image
        :param run_folder:
        :param print_every_n_batches:
        :param initial_epoch:
        :param image:
        :param vae:
        :param resize_shape:
        """
        self.vae = vae
        self.image = image
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self._resize_shape = resize_shape
        self.print_every_n_batches = print_every_n_batches
        images_path = os.path.join(self.run_folder, "images")
        if not os.path.exists(images_path):
            os.mkdir(images_path)

    def on_batch_end(self, batch, logs={}):  
        if batch % self.print_every_n_batches == 0:
            reconstruction = self.vae._model_trainable.predict(self.image)
            x = reconstruction.squeeze()
            x = resize(x, self._resize_shape, order=0)
            filepath_x = os.path.join(
                self.run_folder,
                "images",
                "img_" + str(self.epoch).zfill(3) +
                "_" + str(batch) + ".jpg")
            if len(x.shape) == 2:
                plt.imsave(filepath_x, x, cmap="gray_r")
            else:
                plt.imsave(filepath_x, x)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1


# ==============================================================================

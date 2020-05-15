import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.callbacks import Callback, LearningRateScheduler

# --------------------------------------------------------------------------------


class CustomCallback(Callback):

    def __init__(self, run_folder, print_every_n_batches, initial_epoch, image, vae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.image = image

    def on_batch_end(self, batch, logs={}):  
        if batch % self.print_every_n_batches == 0:
            reconstruction = self.vae.model_trainable.predict(self.image)
            for i in range(self.vae.levels):
                x = reconstruction[i].squeeze()
                x = resize(x, (128, 128), order=0)
                filepath = os.path.join(
                    self.run_folder,
                    "images",
                    "img_" + str(self.epoch).zfill(3) + "_" + str(batch) + "_" + str(i)  + ".jpg")
                if len(x.shape) == 2:
                    plt.imsave(filepath, x, cmap="gray_r")
                else:
                    plt.imsave(filepath, x)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1


# --------------------------------------------------------------------------------


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return new_lr

    return LearningRateScheduler(schedule)

# --------------------------------------------------------------------------------

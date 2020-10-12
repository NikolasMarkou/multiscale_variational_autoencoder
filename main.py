import os
import sys
import mvae
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# %%

# run params
SECTION = "vae"
RUN_ID = "0001"
DATA_NAME = "cifar10"
BASE_DIR = "./run"
BASE_DIR_SECTION = "{0}/{1}/".format(BASE_DIR, SECTION)
RUN_FOLDER = BASE_DIR_SECTION + "_".join([RUN_ID, DATA_NAME])

if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

if not os.path.exists(BASE_DIR_SECTION):
    os.mkdir(BASE_DIR_SECTION)

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, "viz"))
    os.mkdir(os.path.join(RUN_FOLDER, "images"))
    os.mkdir(os.path.join(RUN_FOLDER, "weights"))

mode = "build"

# -----------------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -----------------------------------------

multiscale_vae = mvae.MultiscaleVAE(
    input_dims=(32, 32, 3),
    levels=3,
    z_dims=[128, 128, 128],
    encoder={
        "filters": [32, 32, 32],
        "kernel_size": [(3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (1, 1), (1, 1)]
    },
    decoder={
        "filters": [32, 32, 32],
        "kernel_size": [(3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (1, 1), (1, 1)]
    })

# -----------------------------------------

EPOCHS = 100
STEP_SIZE = 10
LR_DECAY = 0.5
BATCH_SIZE = 32
INITIAL_EPOCH = 0
KL_LOSS_FACTOR = 1
LEARNING_RATE = 0.01
R_LOSS_FACTOR = 100
PRINT_EVERY_N_BATCHES = 1000

# -----------------------------------------

multiscale_vae.compile(
    learning_rate=LEARNING_RATE,
    r_loss_factor=R_LOSS_FACTOR,
    kl_loss_factor=KL_LOSS_FACTOR
)

# serialize model to JSON
with open("model_trainable.json", "w") as json_file:
    json_file.write(multiscale_vae._model_trainable.to_json())

# -----------------------------------------

multiscale_vae.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    initial_epoch=INITIAL_EPOCH,
    step_size=STEP_SIZE,
    lr_decay=LR_DECAY)




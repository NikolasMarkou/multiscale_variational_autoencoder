import os
import mvae
import numpy as np
import tensorflow as tf
from scipy import ndimage
from keras.datasets import cifar10
from mvae.custom_logger import logger

# ==============================================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ==============================================================================

EPOCHS = 150
STEP_SIZE = 30
LR_DECAY = 0.5
BATCH_SIZE = 32
INITIAL_EPOCH = 0
KL_LOSS_FACTOR = 0.01
R_LOSS_FACTOR = 1
LEARNING_RATE = 0.01
EXPAND_DATASET = False
PRINT_EVERY_N_BATCHES = 1000

# run params
SECTION = "vae"
RUN_ID = "0001"
BASE_DIR = "./run"
DATA_NAME = "cifar10"
BASE_DIR_SECTION = "{0}/{1}/".format(BASE_DIR, SECTION)
RUN_FOLDER = BASE_DIR_SECTION + "_".join([RUN_ID, DATA_NAME])

logger.info("Creating training directories")

if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

if not os.path.exists(BASE_DIR_SECTION):
    os.mkdir(BASE_DIR_SECTION)

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, "viz"))
    os.mkdir(os.path.join(RUN_FOLDER, "images"))
    os.mkdir(os.path.join(RUN_FOLDER, "weights"))

# delete existing images
images_directory = os.path.join(RUN_FOLDER, "images")
for filename in os.listdir(images_directory):
    full_image_path = os.path.join(images_directory, filename)
    os.remove(full_image_path)

# ==============================================================================

logger.info("Loading and expanding dataset")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

if EXPAND_DATASET:
    x_train_extra = np.ndarray(x_train.shape,
                               dtype=np.float32)
    no_training_samples = x_train.shape[0]

    for i in range(no_training_samples):
        if i % 1000 == 0:
            logger.info("Expanding [{0}/{1}]".format(i, no_training_samples))
        sample = x_train[i, :, :, :]
        x_train_extra[i, :, :, :] = ndimage.median_filter(sample, size=3)
    x_train = np.concatenate([x_train, x_train_extra, x_test], axis=0)

# ==============================================================================

logger.info("Creating model")

multiscale_vae = mvae.MultiscaleVAE(
    input_dims=(32, 32, 3),
    z_dims=[32, 32],
    encoder={
        "filters": [32, 64, 64],
        "kernel_size": [(3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (2, 2), (2, 2)]
    },
    min_value=0.0,
    max_value=255.0)

# ==============================================================================

multiscale_vae.compile(
    learning_rate=LEARNING_RATE,
    r_loss_factor=R_LOSS_FACTOR,
    kl_loss_factor=KL_LOSS_FACTOR
)

# serialize model to JSON
with open("model_trainable.json", "w") as json_file:
    json_file.write(multiscale_vae.model_trainable.to_json())

# serialize model to JSON
with open("model_encoder.json", "w") as json_file:
    json_file.write(multiscale_vae.encoder.to_json())

# serialize model to JSON
with open("model_decoder.json", "w") as json_file:
    json_file.write(multiscale_vae.decoder.to_json())

# ==============================================================================

logger.info("Training model")

multiscale_vae.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    initial_epoch=INITIAL_EPOCH,
    step_size=STEP_SIZE,
    lr_decay=LR_DECAY)

# ==============================================================================

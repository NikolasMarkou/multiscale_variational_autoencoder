import os
import pathlib
import tensorflow as tf
from typing import Union, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


from .custom_logger import logger

# ---------------------------------------------------------------------


def create_checkpoint(
        step: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.dtypes.int64, name="step"),
        epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.dtypes.int64, name="epoch"),
        model: tf.keras.Model = None,
        path: Union[str, pathlib.Path] = None) -> tf.train.Checkpoint:
    # define common checkpoint
    ckpt = \
        tf.train.Checkpoint(
            step=step,
            epoch=epoch,
            model=model)
    # if paths exists load latest
    if path is not None:
        if os.path.isdir(str(path)):
            ckpt.restore(tf.train.latest_checkpoint(str(path))).expect_partial()
    return ckpt

# ---------------------------------------------------------------------


def model_weights_from_checkpoint(
        path: Union[str, pathlib.Path]):
    reader = tf.train.load_checkpoint(str(path))
    shape_from_key = reader.get_variable_to_shape_map()
    return [
        reader.get_tensor(k)
        for k in shape_from_key.keys()
        if k.startswith("model/")
    ]

# ---------------------------------------------------------------------

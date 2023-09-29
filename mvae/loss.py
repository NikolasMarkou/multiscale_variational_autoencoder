r"""Constructs the loss function for the denoiser task and variational autoencoder task"""

import tensorflow as tf
import tensorflow_addons as tfa
from typing import List, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger

# ---------------------------------------------------------------------


def mae_diff(
        error: tf.Tensor,
        hinge: float = 0.0,
        cutoff: float = 255.0) -> tf.Tensor:
    """
    Mean Absolute Error (mean over channels and batches)

    :param error: diff between prediction and ground truth
    :param hinge: hinge value
    :param cutoff: max value

    :return: mean absolute error
    """
    d = \
        tf.keras.activations.relu(
            x=tf.abs(error),
            threshold=hinge,
            max_value=cutoff)

    # --- mean over batch
    return tf.reduce_mean(d)


# ---------------------------------------------------------------------


def mae(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs) -> tf.Tensor:
    """
    Mean Absolute Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    """
    return \
        mae_diff(
            error=(original - prediction),
            **kwargs)


# ---------------------------------------------------------------------


def rmse_diff(
        error: tf.Tensor,
        hinge: float = 0,
        cutoff: float = (255.0 * 255.0)) -> tf.Tensor:
    """
    Root Mean Square Error (mean over channels and batches)

    :param error:
    :param hinge: hinge value
    :param cutoff: max value
    """
    d = \
        tf.keras.activations.relu(
            x=error,
            threshold=hinge,
            max_value=cutoff)
    d = tf.square(d)
    # mean over all dims
    d = tf.reduce_mean(d, axis=[1, 2, 3])
    d = tf.sqrt(tf.nn.relu(d))
    # mean over batch
    return tf.reduce_mean(d, axis=[0])


# ---------------------------------------------------------------------


def rmse(
        original: tf.Tensor,
        prediction: tf.Tensor,
        **kwargs) -> tf.Tensor:
    """
    Root Mean Square Error (mean over channels and batches)

    :param original: original image batch
    :param prediction: denoised image batch
    """
    return rmse_diff(
        error=(original - prediction),
        **kwargs)


# ---------------------------------------------------------------------


def loss_function_builder(
        config: Dict) -> Callable:
    """
    Constructs the loss function of the denoiser and variational autoencoder

    :param config: configuration dictionary
    :return: callable loss function
    """
    logger.info("building loss_function with config [{0}]".format(config))

    # --- denoiser params
    denoiser_params = config.get(DENOISER_STR, None)
    hinge = denoiser_params.get("hinge", 0.0)
    cutoff = denoiser_params.get("cutoff", 255.0)
    mae_multiplier = denoiser_params.get("mae_multiplier", 1.0)
    use_mae = mae_multiplier > 0.0
    ssim_multiplier = denoiser_params.get("ssim_multiplier", 1.0)
    use_ssim = ssim_multiplier > 0.0
    mse_multiplier = denoiser_params.get("mse_multiplier", 0.0)
    use_mse = mse_multiplier > 0.0

    # --- model params
    model_params = config.get(MODEL_STR, None)
    regularization_multiplier = model_params.get("regularization", 1.0)

    def model_loss(model):
        regularization_loss = \
            tf.add_n(model.losses)
        return {
            REGULARIZATION_LOSS_STR: regularization_loss,
            TOTAL_LOSS_STR: regularization_loss * regularization_multiplier
        }

    # ---
    def denoiser_loss(
            input_batch: tf.Tensor,
            predicted_batch: tf.Tensor) -> Dict[str, tf.Tensor]:

        # --- resize input to match prediction in case they are different sizes
        if tf.reduce_any(
                tf.shape(input_batch) != tf.shape(predicted_batch)):
            # this must be bilinear or nearest for speed during training,
            input_batch = \
                tf.image.resize(
                    images=input_batch,
                    size=tf.shape(predicted_batch)[1:3],
                    method=tf.image.ResizeMethod.BILINEAR)

        # --- loss prediction on mae
        mae_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)

        if use_mae:
            mae_prediction_loss += \
                mae(original=input_batch,
                    prediction=predicted_batch,
                    hinge=hinge,
                    cutoff=cutoff)

        # --- loss ssim
        ssim_loss = tf.constant(0.0, dtype=tf.float32)
        if use_ssim:
            ssim_loss = \
                tf.reduce_mean(
                    tf.image.ssim(input_batch, predicted_batch, 255.0))
            ssim_loss = 1.0 - ssim_loss

        # --- loss prediction on mse
        mse_prediction_loss = \
            tf.constant(0.0, dtype=tf.float32)
        if use_mse:
            mse_prediction_loss += \
                rmse(original=input_batch,
                     prediction=predicted_batch,
                     hinge=hinge,
                     cutoff=(cutoff * cutoff))

        return {
            TOTAL_LOSS_STR:
                mae_prediction_loss * mae_multiplier +
                mse_prediction_loss * mse_multiplier +
                ssim_loss * ssim_multiplier,
            SSIM_LOSS_STR: ssim_loss,
            MAE_LOSS_STR: mae_prediction_loss,
            MSE_LOSS_STR: mse_prediction_loss,
        }

    # ----
    return {
        MODEL_LOSS_FN_STR: model_loss,
        DENOISER_LOSS_FN_STR: denoiser_loss,
    }

# ---------------------------------------------------------------------

import os
import json
import copy
import time
import math
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .loss import \
    loss_function_builder
from .optimizer import optimizer_builder
from .models import model_builder, model_output_indices
from .utilities import \
    save_config, \
    load_config
from .dataset import *
from .metrics import metrics
from .visualize import \
    visualize_weights_boxplot, \
    visualize_confusion_matrix, \
    visualize_weights_heatmap, \
    visualize_gradient_boxplot
from .utilities_checkpoints import \
    create_checkpoint, \
    model_weights_from_checkpoint

# ---------------------------------------------------------------------

CURRENT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))

tf.get_logger().setLevel("WARNING")
tf.autograph.set_verbosity(2)


# ---------------------------------------------------------------------


def train_loop(
        pipeline_config_path: Union[str, Dict, Path],
        model_dir: Union[str, Path],
        weights_dir: Union[str, Path] = None):
    """
    Trains a multiscale variational autoencoder

    This method:
        1. Processes the pipeline configs
        2. (Optionally) saves the as-run config
        3. Builds the model & optimizer
        4. Gets the training input data
        5. Loads a fine-tuning detection
        6. Loops over the train data
        7. Checkpoints the model every `checkpoint_every_n` training steps.
        8. Logs the training metrics as TensorBoard summaries.

    :param pipeline_config_path: filepath to the configuration
    :param model_dir: directory to save checkpoints into
    :param weights_dir: directory to load weights from
    :return:
    """
    # --- load configuration
    config = load_config(pipeline_config_path)

    # --- create model_dir if not exist
    if not os.path.isdir(str(model_dir)):
        # if path does not exist attempt to make it
        Path(str(model_dir)).mkdir(parents=True, exist_ok=True)
        # if it fails again throw exception
        if not os.path.isdir(str(model_dir)):
            raise ValueError("Model directory [{0}] is not valid".format(
                model_dir))

    # --- save configuration into path, makes it easier to compare afterwards
    save_config(
        config=config,
        filename=os.path.join(str(model_dir), CONFIG_PATH_STR))

    # --- build dataset
    dataset = dataset_builder(config[DATASET_STR])

    dataset_eval = dataset.evaluate
    dataset_training_materials = dataset.training_materials
    dataset_testing_materials = dataset.testing_materials
    dataset_training_bg_lumen_wall = dataset.training_bg_lumen_wall
    dataset_testing_bg_lumen_wall = dataset.testing_bg_lumen_wall
    batch_size = dataset.batch_size
    input_shape = dataset.input_shape

    # --- build loss function
    loss_fn_map = loss_function_builder(config=config["loss"])
    model_loss_fn = \
        tf.function(
            func=loss_fn_map[MODEL_LOSS_FN_STR],
            reduce_retracing=True)
    denoiser_loss_fn = \
        tf.function(
            func=loss_fn_map[DENOISER_LOSS_FN_STR],
            input_signature=[
                tf.TensorSpec(shape=[batch_size, None, None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[batch_size, None, None, 1], dtype=tf.float32),
            ],
            reduce_retracing=True)

    # --- build optimizer
    optimizer, lr_schedule = \
        optimizer_builder(config=config["train"]["optimizer"])

    # --- get the train configuration
    train_config = config["train"]
    # controls how different outputs of the model get discounted in the loss
    # 1.0 all equal
    # discount lower depth losses as epochs carry forward
    # assuming output_discount_factor = 0.25
    # for percentage_done in [0.0, 0.25, 0.5, 0.75, 1.0]:
    #     x = [0.25 ** (float(i) * percentage_done) for i in range(5)]
    #     print(x)
    # [1.0, 1.0, 1.0, 1.0, 1.0]
    # [1.0, 0.707, 0.5, 0.353, 0.25]
    # [1.0, 0.5, 0.25, 0.125, 0.0625]
    # [1.0, 0.353, 0.125, 0.0441, 0.015625]
    # [1.0, 0.25, 0.0625, 0.015625, 0.00390625]
    output_discount_factor = train_config.get("output_discount_factor", 1.0)
    if output_discount_factor > 1.0 or output_discount_factor < 0.0:
        raise ValueError(f"output_discount_factor [{output_discount_factor}] "
                         f"must be between 0.0 and 1.0")
    #
    ssl_epochs = train_config.get("ssl_epochs", -1)
    epochs = train_config["epochs"]
    gpu_batches_per_step = int(train_config.get("gpu_batches_per_step", 1))
    use_rotational_invariance = train_config.get("use_rotational_invariance", False)

    if gpu_batches_per_step <= 0:
        raise ValueError("gpu_batches_per_step must be > 0")

    # how many checkpoints to keep
    checkpoints_to_keep = \
        train_config.get("checkpoints_to_keep", 3)
    # checkpoint every so many steps
    checkpoint_every = \
        tf.constant(
            train_config.get("checkpoint_every", -1),
            dtype=tf.dtypes.int64,
            name="checkpoint_every")
    # how many steps to make a visualization
    visualization_every = \
        tf.constant(
            train_config.get("visualization_every", 1000),
            dtype=tf.dtypes.int64,
            name="visualization_every")
    # how many visualizations to show
    visualization_number = train_config.get("visualization_number", 5)

    # --- train the model
    with tf.summary.create_file_writer(model_dir).as_default():
        # --- write configuration in tensorboard
        tf.summary.text("config", json.dumps(config, indent=4), step=0)

        # --- create the help variables
        total_epochs = tf.constant(
            epochs, dtype=tf.dtypes.int64, name="total_epochs")
        total_steps = tf.constant(
            train_config.get("total_steps", -1), dtype=tf.dtypes.int64, name="total_steps")

        # --- build the hydra model
        config[MODEL_STR][BATCH_SIZE_STR] = batch_size
        models = model_builder(config=config[MODEL_STR])
        ckpt = \
            create_checkpoint(
                model=models.hydra,
                path=None)
        # summary of model and save model, so we can inspect with netron
        ckpt.model.summary(print_fn=logger.info)
        ckpt.model.save(
            os.path.join(model_dir, MODEL_HYDRA_DEFAULT_NAME_STR))

        manager = \
            tf.train.CheckpointManager(
                checkpoint_name="ckpt",
                checkpoint=ckpt,
                directory=model_dir,
                max_to_keep=checkpoints_to_keep)

        def save_checkpoint_model_fn():
            # save model and weights
            logger.info("saving checkpoint at step: [{0}]".format(
                int(ckpt.step)))
            save_path = manager.save()
            logger.info(f"saved checkpoint to [{save_path}]")

        if manager.latest_checkpoint:
            logger.info("!!! Found checkpoint to restore !!!")
            ckpt \
                .restore(manager.latest_checkpoint) \
                .expect_partial()
            logger.info(f"restored checkpoint "
                        f"at epoch [{int(ckpt.epoch)}] "
                        f"and step [{int(ckpt.step)}]")
            # restore learning rate
            optimizer.iterations.assign(ckpt.step)
        else:
            logger.info("!!! Did NOT find checkpoint to restore !!!")
            if weights_dir is not None and \
                    len(weights_dir) > 0 and \
                    os.path.isdir(weights_dir):
                # restore weights from a directory
                loaded_weights = False

                for d in [weights_dir]:
                    if not loaded_weights:
                        try:
                            logger.info(f"loading weights from [{d}]")
                            tmp_model = tf.keras.models.clone_model(ckpt.model)
                            # restore checkpoint
                            tmp_checkpoint = create_checkpoint(model=tf.keras.models.clone_model(ckpt.model), path=d)
                            tmp_model.set_weights(tmp_checkpoint.model.get_weights())
                            ckpt.model = tmp_model
                            ckpt.step.assign(0)
                            ckpt.epoch.assign(0)
                            del tmp_model
                            del tmp_checkpoint
                            loaded_weights = True
                            logger.info("successfully loaded weights")
                        except Exception as e:
                            logger.info(
                                f"!!! failed to load weights from [{d}]] !!!")
                            logger.error(f"!!! {e}")

                if not loaded_weights:
                    logger.info("!!! failed to load weights")

            save_checkpoint_model_fn()

        # find indices of denoiser, materials segmentation, bg lumen wall segmentation
        # first third is denoiser, second third is materials, last third is bg_lumen_wall
        model_no_outputs = len(ckpt.model.outputs)
        model_indices = model_output_indices(no_outputs=model_no_outputs)
        denoiser_index = model_indices[DENOISER_STR]
        bg_fg_index = model_indices[SEGMENTATION_BG_FG_STR]
        lumen_wall_index = model_indices[SEGMENTATION_LUMEN_WALL_STR]
        materials_index = model_indices[SEGMENTATION_MATERIALS_STR]

        logger.info(f"model number of outputs: [{model_no_outputs}]")
        logger.info(f"model denoiser_index: {denoiser_index}")
        logger.info(f"model bg_fg_index: {bg_fg_index}")
        logger.info(f"model bg_lumen_wall_index: {lumen_wall_index}")
        logger.info(f"model materials_index: {materials_index}")

        @tf.function(reduce_retracing=True, jit_compile=False)
        def train_step(n: tf.Tensor) -> List[tf.Tensor]:
            return ckpt.model(n, training=True)

        @tf.function(reduce_retracing=True, jit_compile=False)
        def test_step(n: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            results = ckpt.model(n, training=False)
            return \
                results[denoiser_index[0]], \
                results[bg_fg_index[0]], \
                results[lumen_wall_index[0]], \
                results[materials_index[0]]

        if ckpt.step == 0:
            tf.summary.trace_on(graph=True, profiler=False)

            # run a single step
            _ = train_step(iter(dataset_training_materials).get_next()[0])

            tf.summary.trace_export(
                step=ckpt.step,
                name="model_hydra")
            tf.summary.flush()
            tf.summary.trace_off()

        sizes = [
            (int(input_shape[0] / (2 ** i)), int(input_shape[1] / (2 ** i)))
            for i in range(len(materials_index))
        ]

        segmentation_loss_fn_list = [
            tf.function(
                func=copy.deepcopy(loss_fn_map[SEGMENTATION_LOSS_FN_STR]),
                input_signature=[
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], NUMBER_OF_BG_FG_CLASSES], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], NUMBER_OF_BG_FG_CLASSES], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], NUMBER_OF_LUMEN_WALL_CLASSES], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], NUMBER_OF_LUMEN_WALL_CLASSES], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], NUMBER_OF_MATERIALS_CLASSES], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, sizes[i][0], sizes[i][1], NUMBER_OF_MATERIALS_CLASSES], dtype=tf.float32),
                ],
                reduce_retracing=True)
            for i in range(len(materials_index))
        ]

        # ---
        finished_training = False
        ssl_multiplier = tf.constant(1.0, dtype=tf.float32)
        task_multiplier = tf.constant(1.0, dtype=tf.float32)
        dataset_training_bg_lumen_wall_iter = iter(dataset_training_bg_lumen_wall)
        trainable_variables = ckpt.model.trainable_variables

        if use_rotational_invariance:
            transforms = [
                lambda x: x,
                lambda x: tf.image.flip_up_down(x),
                lambda x: tf.image.flip_left_right(x),
                lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x)),
            ]
        else:
            transforms = [lambda x: x]

        while not finished_training and \
                (total_epochs == -1 or ckpt.epoch < total_epochs):
            logger.info("epoch [{0}], step [{1}]".format(
                int(ckpt.epoch), int(ckpt.step)))

            start_time_epoch = time.time()

            # --- training percentage
            if total_epochs > 0:
                percentage_done = float(ckpt.epoch) / float(total_epochs)
            elif total_steps > 0:
                percentage_done = float(ckpt.step) / float(total_steps)
            else:
                percentage_done = 0.0

            depth_weight_str = [
                "{0:.2f}".format(output_discount_factor ** (float(i) * percentage_done))
                for i in range(len(materials_index))
            ]

            logger.info("percentage done [{:.2f}]".format(float(percentage_done)))
            logger.info(f"weight per output index: {depth_weight_str}")

            # --- initialize iterators
            epoch_finished_training = False
            dataset_eval_iter = iter(dataset_eval)
            dataset_train_materials_iter = iter(dataset_training_materials)
            dataset_test_materials_iter = iter(dataset_testing_materials)
            total_loss = tf.constant(0.0, dtype=tf.float32)
            gradients = [
                tf.constant(0.0, dtype=tf.float32)
                for _ in range(len(trainable_variables))
            ]
            gradients_moving_average = [
                tf.constant(0.0, dtype=tf.float32)
                for _ in range(len(trainable_variables))
            ]

            # --- decide on the mixture of learning (ssl, task, or mix)
            if ssl_epochs > 0:
                if ckpt.epoch < ssl_epochs:
                    ssl_multiplier = tf.constant(1.0, dtype=tf.float32)
                    task_multiplier = tf.constant(1.0, dtype=tf.float32)
                else:
                    logger.info("denoiser disabled")
                    ssl_multiplier = tf.constant(0.0, dtype=tf.float32)
                    task_multiplier = tf.constant(1.0, dtype=tf.float32)
            else:
                task_multiplier = tf.constant(1.0, dtype=tf.float32)
                ssl_multiplier = tf.constant(1.0, dtype=tf.float32)

            # --- check if total steps reached
            if total_steps != -1:
                if total_steps <= ckpt.step:
                    logger.info("total_steps reached [{0}]".format(
                        int(total_steps)))
                    finished_training = True

            total_denoiser_loss = tf.constant(0.0, dtype=tf.float32)
            total_segmentation_multiplier = tf.constant(0.0, dtype=tf.float32)
            total_segmentation_materials_loss = tf.constant(0.0, dtype=tf.float32)

            # --- iterate over the batches of the dataset
            while not finished_training and \
                    not epoch_finished_training:

                start_time_forward_backward = time.time()

                for _ in range(gpu_batches_per_step):
                    try:
                        (input_image_batch,
                         noisy_image_batch,
                         gt_all_one_hot_batch,
                         gt_bg_fg_one_hot_batch,
                         gt_lumen_wall_one_hot_batch,
                         gt_materials_one_hot_batch) = \
                            dataset_train_materials_iter.get_next()
                    except tf.errors.OutOfRangeError:
                        epoch_finished_training = True
                        break

                    scale_gt_image_batch = [input_image_batch]
                    scale_gt_bg_fg_one_hot_batch = [gt_bg_fg_one_hot_batch]
                    scale_gt_lumen_wall_one_hot_batch = [gt_lumen_wall_one_hot_batch]
                    scale_gt_materials_one_hot_batch = [gt_materials_one_hot_batch]

                    tmp_gt_image = input_image_batch
                    tmp_gt_bg_fg = gt_bg_fg_one_hot_batch + DEFAULT_EPSILON
                    tmp_gt_lumen_wall = gt_lumen_wall_one_hot_batch + DEFAULT_EPSILON
                    tmp_gt_materials = gt_materials_one_hot_batch + DEFAULT_EPSILON

                    for i in range(len(materials_index)-1):
                        tmp_gt_image = \
                            tf.nn.avg_pool2d(
                                input=tmp_gt_image,
                                ksize=(2, 2),
                                strides=(2, 2),
                                padding="SAME")
                        tmp_gt_bg_fg = \
                            tf.nn.avg_pool2d(
                                input=tmp_gt_bg_fg,
                                ksize=(2, 2),
                                strides=(2, 2),
                                padding="SAME")
                        tmp_gt_lumen_wall = \
                            tf.nn.avg_pool2d(
                                input=tmp_gt_lumen_wall,
                                ksize=(2, 2),
                                strides=(2, 2),
                                padding="SAME")
                        tmp_gt_materials = \
                            tf.nn.avg_pool2d(
                                input=tmp_gt_materials,
                                ksize=(2, 2),
                                strides=(2, 2),
                                padding="SAME")

                        tmp_gt_bg_fg = \
                            tmp_gt_bg_fg / \
                            tf.reduce_sum(tmp_gt_bg_fg, axis=-1, keepdims=True)
                        tmp_gt_lumen_wall = \
                            tmp_gt_lumen_wall / \
                            tf.reduce_sum(tmp_gt_lumen_wall, axis=-1, keepdims=True)
                        tmp_gt_materials = \
                            tmp_gt_materials / \
                            tf.reduce_sum(tmp_gt_materials, axis=-1, keepdims=True)

                        scale_gt_image_batch.append(tmp_gt_image)
                        scale_gt_bg_fg_one_hot_batch.append(tmp_gt_bg_fg)
                        scale_gt_materials_one_hot_batch.append(tmp_gt_materials)
                        scale_gt_lumen_wall_one_hot_batch.append(tmp_gt_lumen_wall)

                    # rotational transforms to make it invariant
                    for transform in transforms:
                        noisy_image_batch_t = transform(noisy_image_batch)

                        with tf.GradientTape() as tape:
                            predictions = \
                                train_step(noisy_image_batch_t)

                            prediction_denoiser = [
                                predictions[i] for i in denoiser_index
                            ]
                            prediction_materials_one_hot = [
                                predictions[i] for i in materials_index
                            ]
                            prediction_bg_fg_one_hot = [
                                predictions[i] for i in bg_fg_index
                            ]
                            prediction_lumen_wall_one_hot = [
                                predictions[i] for i in lumen_wall_index
                            ]

                            # compute the loss value for this mini-batch
                            all_denoiser_loss = [
                                denoiser_loss_fn(
                                    input_batch=transform(scale_gt_image_batch[i]),
                                    predicted_batch=prediction_denoiser[i])
                                for i in range(len(prediction_denoiser))
                            ]

                            # get segmentation loss for each depth,
                            # assumes top performance is the first
                            all_segmentation_loss = [
                                segmentation_loss_fn_list[i](
                                    transform(scale_gt_bg_fg_one_hot_batch[i]),
                                    prediction_bg_fg_one_hot[i],
                                    transform(scale_gt_lumen_wall_one_hot_batch[i]),
                                    prediction_lumen_wall_one_hot[i],
                                    transform(scale_gt_materials_one_hot_batch[i]),
                                    prediction_materials_one_hot[i])
                                for i in range(len(prediction_materials_one_hot))
                            ]

                            total_denoiser_loss *= 0.0
                            total_segmentation_multiplier *= 0.0
                            total_segmentation_materials_loss *= 0.0

                            for i, s in enumerate(all_denoiser_loss):
                                total_denoiser_loss += s[TOTAL_LOSS_STR]

                            for i, s in enumerate(all_segmentation_loss):
                                depth_weight = float(output_discount_factor ** (float(i) * percentage_done))
                                total_segmentation_materials_loss += s[TOTAL_LOSS_STR] * depth_weight
                                total_segmentation_multiplier += depth_weight

                            # combine losses
                            model_loss = model_loss_fn(model=ckpt.model)
                            total_loss = \
                                task_multiplier * total_segmentation_materials_loss / total_segmentation_multiplier + \
                                (1.0 - percentage_done) * ssl_multiplier * total_denoiser_loss + \
                                model_loss[TOTAL_LOSS_STR]

                            gradient = \
                                tape.gradient(
                                    target=total_loss,
                                    sources=trainable_variables)

                        for i, grad in enumerate(gradient):
                            gradients[i] += grad
                        del gradient

                # average out gradients
                for i in range(len(gradients)):
                    gradients[i] /= (float(gpu_batches_per_step) * float(len(transforms)))

                # apply gradient to change weights
                optimizer.apply_gradients(
                    grads_and_vars=zip(
                        gradients,
                        trainable_variables))

                # --- zero gradients to reuse it in the next iteration
                # moved at the end, so we can use it for visualization
                for i in range(len(gradients)):
                    gradients_moving_average[i] = \
                        gradients_moving_average[i] * 0.99 + \
                        gradients[i] * 0.01
                    gradients[i] *= 0.0

                # !!! IMPORTANT !!!!
                # keep first segmentation loss result for visualization
                segmentation_materials_loss = all_segmentation_loss[0]

                # --- add loss summaries for tensorboard
                # denoiser
                for i, d in enumerate(all_denoiser_loss):
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/mae",
                                      data=d[MAE_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/ssim",
                                      data=d[SSIM_LOSS_STR],
                                      step=ckpt.step)
                    tf.summary.scalar(name=f"loss_denoiser/scale_{i}/total",
                                      data=d[TOTAL_LOSS_STR],
                                      step=ckpt.step)

                # segmentation materials
                tf.summary.scalar(name="loss_segmentation/train/dice",
                                  data=segmentation_materials_loss[DICE_STR],
                                  step=ckpt.step)
                tf.summary.scalar(name="loss_segmentation/train/cce",
                                  data=segmentation_materials_loss[CCE_STR],
                                  step=ckpt.step)
                tf.summary.scalar(name="loss_segmentation/train/sfce",
                                  data=segmentation_materials_loss[SFCE_STR],
                                  step=ckpt.step)
                tf.summary.scalar(name="loss_segmentation/train/edge",
                                  data=segmentation_materials_loss[EDGE_STR],
                                  step=ckpt.step)
                tf.summary.scalar(name="loss_segmentation/train/total",
                                  data=segmentation_materials_loss[TOTAL_LOSS_STR],
                                  step=ckpt.step)

                # model
                tf.summary.scalar(name="loss/regularization",
                                  data=model_loss[REGULARIZATION_LOSS_STR],
                                  step=ckpt.step)
                tf.summary.scalar(name="loss/total",
                                  data=total_loss,
                                  step=ckpt.step)

                # --- add train scales to visualization
                if (ckpt.step % visualization_every) == 0:
                    # train scales
                    for i in range(len(materials_index)):
                        p_bg_fg = prediction_bg_fg_one_hot[i]
                        p_lumen_wall = prediction_lumen_wall_one_hot[i]
                        p_materials = prediction_materials_one_hot[i]
                        p_one_hot = models.postprocessor([p_bg_fg, p_lumen_wall, p_materials])

                        m_i_soft = one_hot_to_color(p_one_hot, normalize=True)
                        m_i_hard = colorize_tensor_hard(p_one_hot, normalize=True)
                        b_i_soft = one_hot_to_color(p_bg_fg, normalize=True)
                        b_i_hard = colorize_tensor_hard(p_bg_fg, normalize=True)

                        tf.summary.image(
                            name=f"scales/output_{i}",
                            data=tf.concat(
                                values=[
                                    tf.concat(values=[m_i_soft, m_i_hard], axis=1),
                                    tf.concat(values=[b_i_soft, b_i_hard], axis=1)
                                ],
                                axis=2
                            ),
                            max_outputs=visualization_number,
                            step=ckpt.step)

                # --- add image prediction for tensorboard
                if (ckpt.step % visualization_every) == 0:
                    # --- denoiser
                    tf.summary.image(name="denoiser/input", data=input_image_batch / 255,
                                     max_outputs=visualization_number, step=ckpt.step)
                    # noisy batch
                    tf.summary.image(name="denoiser/noisy", data=noisy_image_batch / 255,
                                     max_outputs=visualization_number, step=ckpt.step)
                    # denoised batch
                    for i, d in enumerate(prediction_denoiser):
                        tf.summary.image(name=f"denoiser/scale_{i}/output", data=d / 255,
                                         max_outputs=visualization_number, step=ckpt.step)

                    # --- test on clean to get clean cut version visualization
                    _, prediction_bg_fg, prediction_lumen_wall, prediction_materials = \
                        test_step(input_image_batch)
                    prediction_one_hot = \
                        models.postprocessor([
                            prediction_bg_fg,
                            prediction_lumen_wall,
                            prediction_materials])

                    # --- inputs / ground truth
                    # original gt
                    tf.summary.image(name="input/gt",
                                     data=one_hot_to_color(gt_all_one_hot_batch, normalize=True),
                                     max_outputs=visualization_number, step=ckpt.step)

                    # --- outputs
                    # predicted segmentation probabilities
                    tf.summary.image(name="segmentation/soft",
                                     data=one_hot_to_color(prediction_one_hot, normalize=True),
                                     max_outputs=visualization_number, step=ckpt.step)
                    # predicted segmentation classes
                    tf.summary.image(name="segmentation/hard",
                                     data=colorize_tensor_hard(prediction_one_hot, normalize=True),
                                     max_outputs=visualization_number, step=ckpt.step)

                    # --- comparison
                    # raw input with ground truth overlay
                    tf.summary.image(name="comparison/raw_gt",
                                     data=one_hot_to_color(gt_all_one_hot_batch, normalize=True) / 2 +
                                          (input_image_batch / 255) / 2,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="ground truth labels with raw data")
                    # raw input with prediction overlay
                    tf.summary.image(name="comparison/raw_prediction",
                                     data=colorize_tensor_hard(prediction_one_hot, normalize=True) / 2 +
                                          (input_image_batch / 255) / 2,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="inference with raw data")

                    gt_prediction_xor = \
                        pixel_differences(
                            gt_all_one_hot_batch,
                            prediction_one_hot)

                    tf.summary.image(name="comparison/xor",
                                     data=gt_prediction_xor,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="show different pixels between "
                                                 "ground truth and prediction")

                    # --- add gradient activity
                    gradient_activity = \
                        visualize_gradient_boxplot(
                            gradients=gradients_moving_average,
                            trainable_variables=trainable_variables) / 255
                    tf.summary.image(name="weights/gradients",
                                     data=gradient_activity,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="gradient activity")

                    # --- add weights distribution
                    weights_boxplot = \
                        visualize_weights_boxplot(
                            trainable_variables=trainable_variables) / 255
                    tf.summary.image(name="weights/boxplot",
                                     data=weights_boxplot,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="weights boxplot")
                    weights_heatmap = \
                        visualize_weights_heatmap(
                            trainable_variables=trainable_variables) / 255
                    tf.summary.image(name="weights/heatmap",
                                     data=weights_heatmap,
                                     max_outputs=visualization_number,
                                     step=ckpt.step,
                                     description="weights heatmap")

                    # --- eval
                    eval_done = False
                    while not eval_done:
                        try:
                            (input_image_batch,
                             _,
                             gt_all_one_hot_batch,
                             _,
                             _,
                             _) = \
                                dataset_eval_iter.get_next()
                            _, prediction_bg_fg, \
                                prediction_lumen_wall_one_hot, \
                                prediction_materials_one_hot = test_step(input_image_batch)

                            prediction_one_hot_hard = \
                                models.postprocessor([prediction_bg_fg,
                                                      prediction_lumen_wall_one_hot,
                                                      prediction_materials_one_hot])

                            # original input
                            tf.summary.image(
                                name="eval/raw", data=input_image_batch / 255,
                                max_outputs=visualization_number, step=ckpt.step)
                            # raw input with ground truth overlay
                            tf.summary.image(
                                name="eval/raw_gt",
                                data=one_hot_to_color(gt_all_one_hot_batch, normalize=True) / 2 +
                                     (input_image_batch / 255) / 2,
                                max_outputs=visualization_number, step=ckpt.step)
                            # raw input with prediction overlay
                            tf.summary.image(
                                name="eval/raw_prediction",
                                data=colorize_tensor_hard(prediction_one_hot_hard, normalize=True) / 2 +
                                     (input_image_batch / 255) / 2,
                                max_outputs=visualization_number, step=ckpt.step)
                            eval_done = True
                        except tf.errors.OutOfRangeError:
                            dataset_eval_iter = iter(dataset_eval)

                # --- test
                test_done = False
                while not test_done:
                    try:
                        (input_image_batch,
                         _,
                         _,
                         gt_bg_fg_one_hot_batch,
                         gt_lumen_wall_one_hot_batch,
                         gt_materials_one_hot_batch) = \
                            dataset_test_materials_iter.get_next()
                        _, prediction_bg_fg, \
                            prediction_lumen_wall_one_hot, \
                            prediction_materials_one_hot = test_step(input_image_batch)

                        segmentation_materials_loss = \
                            segmentation_loss_fn_list[0](
                                gt_bg_fg_one_hot_batch,
                                prediction_bg_fg,
                                gt_lumen_wall_one_hot_batch,
                                prediction_lumen_wall_one_hot,
                                gt_materials_one_hot_batch,
                                prediction_materials_one_hot)

                        # segmentation
                        tf.summary.scalar(name="loss_segmentation/test/dice",
                                          data=segmentation_materials_loss[DICE_STR],
                                          step=ckpt.step)
                        tf.summary.scalar(name="loss_segmentation/test/cce",
                                          data=segmentation_materials_loss[CCE_STR],
                                          step=ckpt.step)
                        tf.summary.scalar(name="loss_segmentation/test/sfce",
                                          data=segmentation_materials_loss[SFCE_STR],
                                          step=ckpt.step)
                        tf.summary.scalar(name="loss_segmentation/test/edge",
                                          data=segmentation_materials_loss[EDGE_STR],
                                          step=ckpt.step)
                        tf.summary.scalar(name="loss_segmentation/test/total",
                                          data=segmentation_materials_loss[TOTAL_LOSS_STR],
                                          step=ckpt.step)
                        test_done = True
                    except tf.errors.OutOfRangeError:
                        dataset_test_materials_iter = iter(dataset_testing_materials)

                # --- check if it is time to save a checkpoint
                if checkpoint_every > 0 and ckpt.step > 0 and \
                        (ckpt.step % checkpoint_every == 0):
                    save_checkpoint_model_fn()

                # --- keep time of steps per second
                stop_time_forward_backward = time.time()
                step_time_forward_backward = \
                    stop_time_forward_backward - \
                    start_time_forward_backward

                tf.summary.scalar(name="training/epoch",
                                  data=int(ckpt.epoch),
                                  step=ckpt.step)
                tf.summary.scalar(name="training/learning_rate",
                                  data=optimizer.learning_rate,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/steps_per_second",
                                  data=1.0 / (step_time_forward_backward + 0.00001),
                                  step=ckpt.step)

                # ---
                ckpt.step.assign_add(1)

                # --- check if total steps reached
                if total_steps > 0:
                    if total_steps <= ckpt.step:
                        logger.info("total_steps reached [{0}]".format(
                            int(total_steps)))
                        finished_training = True

            end_time_epoch = time.time()
            epoch_time = end_time_epoch - start_time_epoch

            # --- evaluation at the end of the epoch
            for d, d_str, d_description_str in [
                (dataset_training_materials.take(count=100), "train", "train: 90% of data with augmentations"),
                (dataset_testing_materials, "test", "test: 10% dataset without augmentations"),
                (dataset_eval, "eval", "eval: 100% dataset without augmentations")
            ]:
                m = metrics(dataset=d, test_fn=test_step, postprocessor_fn=models.postprocessor)
                tf.summary.scalar(name=f"metrics/{d_str}/{IOU_STR}",
                                  data=m[IOU_STR],
                                  step=ckpt.step,
                                  description=d_description_str)
                tf.summary.scalar(name=f"metrics/{d_str}/{CATEGORICAL_ACCURACY_STR}",
                                  data=m[CATEGORICAL_ACCURACY_STR],
                                  step=ckpt.step,
                                  description=d_description_str)
                tf.summary.scalar(name=f"metrics/{d_str}/{DICE_STR}",
                                  data=m[DICE_STR],
                                  step=ckpt.step,
                                  description=d_description_str)
                cm = m[CONFUSION_MATRIX_STR]

                tf.summary.image(
                    name=f"confusion_matrix/{d_str}",
                    data=visualize_confusion_matrix(
                        cm=cm, title="Confusion Matrix") / 255,
                    max_outputs=visualization_number,
                    description="confusion matrix",
                    step=ckpt.step)

            # --- end of the epoch
            logger.info("end of epoch [{0}], took [{1}] seconds".format(
                int(ckpt.epoch), int(round(epoch_time))))
            ckpt.epoch.assign_add(1)
            save_checkpoint_model_fn()

    logger.info("finished training")
    return

# ---------------------------------------------------------------------

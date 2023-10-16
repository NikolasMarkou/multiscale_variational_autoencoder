import os
import time
import json
import psutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from pprint import pformat
from typing import Union, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .dataset import *
from .constants import *

from .custom_logger import logger
from .file_operations import load_image
from .optimizer import optimizer_builder
from .loss import loss_function_builder
from .utilities_checkpoints import \
    create_checkpoint, \
    model_weights_from_checkpoint
from .models import \
    model_builder, \
    model_output_indices
from .utilities import \
    load_config, \
    save_config, \
    depthwise_gaussian_kernel
from .visualize import \
    visualize_weights_boxplot, \
    visualize_weights_heatmap, \
    visualize_gradient_boxplot

# ---------------------------------------------------------------------

CURRENT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))

# ---------------------------------------------------------------------


def train_loop(
        pipeline_config_path: Union[str, Dict, Path],
        model_dir: Union[str, Path],
        weights_dir: Union[str, Path] = None):
    """
    Trains a blind image denoiser

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
    batch_size = dataset.batch_size
    input_shape = dataset.input_shape

    # --- build loss function
    loss_fn_map = loss_function_builder(config=config["loss"])
    model_loss_fn = \
        tf.function(
            func=loss_fn_map[MODEL_LOSS_FN_STR],
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
    epochs = train_config["epochs"]
    gpu_batches_per_step = int(train_config.get("gpu_batches_per_step", 1))
    if gpu_batches_per_step <= 0:
        raise ValueError("gpu_batches_per_step must be > 0")
    gpu_batches_per_step_float = float(gpu_batches_per_step)

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
    with ((tf.summary.create_file_writer(model_dir).as_default())):
        # --- write configuration in tensorboard
        tf.summary.text("config", pformat(config), step=0)

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
                hydra=models.hydra,
                encoder=models.encoder,
                decoder=models.decoder,
                path=None)
        # summary of model and save model, so we can inspect with netron
        ckpt.hydra.summary(print_fn=logger.info)
        ckpt.hydra.save(
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
                            tmp_hydra = tf.keras.models.clone_model(ckpt.hydra)
                            # restore checkpoint
                            tmp_checkpoint = create_checkpoint(
                                encoder=tf.keras.models.clone_model(ckpt.encoder),
                                decoder=tf.keras.models.clone_model(ckpt.decoder),
                                hydra=tf.keras.models.clone_model(ckpt.hydra),
                                path=d)
                            tmp_hydra.set_weights(tmp_checkpoint.hydra.get_weights())
                            ckpt.hydra = tmp_hydra
                            ckpt.step.assign(0)
                            ckpt.epoch.assign(0)
                            del tmp_hydra
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

        # find indices of denoiser,
        model_no_outputs = model_output_indices(len(ckpt.hydra.outputs))
        denoiser_index = model_no_outputs[DENOISER_STR]
        denoiser_levels = len(denoiser_index)

        logger.info(f"model number of outputs: \n"
                    f"{pformat(model_no_outputs)}")

        # this is set only once in the first call
        gaussian_kernel = \
            tf.constant(
                depthwise_gaussian_kernel(
                    channels=input_shape[-1],
                    kernel_size=(5, 5),
                    dtype=np.float32),
                dtype=tf.float32)

        denoiser_loss_fn = \
            tf.function(
                func=copy.deepcopy(loss_fn_map[DENOISER_LOSS_FN_STR]),
                autograph=True,
                jit_compile=False,
                input_signature=[
                    tf.TensorSpec(shape=[batch_size, None, None, input_shape[-1]], dtype=tf.float32),
                    tf.TensorSpec(shape=[batch_size, None, None, input_shape[-1]], dtype=tf.float32),
                ],
                reduce_retracing=True)

        @tf.function(reduce_retracing=True, jit_compile=False, autograph=True)
        def train_denoiser_step(n: tf.Tensor) -> List[tf.Tensor]:
            return ckpt.hydra(n, training=True)

        @tf.function(reduce_retracing=True, jit_compile=False, autograph=True)
        def test_denoiser_step(n: tf.Tensor) -> tf.Tensor:
            results = ckpt.hydra(n, training=False)
            return results[denoiser_index[0]]

        @tf.function(reduce_retracing=True, jit_compile=False, autograph=True)
        def downsample_step(n: tf.Tensor) -> List[tf.Tensor]:
            scales = []
            n_scale = n

            for _ in range(denoiser_levels):
                scales.append(n_scale)
                # downsample, clip and round
                n_scale = \
                    tf.round(
                        tf.clip_by_value(
                            tf.nn.depthwise_conv2d(
                                input=n_scale,
                                filter=gaussian_kernel,
                                strides=(1, 2, 2, 1),
                                data_format=None,
                                dilations=None,
                                padding="SAME"),
                            clip_value_min=0.0,
                            clip_value_max=255.0))
            return scales

        if ckpt.step == 0:
            # trace train network
            tf.summary.trace_on(graph=True, profiler=False)
            _ = train_denoiser_step(iter(dataset.training).get_next()[0])
            tf.summary.trace_export(
                step=ckpt.step,
                name="train/model_hydra")
            tf.summary.flush()
            tf.summary.trace_off()

            # trace test network
            tf.summary.trace_on(graph=True, profiler=False)
            _ = test_denoiser_step(iter(dataset.training).get_next()[0])
            tf.summary.trace_export(
                step=ckpt.step,
                name="test/model_hydra")
            tf.summary.flush()
            tf.summary.trace_off()

        finished_training = False
        trainable_variables = ckpt.hydra.trainable_variables
        depth_weight = [
            0.0
            for _ in range(denoiser_levels)
        ]
        gradients = [
            0.0
            for _ in range(len(trainable_variables))
        ]

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

            # adjust weights per depth
            sum_depth_weight = 0.0
            for i in range(len(denoiser_index)):
                depth_weight[i] = \
                            float(output_discount_factor ** (float(i) * percentage_done))
                sum_depth_weight += depth_weight[i]

            for i in range(len(denoiser_index)):
                depth_weight[i] = depth_weight[i] / sum_depth_weight

            depth_weight_str = [
                "{0:.2f}".format(depth_weight[i])
                for i in range(len(denoiser_index))
            ]

            logger.info("percentage done [{:.2f}]".format(float(percentage_done)))
            logger.info(f"weight per output index: {depth_weight_str}")

            # --- initialize iterators
            epoch_finished_training = False
            dataset_train = iter(dataset.training)
            dataset_test = iter(dataset.testing)

            # --- check if total steps reached
            if total_steps != -1:
                if total_steps <= ckpt.step:
                    logger.info("total_steps reached [{0}]".format(
                        int(total_steps)))
                    finished_training = True

            # --- iterate over the batches of the dataset
            while not finished_training and \
                    not epoch_finished_training:

                start_time_forward_backward = time.time()

                for i in range(len(gradients)):
                    gradients[i] *= 0.0

                for b in range(gpu_batches_per_step):
                    try:
                        (input_image_batch, noisy_image_batch) = \
                            dataset_train.get_next()

                        scale_gt_image_batch = \
                            downsample_step(input_image_batch)

                        # zero out loss
                        with tf.GradientTape(persistent=False,
                                             watch_accessed_variables=False) as tape:
                            tape.watch(trainable_variables)

                            total_denoiser_loss = 0.0
                            predictions = \
                                train_denoiser_step(noisy_image_batch)

                            # compute the loss value for this mini-batch
                            for i, idx in enumerate(denoiser_index):
                                loss_train = \
                                    denoiser_loss_fn(
                                        input_batch=scale_gt_image_batch[i],
                                        predicted_batch=predictions[idx])
                                total_denoiser_loss += \
                                    loss_train[TOTAL_LOSS_STR] * \
                                    depth_weight[i]

                            # combine losses
                            model_loss = \
                                model_loss_fn(model=ckpt.hydra)
                            total_loss = \
                                total_denoiser_loss + \
                                model_loss[TOTAL_LOSS_STR]

                            gradient = \
                                tape.gradient(
                                    target=total_loss,
                                    sources=trainable_variables)

                            # aggregate gradients
                            for i, grad in enumerate(gradient):
                                gradients[i] += (grad / gpu_batches_per_step_float)
                            del gradient

                        del model_loss
                        del total_loss
                        del predictions
                        del input_image_batch
                        del noisy_image_batch
                        del total_denoiser_loss
                        del scale_gt_image_batch

                        # apply gradient to change weights
                        optimizer.apply_gradients(
                            grads_and_vars=zip(
                                gradients,
                                trainable_variables),
                            skip_gradients_aggregation=False)

                    except tf.errors.OutOfRangeError:
                        epoch_finished_training = True
                        break

                # # --- add loss summaries for tensorboard
                # tf.summary.scalar(name=f"train/mae",
                #                   data=all_denoiser_loss[0][MAE_LOSS_STR],
                #                   step=ckpt.step)
                # tf.summary.scalar(name=f"train/ssim",
                #                   data=all_denoiser_loss[0][SSIM_LOSS_STR],
                #                   step=ckpt.step)
                # tf.summary.scalar(name=f"train/total",
                #                   data=all_denoiser_loss[0][TOTAL_LOSS_STR],
                #                   step=ckpt.step)
                # for i, loss_train in enumerate(all_denoiser_loss):
                #     tf.summary.scalar(name=f"debug/scale_{i}/mae",
                #                       data=loss_train[MAE_LOSS_STR],
                #                       step=ckpt.step)
                # tf.summary.scalar(name="loss/regularization",
                #                   data=model_loss[REGULARIZATION_LOSS_STR],
                #                   step=ckpt.step)
                # tf.summary.scalar(name="loss/total",
                #                   data=total_loss,
                #                   step=ckpt.step)
                # del total_loss
                # del model_loss
                # del all_denoiser_loss
                #
                # # --- add image prediction for tensorboard
                # if (ckpt.step % visualization_every) == 0:
                #     # add prediction of training image using the test (to stop dropout)
                #     predictions = \
                #         test_denoiser_step(noisy_image_batch)
                #
                #     x_error = \
                #         tf.clip_by_value(
                #             tf.abs(input_image_batch - predictions),
                #             clip_value_min=0.0,
                #             clip_value_max=255.0
                #         )
                #     x_collage = \
                #         np.concatenate(
                #             [
                #                 np.concatenate(
                #                     [input_image_batch.numpy(), noisy_image_batch.numpy()],
                #                     axis=2),
                #                 np.concatenate(
                #                     [predictions.numpy(), x_error.numpy()],
                #                     axis=2)
                #             ],
                #             axis=1) / 255
                #     tf.summary.image(name="train/collage",
                #                      data=x_collage,
                #                      max_outputs=visualization_number,
                #                      step=ckpt.step)
                #     del x_error
                #     del x_collage
                #     del predictions
                #
                #     # add weights heatmap and boxplot distribution
                #     weights_boxplot = \
                #         visualize_weights_boxplot(
                #             trainable_variables=trainable_variables) / 255
                #     tf.summary.image(name="weights/boxplot",
                #                      data=weights_boxplot,
                #                      max_outputs=visualization_number,
                #                      step=ckpt.step,
                #                      description="weights boxplot")
                #     del weights_boxplot
                #
                #     weights_heatmap = \
                #         visualize_weights_heatmap(
                #             trainable_variables=trainable_variables) / 255
                #     tf.summary.image(name="weights/heatmap",
                #                      data=weights_heatmap,
                #                      max_outputs=visualization_number,
                #                      step=ckpt.step,
                #                      description="weights heatmap")
                #     del weights_heatmap
                #
                # del input_image_batch
                # del noisy_image_batch
                #
                # # --- test
                # test_done = False
                # while not test_done:
                #     try:
                #         (input_image_batch_test, noisy_image_batch_test) = \
                #             dataset_test.get_next()
                #         predictions = \
                #             test_denoiser_step(noisy_image_batch_test)
                #
                #         # compute the loss value for this mini-batch
                #         loss_test = \
                #             denoiser_loss_fn(
                #                 input_batch=input_image_batch_test,
                #                 predicted_batch=predictions)
                #         tf.summary.scalar(name=f"test/mae",
                #                           data=loss_test[MAE_LOSS_STR],
                #                           step=ckpt.step)
                #         tf.summary.scalar(name=f"test/ssim",
                #                           data=loss_test[SSIM_LOSS_STR],
                #                           step=ckpt.step)
                #         tf.summary.scalar(name=f"test/total",
                #                           data=loss_test[TOTAL_LOSS_STR],
                #                           step=ckpt.step)
                #
                #         if (ckpt.step % visualization_every) == 0:
                #             x_error = \
                #                 tf.clip_by_value(
                #                     tf.abs(input_image_batch_test - predictions),
                #                     clip_value_min=0.0,
                #                     clip_value_max=255.0
                #                 )
                #             x_collage = \
                #                 np.concatenate(
                #                     [
                #                         np.concatenate(
                #                             [input_image_batch_test.numpy(), noisy_image_batch_test.numpy()],
                #                             axis=2),
                #                         np.concatenate(
                #                             [predictions.numpy(), x_error.numpy()],
                #                             axis=2)
                #                     ],
                #                     axis=1) / 255
                #             tf.summary.image(name="test/collage",
                #                              data=x_collage,
                #                              max_outputs=visualization_number,
                #                              step=ckpt.step)
                #             del x_collage
                #             del noisy_image_batch_test
                #             del input_image_batch_test
                #         test_done = True
                #     except tf.errors.OutOfRangeError:
                #         dataset_test = iter(dataset.testing)

                # --- check if it is time to save a checkpoint
                if checkpoint_every > 0 and ckpt.step > 0 and \
                        (ckpt.step % checkpoint_every == 0):
                    save_checkpoint_model_fn()

                # --- keep time of steps per second
                stop_time_forward_backward = time.time()
                step_time_forward_backward = \
                    stop_time_forward_backward - \
                    start_time_forward_backward

                # Getting all memory using os.popen()
                ram_percentage = psutil.virtual_memory()[2]
                ram_gb = psutil.virtual_memory()[3]/1000000000

                tf.summary.scalar(name="training/ram % used",
                                  data=ram_percentage,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/ram gb used",
                                  data=ram_gb,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/epoch",
                                  data=int(ckpt.epoch),
                                  step=ckpt.step)
                tf.summary.scalar(name="training/learning_rate",
                                  data=optimizer.learning_rate,
                                  step=ckpt.step)
                tf.summary.scalar(name="training/steps_per_second",
                                  data=1.0 / (step_time_forward_backward + DEFAULT_EPSILON),
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

            # --- end of the epoch
            logger.info("end of epoch [{0}], took [{1}] seconds".format(
                int(ckpt.epoch), int(round(epoch_time))))
            ckpt.epoch.assign_add(1)
            save_checkpoint_model_fn()

    logger.info("finished training")
    return

# ---------------------------------------------------------------------

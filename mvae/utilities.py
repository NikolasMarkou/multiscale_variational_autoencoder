import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# ==============================================================================


def collage(images_batch):
    """
    Create a collage of image from a batch

    :param images_batch:
    :return:
    """
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))

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

# ==============================================================================


def noisy_image_data_generator(
        dataset,
        batch_size: int = 32,
        min_value: float = 0.0,
        max_value: float = 255.0,
        min_noise_std: float = 0.0,
        max_noise_std: float = 1.0,
        random_invert: bool = False,
        random_brightness: bool = False,
        zoom_range: float = 0.25,
        rotation_range: int = 90,
        width_shift_range: float = 0.0,
        height_shift_range: float = 0.0,
        vertical_flip: bool = True,
        horizontal_flip: bool = True):
    """
    Create a dataset generator flow that adds noise to a dateset

    :param dataset:
    :param min_value: Minimum allowed value
    :param max_value: Maximum allowed value
    :param batch_size: Batch size
    :param random_invert: Randomly (50%) invert the image
    :param random_brightness: Randomly add offset or multiplier
    :param zoom_range: Randomly zoom in (percentage)
    :param rotation_range: Add random rotation range (in degrees)
    :param min_noise_std: Min standard deviation of noise
    :param max_noise_std: Max standard deviation of noise
    :param horizontal_flip: Randomly horizontally flip image
    :param vertical_flip: Randomly vertically flip image
    :param height_shift_range: Add random vertical shift (percentage of image)
    :param width_shift_range: Add random horizontal shift (percentage of image)

    :return:
    """
    # --- argument checking
    if dataset is None:
        raise ValueError("dataset cannot be empty")
    if min_noise_std > max_noise_std:
        raise ValueError("min_noise_std must be < max_noise_std")
    if min_value > max_value:
        raise ValueError("min_value must be < max_value")

    # --- variables setup
    max_min_diff = (max_value - min_value)

    # --- create data generator
    if isinstance(dataset, np.ndarray):
        data_generator = \
            ImageDataGenerator(
                zoom_range=zoom_range,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                vertical_flip=vertical_flip,
                horizontal_flip=horizontal_flip,
                zca_whitening=False,
                featurewise_center=False,
                featurewise_std_normalization=False)
    else:
        raise NotImplementedError()

    # --- iterate over random batches
    for x_batch in \
            data_generator.flow(
                x=dataset,
                shuffle=True,
                batch_size=batch_size):
        # randomly invert batch
        if random_invert:
            if np.random.choice([False, True]):
                x_batch = (max_value - x_batch) + min_value

        # add random offset
        if random_brightness:
            if np.random.choice([False, True]):
                offset = \
                    np.random.uniform(
                        low=0.0,
                        high=0.1 * max_min_diff)
                x_batch = x_batch + offset

        # adjust the std of the noise
        # pick std between min and max std
        if max_noise_std > 0.0:
            if np.random.choice([False, True]):
                std = \
                    np.random.uniform(
                        low=min_noise_std,
                        high=max_noise_std)

                # add noise to create the noisy input
                x_batch = \
                    x_batch + \
                    np.random.normal(0.0, std, x_batch.shape)

        # clip all to be between min and max value
        x_batch = \
            np.clip(
                x_batch,
                a_min=min_value,
                a_max=max_value)

        # return input, target
        yield x_batch, x_batch


# ==============================================================================


def get_conv2d_weights(
        model: keras.Model):
    """
    Get the conv2d weights from the model
    """
    weights = []
    for layer in model.layers:
        layer_config = layer.get_config()
        layer_weights = layer.get_weights()
        if "layers" not in layer_config:
            continue
        for i, l in enumerate(layer_config["layers"]):
            if l["class_name"] == "Conv2D":
                for w in layer_weights[i]:
                    w_flat = w.flatten()
                    weights.append(w_flat)
    return np.concatenate(weights)

# ==============================================================================

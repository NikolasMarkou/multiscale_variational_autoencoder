import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Union, List

# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------


class Mish(tf.keras.layers.Layer):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1

    x = x * tanh(softplus(x))
    """

    def __init__(self,
                 name=None,
                 **kwargs):
        super(Mish, self).__init__(
            trainable=False,
            name=name,
            **kwargs)

    def build(self, input_shape):
        super(Mish, self).build(input_shape)

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {}


# ---------------------------------------------------------------------

class GaussianFilter(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Tuple[int, int] = [1, 1, 1, 1],
            sigma: Tuple[float, float] = (1.0, 1.0),
            name: str = None,
            **kwargs):
        super(GaussianFilter, self).__init__(
            trainable=False,
            name=name,
            **kwargs)
        if len(kernel_size) != 2:
            raise ValueError("kernel size must be length 2")
        if len(strides) != 2:
            raise ValueError("strides size must be length 2")
        if len(sigma) != 2:
            raise ValueError("sigma size must be length 2")
        self._kernel_size = kernel_size
        self._strides = strides
        self._sigma = sigma
        self._kernel = None

    def build(self, input_shape):
        def gaussian_kernel(
                size: Tuple[int, int],
                nsig: Tuple[float, float],
                dtype: np.float64) -> np.ndarray:
            kern1d = [
                np.linspace(
                    start=-np.abs(nsig[i]),
                    stop=np.abs(nsig[i]),
                    num=size[i],
                    endpoint=True,
                    dtype=dtype)
                for i in range(2)
            ]
            x, y = np.meshgrid(kern1d[0], kern1d[1])
            d = np.sqrt(x * x + y * y)
            sigma, mu = 1.0, 0.0
            g = np.exp(-((d - mu) ** 2 / (2.0 * (sigma ** 2))))
            return g / g.sum()
        def kernel_init(shape, dtype):
            logger.info(f"building gaussian kernel with size: {shape}")
            kernel = np.zeros(shape)
            kernel_channel = \
                gaussian_kernel(
                    size=(shape[0], shape[1]),
                    nsig=self._sigma)
            for i in range(shape[2]):
                kernel[:, :, i, 0] = kernel_channel
            return kernel
        # [filter_height, filter_width, in_channels, channel_multiplier]
        self._kernel = \
            kernel_init(
                shape=(self._kernel_size[0], self._kernel_size[1], input_shape[-1], 1))

    def get_config(self):
        return {
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "sigma": self._sigma
        }

    def call(self, inputs, training):
        return \
            tf.nn.depthwise_conv2d(
                input=inputs,
                filter=self._kernel,
                strides=self._strides,
                data_format=None,
                dilations=None,
                padding="SAME")


# ---------------------------------------------------------------------


class RandomOnOff(tf.keras.layers.Layer):
    """randomly drops the whole connection"""

    def __init__(self,
                 rate: float = 0.5,
                 trainable: bool = False,
                 name=None,
                 **kwargs):
        super(RandomOnOff, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self._w0 = None
        self._rate = rate
        self._dropout = None

    def build(self, input_shape):
        def init_w0_fn(shape, dtype):
            return np.ones(shape, dtype=np.float32)

        self._w0 = \
            self.add_variable(
                shape=[1],
                trainable=False,
                regularizer=None,
                name="placeholder",
                initializer=init_w0_fn)
        self._dropout = tf.keras.layers.Dropout(rate=self._rate)
        super(RandomOnOff, self).build(input_shape)

    def call(self, inputs, training):
        return self._dropout(
            self._w0, training=training) * inputs

    def get_config(self):
        return {
            "rate": self._rate
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------

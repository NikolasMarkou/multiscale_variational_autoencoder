import copy

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Union, List


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
            strides: Tuple[int, int] = [1, 1],
            name: str = None,
            **kwargs):
        super(GaussianFilter, self).__init__(
            trainable=False,
            name=name,
            **kwargs)
        if len(kernel_size) != 2:
            raise ValueError("kernel size must be length 2")
        if len(strides) == 2:
            strides = [1] + list(strides) + [1]
        self._kernel_size = kernel_size
        self._strides = strides
        self._sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        self._kernel = None

    def build(self, input_shape):
        from .utilities import depthwise_gaussian_kernel
        self._kernel = \
            depthwise_gaussian_kernel(
                channels=input_shape[-1],
                kernel_size=self._kernel_size,
                nsig=self._sigma).astype("float32")

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
                 name=None,
                 **kwargs):
        super(RandomOnOff, self).__init__(
            trainable=False,
            name=name,
            **kwargs)
        self._rate = rate
        self._dropout = None
        self._noise_shape = None

    def build(self, input_shape):
        noise_shape = [1, ] * len(input_shape)
        noise_shape[0] = input_shape[0]
        self._noise_shape = noise_shape
        self._dropout = (
            tf.keras.layers.Dropout(
                rate=self._rate,
                noise_shape=noise_shape))
        super(RandomOnOff, self).build(input_shape)

    def call(self, inputs, training):
        if training:
            return self._dropout(
                inputs, training=training)
        return inputs

    def get_config(self):
        return {
            "rate": self._rate,
            "noise_shape": self._noise_shape
        }

    def compute_output_shape(self, input_shape):
        return input_shape


# ---------------------------------------------------------------------


class RandomOnOffGradient(tf.keras.layers.Layer):
    """randomly drops the whole connection"""

    def __init__(self,
                 rate: float = 0.5,
                 trainable: bool = False,
                 name=None,
                 **kwargs):
        super(RandomOnOffGradient, self).__init__(
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
        super(RandomOnOffGradient, self).build(input_shape)

    def call(self, inputs, training):
        if training:
            on_off = self._dropout(
                self._w0, training=training)
            return tf.cond(
                on_off > 0.0,
                true_fn=lambda: inputs,
                false_fn=lambda: tf.stop_gradient(inputs))
        return inputs

    def get_config(self):
        return {
            "rate": self._rate
        }

    def compute_output_shape(self, input_shape):
        return input_shape

# ---------------------------------------------------------------------

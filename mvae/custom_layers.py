import numpy as np
import tensorflow as tf

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

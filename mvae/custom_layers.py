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

class GaussianFilter2d(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Tuple[int, int] = [1, 1],
            sigma: Tuple[float, float] = None,
            trainable: bool = False,
            name: str = None,
            **kwargs):
        super(GaussianFilter2d, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        # check kernel size
        if len(kernel_size) != 2:
            raise ValueError("kernel size must be length 2")
        self._kernel_size = kernel_size

        # check strides
        if strides is None:
            strides = [1, 1]
        if len(strides) == 2:
            strides = [1] + list(strides) + [1]
        self._strides = strides

        # check sigma
        if sigma is None:
            # adjust sigma (it's always 1 per pixel)
            sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        self._sigma = sigma

        # add placeholder for kernel
        self._kernel = \
            self.add_weight(
                name="gaussian_kernel",
                shape=(None, None, None, None),
                dtype=tf.float32,
                trainable=trainable)

    def build(self, input_shape):
        from .utilities import depthwise_gaussian_kernel

        if len(input_shape) != 4:
            raise ValueError(f"GaussianFilter2d expects 4 dimensional input shape, "
                             f"got [{input_shape}] instead")

        kernel = \
            depthwise_gaussian_kernel(
                channels=input_shape[-1],
                kernel_size=self._kernel_size,
                nsig=self._sigma).astype("float32")

        self._kernel.assign(kernel)

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


class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if drop_path_rate < 0.0 or drop_path_rate > 1.0:
            raise ValueError("drop_path_rate must be between 0.0 and 1.0")
        self.drop_path_rate = drop_path_rate
        self.dropout = (
            tf.keras.layers.Dropout(
                rate=self.drop_path_rate,
                noise_shape=(1,)))

    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            keep_prob = 1 - self.drop_path_rate
            return self.dropout((x / keep_prob))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


# ---------------------------------------------------------------------


class ConvolutionalSelfAttention(tf.keras.layers.Layer):
    """
    Self attention layer

        implements self-attention block as described in
    Non-local Neural Networks (2018) by Facebook AI research

    A spacetime non-local block. The feature maps are
    shown as the shape of their tensors, e.g., T ×H×W ×1024 for
    1024 channels (proper reshaping is performed when noted). “⊗”
    denotes matrix multiplication, and “⊕” denotes element-wise sum.
    The softmax operation is performed on each row.
    Here we show the embedded Gaussian
    version, with a bottleneck of 512 channels. The vanilla Gaussian
    version can be done by removing θ and φ, and the dot-product
    version can be done by replacing softmax with scaling by 1/N .

    """

    def __init__(self,
                 attention_channels: int,
                 use_scale: bool = True,
                 use_residual: bool = True,
                 attention_activation: str = "linear",
                 output_activation: str = "linear",
                 bn_params: Dict = None,
                 ln_params: Dict = None,
                 **kwargs):
        if attention_channels is None or attention_channels <= 0:
            raise ValueError("attention_channels should be > 0")
        super().__init__(**kwargs)

        conv_params = dict(
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation=attention_activation,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
            kernel_initializer="glorot_normal"
        )
        output_params = dict(
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation=output_activation,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
            kernel_initializer="glorot_normal"
        )
        self.use_scale = use_scale
        self.use_residual = use_residual
        self.params = copy.deepcopy(conv_params)
        self.output_params = copy.deepcopy(output_params)
        self.attention_channels = attention_channels

        query_params = copy.deepcopy(self.params)
        query_params["filters"] = attention_channels
        key_params = copy.deepcopy(self.params)
        key_params["filters"] = attention_channels
        value_params = copy.deepcopy(self.params)
        value_params["filters"] = attention_channels

        self.attention_channels = attention_channels
        self.query_conv = tf.keras.layers.Conv2D(**query_params)
        self.key_conv = tf.keras.layers.Conv2D(**key_params)
        self.value_conv = tf.keras.layers.Conv2D(**value_params)
        self.attention = tf.keras.layers.Attention(use_scale=self.use_scale)
        self.output_conv = None

        if bn_params is not None:
            self.bn = tf.keras.layers.BatchNormalization(**bn_params)
        else:
            self.bn = tf.keras.activations.linear

        if ln_params is not None:
            self.ln = tf.keras.layers.LayerNormalization(**ln_params)
        else:
            self.ln = tf.keras.activations.linear

    def build(self, input_shape):
        output_params = copy.deepcopy(self.output_params)
        output_params["filters"] = input_shape[-1]
        self.output_conv = tf.keras.layers.Conv2D(**output_params)

    def call(self, inputs):
        x = inputs
        # compute query, key, value
        q_x = self.query_conv(x)
        k_x = self.key_conv(x)
        v_x = self.value_conv(x)
        # compute attention
        attention = self.attention([q_x, k_x, v_x])
        attention = self.bn(attention)
        attention = self.ln(attention)
        # compute output conv
        output = self.output_conv(attention)
        # add residual
        return output + x


# ---------------------------------------------------------------------

class SqueezeExcitation(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation Networks (2019)
    https://arxiv.org/abs/1709.01507

    General squeeze and excite block,
    has some differences from keras build-in

    smaller regularization than default

    based on the:
        https://github.com/lvpeiqing/SAR-U-Net-liver-segmentation/blob/master/models/core/modules.py
    """

    def __init__(self, r_ratio: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        if r_ratio <= 0.0 or r_ratio > 1.0:
            raise ValueError(f"reduction [{r_ratio}] must be > 0 and <= 1")
        self.r_ratio = r_ratio
        self.channels = -1
        self.channels_squeeze = -1
        self.conv_0 = None
        self.conv_1 = None
        self.avg_pool = tf.keras.layers.GlobalAvgPool2D(keepdims=True)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.channels_squeeze = max(1, int(round(self.channels * self.r_ratio)))
        self.conv_0 = \
            tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                filters=self.channels_squeeze,
                activation="linear",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
                kernel_initializer="glorot_normal")
        self.conv_1 = \
            tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                filters=self.channels,
                activation="linear",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
                kernel_initializer="glorot_normal")

    def call(self, x, training=None):
        y = x
        y = self.avg_pool(y)
        y = self.conv_0(y)
        # small leak to let the gradient flow
        y = tf.nn.leaky_relu(features=y, alpha=0.1)
        y = self.conv_1(y)
        # learn to turn off
        o = tf.keras.activations.hard_sigmoid(2.5 - tf.nn.relu(y))
        return tf.math.multiply(x, o)


# ---------------------------------------------------------------------

class AttentionGate(tf.keras.layers.Layer):
    """
    Architecture of Attention UNet

    """

    def __init__(self,
                 attention_channels: int,
                 use_bias: bool = True,
                 use_bn: bool = True,
                 use_ln: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention_channels = attention_channels
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.conv_0 = None
        self.bn_0 = None
        self.conv_1 = None
        self.bn_1 = None
        self.conv_o = None
        self.bn_o = None
        self.ln_0 = None
        self.ln_1 = None

    def build(self, input_shapes):
        encoder_feature, upsample_signal = input_shapes
        output_channels = encoder_feature[-1]
        # ---
        self.conv_0 = (
            tf.keras.layers.Conv2D(
                filters=self.attention_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
                kernel_initializer="glorot_normal"))

        if self.use_bn:
            self.bn_0 = tf.keras.layers.BatchNormalization(center=self.use_bias)
        else:
            self.bn_0 = tf.keras.activations.linear

        if self.use_ln:
            self.ln_0 = tf.keras.layers.LayerNormalization(center=self.use_bias)
        else:
            self.ln_0 = tf.keras.activations.linear

        # ---
        self.conv_1 = (
            tf.keras.layers.Conv2D(
                filters=self.attention_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
                kernel_initializer="glorot_normal"))
        if self.use_bn:
            self.bn_1 = tf.keras.layers.BatchNormalization(center=self.use_bias)
        else:
            self.bn_1 = tf.keras.activations.linear

        if self.use_ln:
            self.ln_1 = tf.keras.layers.LayerNormalization(center=self.use_bias)
        else:
            self.ln_1 = tf.keras.activations.linear

        # ---
        self.conv_o = (
            tf.keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=(1, 1),
                activation="linear",
                use_bias=self.use_bias,
                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
                kernel_initializer="glorot_normal"))

    def call(self, inputs, training=None):
        encoder_feature, upsample_signal = inputs
        # ---
        x = self.conv_0(upsample_signal)
        x = self.bn_0(x)
        x = self.ln_0(x)
        # ---
        y = self.conv_1(encoder_feature)
        y = self.bn_1(y)
        y = self.ln_1(y)
        # ---
        o = tf.nn.leaky_relu(x + y, alpha=0.1)
        o = self.conv_o(o)
        # learn to turn off
        o = tf.keras.activations.hard_sigmoid(2.5 - tf.nn.relu(o))
        # ---
        return tf.math.multiply(encoder_feature, o)

# ---------------------------------------------------------------------


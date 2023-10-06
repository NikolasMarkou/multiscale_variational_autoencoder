import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Union, List

# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import conv2d_wrapper


# ---------------------------------------------------------------------

def squeeze_and_excite_block(
        input_layer,
        r_ratio: float = 0.25,
        use_bias: bool = True,
        hard_sigmoid_version: bool = False,
        learn_to_turn_off: bool = False,
        kernel_regularizer: str = "l2",
        kernel_initializer: str = "glorot_normal"):
    """
    Squeeze-and-Excitation Networks (2019)
    https://arxiv.org/abs/1709.01507

    General squeeze and excite block,
    has some differences from keras build-in

    smaller regularization than default
    """
    # --- argument checking
    if r_ratio <= 0.0:
        raise ValueError("r_ratio should be > 0.0")
    channels = tf.keras.backend.int_shape(input_layer)[-1]
    channels_squeeze = max(1, int(round(channels * r_ratio)))

    x = input_layer
    x = tf.keras.layers.GlobalAvgPool2D(keepdims=True)(x)

    x = \
        tf.keras.layers.Conv2D(
            kernel_size=(1, 1),
            filters=channels_squeeze,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            activation="linear")(x)

    # small leak to let the gradient flow
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    if hard_sigmoid_version:
        x = \
            tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                filters=channels,
                use_bias=use_bias,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                activation="linear")(x)
        if learn_to_turn_off:
            # all channels on by default, learns to shut them off
            x = 2.5 - tf.nn.relu(x)
        x = tf.keras.activations.hard_sigmoid(x)
    else:
        # default
        x = \
            tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                filters=channels,
                use_bias=use_bias,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                activation="sigmoid")(x)

    return tf.math.multiply(x, input_layer)


# ---------------------------------------------------------------------


def skip_squeeze_and_excite_block(
        control_layer,
        signal_layer,
        r_ratio: float = 0.25,
        use_bias: bool = False,
        flatten: bool = True,
        hard_sigmoid_version: bool = False,
        learn_to_turn_off: bool = False,
        kernel_regularizer: str = "l2",
        kernel_initializer: str = "glorot_normal",
        bn_params: Dict = None,
        ln_params: Dict = None,
        dropout_params: Dict = None):
    """
    Skip Squeeze-and-Excitation Networks (2019)
    https://arxiv.org/abs/1709.01507

    General squeeze and excite block,
    has some differences from keras build-in

    smaller regularization than default
    """
    # --- argument checking
    if r_ratio <= 0.0:
        raise ValueError("r_ratio should be > 0.0")
    channels = tf.keras.backend.int_shape(signal_layer)[-1]
    channels_squeeze = max(1, int(round(channels * r_ratio)))
    channels_output = tf.keras.backend.int_shape(signal_layer)[-1]

    # ---
    params_1 = dict(
        kernel_size=(1, 1),
        filters=channels_squeeze,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        activation="leaky_relu"
    )

    params_2 = dict(
        kernel_size=(1, 1),
        filters=channels_output,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        activation="linear"
    )

    x = control_layer

    if flatten:
        x = tf.keras.layers.GlobalAvgPool2D(keepdims=True)(x)

    x = \
        conv2d_wrapper(
            input_layer=x,
            bn_post_params=bn_params,
            ln_post_params=ln_params,
            conv_params=params_1)

    if dropout_params is not None:
        x = tf.keras.layers.Dropout(rate=dropout_params["rate"])(x)

    if hard_sigmoid_version:
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=None,
                ln_post_params=None,
                conv_params=params_2)
        if learn_to_turn_off:
            # all channels on by default, learns to shut them off
            x = 2.5 - tf.nn.relu(x)
        x = tf.keras.activations.hard_sigmoid(x)
    else:
        # default
        x = \
            conv2d_wrapper(
                input_layer=x,
                bn_post_params=None,
                ln_post_params=None,
                conv_params=params_2)
        x = tf.keras.activations.sigmoid(x)

    return tf.math.multiply(x, signal_layer)


# ---------------------------------------------------------------------


def self_attention_block(
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None,
        ln_params: Dict = None):
    """
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

    :param input_layer:
    :param conv_params:
    :param bn_params:
    :param ln_params:
    :return:
    """
    # --- argument checking
    channels = conv_params["filters"]
    tp_conv_params = copy.deepcopy(conv_params)
    tp_conv_params["filters"] = channels // 8
    tp_conv_params["activation"] = "linear"
    tp_conv_params["kernel_size"] = (1, 1)
    g_conv_params = copy.deepcopy(tp_conv_params)
    g_conv_params["filters"] = channels // 2

    # --- set network
    x = input_layer

    # --- compute f, g, h
    thi_x = \
        conv2d_wrapper(
            input_layer=x,
            bn_post_params=None,
            conv_params=tp_conv_params)
    phi_x = \
        conv2d_wrapper(
            input_layer=x,
            bn_post_params=None,
            conv_params=tp_conv_params)
    g_x = \
        conv2d_wrapper(
            input_layer=x,
            bn_post_params=None,
            conv_params=g_conv_params)
    g_shape = tf.shape(g_x)
    # reshape (hxw, hxw) ->
    g_x = tf.keras.layers.Reshape(target_shape=(-1, channels // 2))(g_x)
    # thi_x is (h x w, channels)
    thi_x = tf.keras.layers.Reshape(target_shape=(-1, channels // 8))(thi_x)
    # phi_x is (h x w, channels)
    phi_x = tf.keras.layers.Reshape(target_shape=(-1, channels // 8))(phi_x)
    # phi_x is (channels, h x w)
    phi_x = tf.keras.layers.Permute(dims=(2, 1))(phi_x)
    # attention is (h x w, channels) x (channels, h x w) -> (hxw, hxw)
    attention = tf.keras.layers.Dot(axes=(2, 1))([thi_x, phi_x])
    # the softmax operation is performed on each row
    attention = tf.keras.layers.Softmax(axis=-1)(attention)
    # multiply attention map with g_x
    x = tf.keras.layers.Dot(axes=(2, 1))([attention, g_x])
    # bring result to original size
    x = tf.reshape(tensor=x, shape=g_shape)
    # final convolution
    v_x = \
        conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=conv_params)
    return v_x


# ---------------------------------------------------------------------

class RandomOnOff(tf.keras.layers.Layer):
    """
    randomly drops the whole connection during training
    """

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

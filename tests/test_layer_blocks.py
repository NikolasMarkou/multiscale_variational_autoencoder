import mvae
import pytest
import numpy as np

# ==============================================================================


def test_gaussian_filter_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.gaussian_filter_block(x)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 3

# ==============================================================================


def test_resnet_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.resnet_block(x, 32)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 32


# ==============================================================================


def test_attention_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.attention_block(x, 32)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 32

# ==============================================================================


def test_self_attention_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.self_attention_block(x, 32)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 3

# ==============================================================================


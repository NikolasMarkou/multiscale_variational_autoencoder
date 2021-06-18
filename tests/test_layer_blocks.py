import mvae
import pytest
import numpy as np
import keras.backend as K

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


def test_squeeze_excite_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.squeeze_excite_block(x, 32)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 3

# ==============================================================================


def test_mobilenetV2_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.mobilenetV2_block(x, 32)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 3

# ==============================================================================


def test_mobilenetV3_block():
    x = np.zeros((3, 256, 256, 3), dtype=np.float)
    x = mvae.layer_blocks.mobilenetV3_block(x, 32)
    shape = np.shape(x)
    assert shape[0] == 3
    assert shape[1] == 256
    assert shape[2] == 256
    assert shape[3] == 3

# ==============================================================================


def test_laplacian_transform_split():
    input_dims = (18, 32, 32, 3)
    x = np.random.uniform(low=0.0, high=255.0, size=input_dims)
    model = \
        mvae.layer_blocks.laplacian_transform_split(
            input_dims=input_dims[1:],
            levels=3,
            min_value=0.0,
            max_value=255.0)
    results = model(x)
    assert len(results) == 3
    assert K.int_shape(results[0]) == (18, 32, 32, 3)
    assert K.int_shape(results[1]) == (18, 16, 16, 3)
    assert K.int_shape(results[2]) == (18, 8, 8, 3)


# ==============================================================================


def test_laplacian_transform_merge():
    x = [
        np.random.uniform(low=0.0, high=255.0, size=(18, 32, 32, 3)),
        np.random.uniform(low=0.0, high=255.0, size=(18, 16, 16, 3)),
        np.random.uniform(low=0.0, high=255.0, size=(18, 8, 8, 3)),
    ]
    model = \
        mvae.layer_blocks.laplacian_transform_merge(
            input_dims=[
                (32, 32, 3),
                (16, 16, 3),
                (8, 8, 3)
            ],
            levels=3,
            min_value=0.0,
            max_value=255.0)
    results = model(x)
    assert K.int_shape(results) == (18, 32, 32, 3)


# ==============================================================================


def test_laplacian_transform_split_merge():
    input_dims = (18, 32, 32, 3)
    x = np.random.uniform(low=0.0, high=255.0, size=input_dims)
    model_split = \
        mvae.layer_blocks.laplacian_transform_split(
            input_dims=input_dims[1:],
            levels=3,
            min_value=0.0,
            max_value=255.0)
    model_merge = \
        mvae.layer_blocks.laplacian_transform_merge(
            input_dims=[
                (32, 32, 3),
                (16, 16, 3),
                (8, 8, 3)
            ],
            levels=3,
            min_value=0.0,
            max_value=255.0)
    results_split = model_split(x)
    results_merge = model_merge(results_split)
    assert K.int_shape(results_merge) == (18, 32, 32, 3)

# ==============================================================================


import numpy as np
from numpy.lib.stride_tricks import as_strided


def pool2d(
        A,
        kernel_size=2,
        stride=2,
        padding=0,
        pool_mode='max'):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                        strides=(stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def pool3d(
        A,
        kernel_size=2,
        stride=2,
        padding=0,
        pool_mode="max"):
    no_channels = A.shape[2]
    channels = [
        pool2d(A[:, :, i], kernel_size=kernel_size, stride=stride, padding=padding, pool_mode=pool_mode)
        for i in range(no_channels)
    ]
    dim0 = channels[0].shape[0]
    dim1 = channels[0].shape[1]
    return np.ndarray(
        shape=(dim0, dim1, no_channels),
        buffer=np.array(channels),
        dtype=A.dtype)


def pool4d(
        A,
        kernel_size=2,
        stride=2,
        padding=0,
        pool_mode="max"):
    no_images = A.shape[0]
    return np.array([
        pool3d(A[i, :, :, :], kernel_size=kernel_size, stride=stride, padding=padding, pool_mode=pool_mode)
        for i in range(no_images)
    ])

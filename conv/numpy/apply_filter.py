import numpy as np


def apply_filter(a_slice_prev, W, b):
    """
    Applies a filter to an image slice of the same dimensions.

    :param a_slice_prev: A slice of activations of the previous layer. R=(f, f, n_C_prev).
    :param W: Convolution filter parameters. R=(f, f, n_C_prev).
    :param b: Convolution filter bias. R=(1, 1, 1).
    :return: A convolution activation. Integer
    """
    product = np.multiply(a_slice_prev, W)
    z = np.sum(product) + np.float(b)
    return z
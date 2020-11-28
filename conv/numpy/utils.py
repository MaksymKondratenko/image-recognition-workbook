import numpy as np


def calc_conv_img_res(prev_res, f, pad, stride):
    return int((prev_res - f + 2 * pad)/stride) + 1


def calc_pool_img_res(prev_res, f, stride):
    return int((prev_res - f)/stride) + 1


def pad(X, pad_amount):
    """
    Adds a frame to a picture to return to original size after a convolution
    :param X: input images, R=(m, n_H, n_W, n_C)
    :param pad_amount: pixels to add from each side, a frame width. Integer.
    :return: padded images, R=(m, n_H, n_W, n_C)
    """
    return np.pad(X, ((0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount ), (0, 0)), mode='constant', constant_values=(0, 0))


def create_max_pool_mask(x):
    return x == np.max(x)


def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    return np.ones(shape) * average

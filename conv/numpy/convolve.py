import numpy as np

from conv.numpy.utils import calc_conv_img_res, pad


def convolve(A_prev, W, b, hyper_params):
    """
    Conducts a forward convolution of one layer.

    :param A_prev: Activations of the previous layer. R=(m, n_H_prev, n_W_prev, n_C_prev).
    :param W: Convolution filter parameters. R=(f, f, n_C_prev, n_C).
    :param b: Convolution filter bias. R=(1, 1, 1, n_C).
    :param hyper_params: a dictionary of hyper parameters of the network implementation.
    :return:
    Z: Activations of the current layer. R=(m, n_H, n_W, n_C).
    cache: dictionary of a cached values.
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (_, f, n_C) = np.shape(W)

    f = hyper_params['f']
    pad_amount = hyper_params['pad']
    stride = hyper_params['stride']

    n_H = calc_conv_img_res(n_H_prev, f, pad_amount, stride)
    n_W = calc_conv_img_res(n_W_prev, f, pad_amount, stride)

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_padded = pad(A_prev, pad_amount)

    for i in range(m):
        a_prev = A_prev_padded[i]

        for h in range(n_H):
            h_start = h / stride
            h_end = h_start + f

            for w in range(n_W):
                w_start = w / stride
                w_end = w_start + f

                for c in range(n_C):
                    a_slice_prev = a_prev[h_start:h_end, w_start:w_end, c]
                    weights = np.multiply(a_slice_prev, W[:, :, :, c])
                    bias = b[:, :, :, c]
                    Z[i, h, w, c] = weights + np.float(bias)

    cache = (A_prev, W, b, hyper_params)

    return Z, cache

import numpy as np

from conv.numpy.utils import pad


def convolve_backwards(dZ, cache):
    """
    Conducts backwards propagation through the convolution layer.

    :param dZ: Gradient of the cost of the current layer. R=(m, n_H, n_W, n_C).
    :param cache: dictionary contains useful cached values.
    :return:
    dA_prev: Cost gradient of the previous layer. R=(m, n_H_prev, n_W_prev, n_C_prev).
    dW: Cost gradient of filters' params from the previous layer. R=(f, f, n_C_prev, n_C).
    db: Cost gradient of filters' biases from the previous layer. R=(1, 1, 1, n_C).
    """

    (A_prev, W, b, hyper_params) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, _, _, _) = np.shape(W)
    (_, n_H, n_W, n_C) = np.shape(dZ)
    pad_amount = hyper_params['pad']
    stride = hyper_params['stride']

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_padded = pad(A_prev, pad_amount)
    dA_prev_padded = pad(dA_prev, pad_amount)

    for i in range(m):
        a_prev_i = A_prev_padded[i]
        da_prev_i = dA_prev_padded[i]

        for h in range(n_H):
            h_start = h / stride
            h_end = h_start + f

            for w in range(n_W):
                w_start = w / stride
                w_end = w_start + f

                for c in range(n_C):
                    a_slice = a_prev_i[h_start:h_end, w_start:w_end, c]
                    dZcurr = dZ[i, w, h, c]
                    da_prev_i[h_start:h_end, w_start:w_end, c] += W[:, :, :, c] * dZcurr
                    dW[:, :, :, c] += a_slice * dZcurr
                    db[:, :, :, c] += dZcurr

        dA_prev = da_prev_i[pad_amount:-pad_amount, pad_amount:-pad_amount, :]

    return dA_prev, dW, db


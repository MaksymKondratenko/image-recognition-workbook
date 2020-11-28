import numpy as np

from conv.numpy.utils import create_max_pool_mask, distribute_value


def pool_backwards(dA, cache, mode='max'):
    """
    Conduct backwards propagation through a pooling layer.

    :param dA: Cost gradient of a pooling layer output. R=(m, n_H, n_W, n_C).
    :param cache: Dictionary with useful cached values.
    :param mode: A way in which pooling was done for the the layer.
    :return: dA_prev: Cost gradient of a pooling of the prev layer.
    """

    (A_prev, hyper_params) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (_, n_H, n_W, n_C) = np.shape(dA)
    f = hyper_params['f']
    stride = hyper_params['stride']

    dA_prev = np.zeros(np.shape(A_prev))

    for i in range(m):

        for h in range(n_H):
            h_start = h / stride
            h_end = h_start + f

            for w in range(n_W):
                w_start = w / stride
                w_end = w_start + f

                for c in range(n_C):
                    da_curr = dA[i, h, w, c]

                    if mode == 'max':
                        a_prev_slice = A_prev[h_start:h_end, w_start:w_end, c]
                        mask = create_max_pool_mask(a_prev_slice)
                        dA_prev[h_start:h_end, w_start:w_end, c] += np.multiply(mask, da_curr)
                    elif mode == 'average':
                        da = da_curr
                        shape = (f, f)
                        dA_prev[h_start:h_end, w_start:w_end, c] += distribute_value(da, shape)

    return dA_prev
import numpy as np

from conv.numpy.utils import calc_pool_img_res


def pool(A_prev, hyper_params, mode = 'max'):
    """
    Conducts image pooling to reduce its measurements.

    :param A_prev: Activations of the previous layer. R=(m, n_H_prev, n_W_prev, n_C_prev).
    :param hyper_params: A dictionary of the hyper-parameters of the network.
    :param mode: A way of pooling one of ['max', 'average']
    :return:
    A: activation of a current layer after pooling.
    cache: a dictionary of useful cached values.
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    f = hyper_params['f']
    stride = hyper_params['stride']

    n_H = calc_pool_img_res(n_H_prev, f, stride)
    n_W = calc_pool_img_res(n_W_prev, f, stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):
            h_start = h / stride
            h_end = h_start + f

            for w in range(n_W):
                w_start = w / stride
                w_end = w_start + f

                for c in range(n_C):
                    a_prev_slice = a_prev[h_start:h_end, w_start:w_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.average(a_prev_slice)

    cache = (A_prev, hyper_params)

    return A, cache
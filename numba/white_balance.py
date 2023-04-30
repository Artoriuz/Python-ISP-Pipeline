import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def white_balance(input):
    output = np.zeros((input.shape[0], input.shape[1], 3), np.float64)

    for idz in prange (0, 3):
        channel_mean = np.mean(input[:, :, idz])
        output[:, :, idz] = input[:, :, idz] * (0.5 / channel_mean)

    global_scale = np.mean(input) / np.mean(output)
    output = output * global_scale

    return output

@njit(parallel=True, fastmath=True)
def white_balance_gimp(input):
    output = np.zeros((input.shape[0], input.shape[1], 3), np.float64)
    
    for idz in prange (0, 3):
        # white balance for every channel independently
        mi = np.percentile(input[:, :, idz], 0.05)
        ma = np.percentile(input[:, :, idz], 99.95)
        output[:, :, idz] = (input[:, :, idz] - mi) / (ma - mi)
        output[:, :, idz] = np.clip(output[:, :, idz], 0, 1)

    return output

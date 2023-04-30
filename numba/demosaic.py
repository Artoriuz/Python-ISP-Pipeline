import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def demosaic(input):
    output = np.zeros((input.shape[0] + 4, input.shape[1] + 4, 3), np.float64)
    temp = np.zeros((input.shape[0] + 4, input.shape[1] + 4), np.float64)
    temp[2:-2, 2:-2] = input
    temp[0,:] = temp[1,:] = temp[2,:]
    temp[:,0] = temp[:,1] = temp[:,2]
    temp[-1,:] = temp[-2,:] = temp[-3,:]
    temp[:,-1] = temp[:,-2] = temp[:,-3]

    #GBGB
    #RGRG
    # Pixel mapping
    for idy in range (1, temp.shape[0] - 1):
        for idx in range (1, temp.shape[1] - 1):
            if ((idy % 2 == 0) and (idx % 2 == 0)): # Green pixel
                output[idy, idx, 0] = (temp[idy, idx - 1] + temp[idy, idx + 1]) / 2.0
                output[idy, idx, 1] = temp[idy, idx]
                output[idy, idx, 2] = (temp[idy - 1, idx] + temp[idy + 1, idx]) / 2.0
            elif ((idy % 2 == 0) and (idx % 2 != 0)): # Blue pixel
                output[idy, idx, 0] = temp[idy, idx]
                output[idy, idx, 1] = (temp[idy - 1, idx] + temp[idy + 1, idx] + temp[idy, idx - 1] + temp[idy, idx + 1]) / 4.0
                output[idy, idx, 2] = (temp[idy - 1, idx - 1] + temp[idy + 1, idx + 1] + temp[idy + 1, idx - 1] + temp[idy - 1, idx + 1]) / 4.0
            elif ((idy % 2 != 0) and (idx % 2 == 0)): # Red pixel
                output[idy, idx, 0] = (temp[idy - 1, idx - 1] + temp[idy + 1, idx + 1] + temp[idy + 1, idx - 1] + temp[idy - 1, idx + 1]) / 4.0
                output[idy, idx, 1] = (temp[idy - 1, idx] + temp[idy + 1, idx] + temp[idy, idx - 1] + temp[idy, idx + 1]) / 4.0
                output[idy, idx, 2] = temp[idy, idx]
            elif ((idy % 2) != 0 and (idx % 2 != 0)): # Green pixel
                output[idy, idx, 0] = (temp[idy - 1, idx] + temp[idy + 1, idx]) / 2.0
                output[idy, idx, 1] = temp[idy, idx]
                output[idy, idx, 2] = (temp[idy, idx - 1] + temp[idy, idx + 1]) / 2.0

    return output[2:-2, 2:-2, :]

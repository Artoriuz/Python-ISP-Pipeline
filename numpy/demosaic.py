import numpy as np
from scipy import signal

def demosaic(input):
    output = np.zeros([input.shape[0], input.shape[1], 3], np.float64)

    #GBGB
    #RGRG
    # Pixel mapping
    for idy in range (0, input.shape[0]):
        for idx in range (0, input.shape[1]):
            if ((idy % 2 == 0) and (idx % 2 == 0)): # Green pixel
                output[idy, idx, 1] = input[idy, idx]
            elif ((idy % 2 == 0) and (idx % 2 != 0)): # Blue pixel
                output[idy, idx, 0] = input[idy, idx]
            elif ((idy % 2 != 0) and (idx % 2 == 0)): # Red pixel
                output[idy, idx, 2] = input[idy, idx]
            elif ((idy % 2) != 0 and (idx % 2 != 0)): # Green pixel
                output[idy, idx, 1] = input[idy, idx]

    # Separable 2D convolution with 1D FIR filters
    lin_filter = np.array([0.5, 1.0, 0.5])

    for idy in range (0, input.shape[0]):
        output[idy, :, 0] = np.convolve(output[idy, :, 0], lin_filter, 'same')
        output[idy, :, 1] = np.convolve(output[idy, :, 1], lin_filter, 'same')
        output[idy, :, 2] = np.convolve(output[idy, :, 2], lin_filter, 'same')

    for idx in range (0, input.shape[1]):
        output[:, idx, 0] = np.convolve(output[:, idx, 0], lin_filter, 'same')
        output[:, idx, 1] = np.convolve(output[:, idx, 1], lin_filter / 2.0, 'same')
        output[:, idx, 2] = np.convolve(output[:, idx, 2], lin_filter, 'same')

    return output

def demosaic_fast(input):
    
    green_mask = np.array([[1, 0], [0, 1]])
    blue_mask = np.array([[0, 1], [0, 0]])
    red_mask = np.array([[0, 0], [1, 0]])

    green_repeat = np.tile(green_mask, (1536, 2040))
    blue_repeat = np.tile(blue_mask, (1536, 2040))
    red_repeat = np.tile(red_mask, (1536, 2040))

    input_g = input * green_repeat
    input_b = input * blue_repeat
    input_r = input * red_repeat

    # Separable 2D convolution with 1D FIR filters
    lin_filter_2d = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]])
    lin_filter_2db = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    
    output_g = np.expand_dims(signal.convolve(input_g, lin_filter_2d, mode='same'), axis = -1)
    output_b = np.expand_dims(signal.convolve(input_b, lin_filter_2db, mode='same'), axis = -1)
    output_r = np.expand_dims(signal.convolve(input_r, lin_filter_2db, mode='same'), axis = -1)

    output_uncapped = np.concatenate((output_b, output_g, output_r), axis = -1)
    output = np.clip(output_uncapped, 0, 1)

    return output

import numpy as np
import cv2

def white_balance(input):
    output = np.zeros([input.shape[0], input.shape[1], 3], np.float64)
    
    for idz in range (0, 3):
        channel_mean = np.mean(input[:, :, idz])
        output[:, :, idz] = input[:, :, idz] * (0.5 / channel_mean)
    
    global_scale = np.mean(input) / np.mean(output)
    output = output * global_scale
    return output

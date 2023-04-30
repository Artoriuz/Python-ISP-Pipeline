import numpy as np

def gamma_correction(input, power, gain):
    output = np.power(input, power) * gain
    output = np.clip(output, 0.0, 1.0) # limits the output to [0, 1]
    return output

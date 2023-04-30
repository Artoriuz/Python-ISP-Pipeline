import jax.numpy as jnp
from jax import jit
    
@jit
def black_level_correction_jax(input, lux):
    black_levels = jnp.array([0.00097831, 0.00097821, 0.0009782, 0.00097831, 0.00097839, 0.00097839, 0.00097829, 0.00097829, 0.00097829, 0.00097827]) # THIS NEEDS TO BE COMPUTED BY LINEARISATION_CALIBRATION
    lux_levels = jnp.array([5, 10, 50, 125, 250, 500, 1000, 1500, 2000, 2500]) # THIS NEEDS TO BE RETRIEVED WHEN TAKING THE CALIBRATION SHOTS

    ceil_idx = jnp.searchsorted(lux_levels, lux) # maps AEC gain into an "index"
    floor_idx = ceil_idx - 1
    interpolation_ratio = (lux - lux_levels[floor_idx]) / (lux_levels[ceil_idx] - lux_levels[floor_idx])

    black_level = black_levels[floor_idx] + interpolation_ratio * (black_levels[ceil_idx] - black_levels[floor_idx])
    # performs a linear interpolation of the black levels obtained in the calibration step to estimate current black level

    output = (input - black_level) * (1 / (1 - black_level)) # corrects the black level and ensure pixels can still use all quantisation levels
    output = jnp.clip(output, 0.0, 1.0) # limits the output to [0, 1]
    
    return output

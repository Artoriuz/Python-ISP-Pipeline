import jax.numpy as jnp
from jax import jit
from scipy import misc
import jax.scipy as jsp

@jit
def demosaic_jax(input):
    
    green_mask = jnp.array([[1, 0], [0, 1]])
    blue_mask = jnp.array([[0, 1], [0, 0]])
    red_mask = jnp.array([[0, 0], [1, 0]])

    green_repeat = jnp.tile(green_mask, input.shape)
    blue_repeat = jnp.tile(blue_mask, input.shape)
    red_repeat = jnp.tile(red_mask, input.shape)

    input_g = input * green_repeat
    input_b = input * blue_repeat
    input_r = input * red_repeat

    #input_bgr = jnp.concatenate((input_b, input_g, input_r), axis = 2)

    # Separable 2D convolution with 1D FIR filters
    lin_filter_2d = jnp.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]])
    lin_filter_2db = jnp.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    
    output_g = jnp.expand_dims(jsp.signal.convolve(input_g, lin_filter_2d, mode='same'), axis = -1)
    output_b = jnp.expand_dims(jsp.signal.convolve(input_b, lin_filter_2db, mode='same'), axis = -1)
    output_r = jnp.expand_dims(jsp.signal.convolve(input_r, lin_filter_2db, mode='same'), axis = -1)

    output_uncapped = jnp.concatenate((output_b, output_g, output_r), axis = -1)
    output = jnp.clip(output_uncapped, 0, 1)

    return output

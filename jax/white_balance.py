import jax.numpy as jnp
from jax import jit

@jit
def white_balance_jax(input):

    b_mean = jnp.mean(input[:, :, 0])
    g_mean = jnp.mean(input[:, :, 1])
    r_mean = jnp.mean(input[:, :, 2])

    output_b = jnp.expand_dims(input[:, :, 0] * (0.5 / b_mean), axis = -1)
    output_g = jnp.expand_dims(input[:, :, 1] * (0.5 / g_mean), axis = -1)
    output_r = jnp.expand_dims(input[:, :, 2] * (0.5 / r_mean), axis = -1)

    temp = jnp.concatenate((output_b, output_g, output_r), axis = -1)
    
    global_scale = jnp.mean(input) / jnp.mean(temp)
    output = temp * global_scale

    return output
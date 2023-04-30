import jax.numpy as jnp
from jax import jit

@jit
def gamma_correction_jax(input, power, gain):
    output = jnp.power(input, power) * gain
    output = jnp.clip(output, 0.0, 1.0) # limits the output to [0, 1]
    
    return output
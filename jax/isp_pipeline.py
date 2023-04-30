import jax.numpy as jnp
import numpy as np
import argparse
import rawpy
import cv2
import time
from black_level_correction import *
from gamma_correction import *
from demosaic import *
from white_balance import *

parser = argparse.ArgumentParser(description='Linearisation Block')
parser.add_argument("-i", "--input", dest="filename", type=str, required=True, help="enter input file")
parser.add_argument("-l", "--lux", dest="lux", type=float, required=True, help="enter lux level")
args = parser.parse_args()

input = rawpy.imread(args.filename).raw_image
input = input.astype(jnp.float64) / (jnp.power(2, 10) - 1)

start = time.time()
for _ in range(10):
    blc_out = black_level_correction_jax(input, args.lux)
    dem_out = demosaic_jax(blc_out)
    gam_out = gamma_correction_jax(dem_out, 1.0 / 2.2, 1.0)
    wba_out = white_balance_jax(gam_out)
end = time.time()
print(f"JAX took {end - start} seconds to complete 10 runs")

# blc_out = black_level_correction(input, args.lux)
# dem_out = demosaic2(blc_out)
# gam_out = gamma_correction(dem_out, 1.0 / 2.2, 1.0)
# wba_out = white_balance(gam_out)

output = np.array(wba_out)
cv2.imwrite('./output.png', np.around(output * (np.power(2, 16) - 1)).astype(np.uint16)) # converts the output to uint16 and saves the result

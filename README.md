# Python-ISP-Pipeline
Basic ISP pipeline written in Python. Numba and JAX variants included for increased performance when applicable.

The pipeline consists of a few basic image processing steps that are all implements in their respective files.

For black level correction, you first need to run black level calibration with black photos (sensor covered by a dark object in a dark room) taken at different AEC gains. This will give you how much of the signal is actually being created by thermal noise.

All other steps are pretty much self explanatory, and to run the entire pipeline all you need is a GBRG raw input with 10 bits per pixel. Colour depth can be changed manually in the code if that's necessary.

RGB output is saved as a 16 bits PNG.

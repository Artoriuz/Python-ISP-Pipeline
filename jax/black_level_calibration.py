import numpy as np
import glob
import rawpy

# Load data into memory
filelist = sorted(glob.glob('./inputs/*.dng')) # creates a file list of all images under input/
print(f"Reading images:\n\n{filelist}\n")
input_imgs = [] # creates an empty list
for myFile in filelist: # loops through all files
    input = rawpy.imread(myFile).raw_image # reads the image and loads it as a numpy array
    input_imgs.append(input) # appends the array into the list
input_imgs = np.array(input_imgs).astype(np.float64) / np.power(2, 16) # turns the list into a single float64 multidimensional array

black_levels = np.mean(input_imgs, axis=(1,2)) # computes the average pixel level for each image separately

print(f"Computed black levels:\n\n {black_levels}") # simply prints the results

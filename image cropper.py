"""
This file is for cropping medical images around the gross tumour volume(GTV). All the voxels 
of the input image will need to be of the same size, 1x1x1. The result should be a cube crop 
around the largest GTV in the dataset with a small window around the edge.
"""
import nibabel as nib
import numpy as np
import os
niftypath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_NIFTY"

def CoM_tumour(mask):
    #function to find the CoM of the tumour
    print(mask.shape[2])
    array = np.argwhere(mask)
    x_coord = np.average(array[:,0])
    y_coord = np.average(array[:,1])
    z_coord = np.average(array[:,2])
    return(x_coord, y_coord, z_coord)
def largest_gtv_finder(mask):
    #finds the difference between lowest index anf highest index of x, y and z and returns.#
    #print(mask[:,:,0])
    diff_x, diff_y, diff_z = 0
    return(diff_x, diff_y, diff_z)#return furthest distance above and below in x y and z
def cropping():
    #cropping function, gets passed an mask and image and returns cropped versions.
    print()


for file in os.listdir(niftypath):
    if "-RTSTRUCT" in file:
        mask = nib.load(os.path.join(niftypath, file)).get_data()
        temp_x, temp_y, temp_z = CoM_tumour(mask)
        print(temp_x, temp_y, temp_z)

    else:
        continue
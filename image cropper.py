"""
This file is for cropping medical images around the gross tumour volume(GTV). All the voxels 
of the input image will need to be of the same size, 1x1x1. The result should be a cube crop 
around the largest GTV in the dataset with a small window around the edge.
"""
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
niftypath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_NIFTY"
outputpath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_CROPPED"

def CoM_tumour(mask):
    #function to find the CoM of the tumour
    array = np.argwhere(mask)
    x_coord = np.round(np.average(array[:,0]))
    y_coord = np.round(np.average(array[:,1]))
    z_coord = np.round(np.average(array[:,2]))
    return(x_coord, y_coord, z_coord)
def largest_gtv_finder(mask, CoMs):
    #finds the difference between lowest index anf highest index of x, y and z and returns.#
    #print(mask[:,:,0])
    bignums = []
    positions = np.argwhere(mask)
    CoM = CoMs.pop()
    bignums.append(np.abs(np.max(positions[:, 0]) - CoM[0]))
    bignums.append(np.abs(np.min(positions[:, 0]) - CoM[0]))
    bignums.append(np.abs(np.max(positions[:, 1]) - CoM[1]))
    bignums.append(np.abs(np.min(positions[:, 1]) - CoM[1]))
    bignums.append(np.abs(np.max(positions[:, 2]) - CoM[2]))
    bignums.append(np.abs(np.min(positions[:, 2]) - CoM[2]))
    largest_dist = np.max(bignums)
    CoMs.append(CoM)
    return(largest_dist)#return furthest distance above and below in x y and z
def cropping(array, CoM_array, cropping_size):
    #cropping function, gets passed a mask or an image and returns cropped versions.
    xstart = CoM_array[0]-cropping_size
    xend = CoM_array[0]+cropping_size
    ystart = CoM_array[1]-cropping_size
    yend = CoM_array[1]+cropping_size
    zstart = CoM_array[2]-cropping_size
    zend = CoM_array[2]+cropping_size
    xstart = xstart.astype(int)
    xend = xend.astype(int)
    ystart = xstart.astype(int)
    yend = xend.astype(int)
    zstart = xstart.astype(int)
    zend = xend.astype(int)
    #print(xstart, xend, ystart, yend, zstart, zend)
    cropped_array = array[xstart:xend, ystart:yend, zstart:zend]
    return(cropped_array)

CoMs = []
largest_tumour_axis = 0
counter = -1
for file in os.listdir(niftypath):
    if "-RTSTRUCT" in file:
        #loops over all files looking for masks. Finds CoM of tumour from mask
        #finds largest axis in that tumour
        #finds largest overall distance from CoM to edge of a tumour, this defines the size of out crop.
        mask = nib.load(os.path.join(niftypath, file)).get_data()
        
        CoMs.append(CoM_tumour(mask))
        #print(comx, comy, comz)
        temp_largest = largest_gtv_finder(mask, CoMs)
        #print(temp_largest)
        if temp_largest > largest_tumour_axis:
            largest_tumour_axis = temp_largest
    else:
        continue
cropping_size = largest_tumour_axis + 15
for file in os.listdir(niftypath):
    counter+=0.5
    index = np.floor(counter)
    index = index.astype(int)
    print(index)
    CoM_index = CoMs[index]
    image = nib.load(os.path.join(niftypath, file)).get_data()
    cropped_array = cropping(image, CoM_index, cropping_size)
    print(cropped_array.shape)
    cropped_image = sitk.GetImageFromArray(cropped_array)
    # cropped_image.SetDirection()
    # cropped_image.SetOrigin()
    sitk.WriteImage(cropped_image, f"{outputpath}/{file}.nii")
    
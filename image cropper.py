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
    ystart = ystart.astype(int)
    yend = yend.astype(int)
    zstart = zstart.astype(int)
    zend = zend.astype(int)
    #print(xstart, xend, ystart, yend, zstart, zend)
    #print(f"array:{np.argwhere(array)}")
    cropped_array = array[xstart:xend, ystart:yend, zstart:zend]
    print(cropped_array.shape)
    return(cropped_array)

CoMs = []
largest_tumour_axis = 0
temp_largest = 0
print("========================PROGRAM STARTING========================")
for file in os.listdir(niftypath):
    print(file)
    if "-RTSTRUCT" in file:
        #loops over all files looking for masks. Finds CoM of tumour from mask
        #finds largest axis in that tumour
        #finds largest overall distance from CoM to edge of a tumour, this defines the size of out crop.
        mask = sitk.ReadImage(os.path.join(niftypath, file))
        mask_array = sitk.GetArrayFromImage(mask)
        CoM_temp = CoM_tumour(mask_array)
        print(CoM_temp)
        CoMs.append(CoM_temp)
        #print(comx, comy, comz)
        temp_largest = largest_gtv_finder(mask_array, CoMs)
        print(f"temp:{temp_largest}")
        #print(temp_largest)
        if temp_largest > largest_tumour_axis:
            largest_tumour_axis = temp_largest
    else:
        continue
cropping_size = largest_tumour_axis + 15
print(f"cropping size:{cropping_size}")
counter = -0.5
print(CoMs)
for file in os.listdir(niftypath):
    print(file)
    counter+=0.5
    index = np.floor(counter)
    index = index.astype(int)
    print(index)
    CoM_index = CoMs[index]
    print(f"CoM:{CoM_index}")

    image = sitk.ReadImage(os.path.join(niftypath, file))
    array = sitk.GetArrayFromImage(image)
    print(array.shape)
    print(cropping_size)
    cropped_array = cropping(array, CoM_index, cropping_size)
    print(cropped_array.shape)

    cropped_image = sitk.GetImageFromArray(cropped_array)
    cropped_image.SetDirection(image.GetDirection())
    cropped_image.SetOrigin(image.GetOrigin())
    sitk.WriteImage(cropped_image, f"{outputpath}/{file}.nii")
    
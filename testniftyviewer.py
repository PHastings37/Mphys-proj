"""
This file was just for a test, see image_cropper for final program
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
    CoM = CoMs
    bignums.append(np.abs(np.max(positions[:, 0]) - CoM[0]))
    bignums.append(np.abs(np.min(positions[:, 0]) - CoM[0]))
    bignums.append(np.abs(np.max(positions[:, 1]) - CoM[1]))
    bignums.append(np.abs(np.min(positions[:, 1]) - CoM[1]))
    bignums.append(np.abs(np.max(positions[:, 2]) - CoM[2]))
    bignums.append(np.abs(np.min(positions[:, 2]) - CoM[2]))
    largest_dist = np.max(bignums)
    #CoMs.append(CoM)
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
    return(cropped_array)

def permute_axes(volume) :
   #This function permutes the axes of the input volume.
    
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder([1,2,0])

    return permute.Execute(volume)
   

CT_image = sitk.ReadImage(f"{niftypath}/LUNG1-001-CT.nii")
mask_image = sitk.ReadImage(f"{niftypath}/LUNG1-001-RTSTRUCT.nii")

mask_array = sitk.GetArrayFromImage(mask_image)
CT_array = sitk.GetArrayFromImage(CT_image)

print(CT_array.shape)
print(mask_array.shape)

CoM = CoM_tumour(mask_array)
largest_dist = largest_gtv_finder(mask_array, CoM)

print(f"CoM:{CoM}")
print(f"dist:{largest_dist}")
cropping_size = largest_dist + 15

cropped_mask = cropping(mask_array, CoM, cropping_size)
cropped_CT = cropping(CT_array, CoM, cropping_size)
#print(f"cropped mask:{np.argwhere(cropped_mask)}")
#print(f"mask:{np.argwhere(mask_array)}")

cropped_mask = sitk.GetImageFromArray(cropped_mask)
#cropped_mask = permute_axes(cropped_mask)
cropped_mask.SetDirection(CT_image.GetDirection())
cropped_mask.SetOrigin(CT_image.GetOrigin())
sitk.WriteImage(cropped_mask, "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_CROPPED/masktest.nii")

cropped_CT = sitk.GetImageFromArray(cropped_CT)
#cropped_CT = permute_axes(cropped_CT)
cropped_CT.SetDirection(CT_image.GetDirection())
cropped_mask.SetOrigin(CT_image.GetOrigin())
print("writeCT")
sitk.WriteImage(cropped_CT, "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_CROPPED/CTtest.nii")
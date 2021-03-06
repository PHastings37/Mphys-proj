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

def cropping(array, CoM_array, cropping_size, file):
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
    
    coords = []
    sub_zero = True
    coords.extend([xstart, xend, ystart, yend, zstart, zend])#appends all starts and ends to a list

    #determines which direction to pad in event of out of bound index
    if np.any(coords<0) ==True:
        sub_zero = True
        coords = [0 if i < 0 else i for i in coords]#this line prevents the cropped from trying to -ve index an array
    elif np.any(coords>511) ==True:
        sub_zero = False
        coords = [0 if i > 511 else i for i in coords]#this line prevents the cropped from trying to -ve index an array
    
    
    cropped_array = array[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]
    if cropped_array.shape != (cropping_size*2,cropping_size*2,cropping_size*2) and sub_zero == True:
        print(f"file {file} is being padded")
        pad_width = []
        pad_width = cropped_array.shape - 2*cropping_size
        pad_width = np.abs(pad_width)
        pad_width=pad_width.astype(int)
        print(f"pad width is {pad_width}")

        if "-CT" in file:
            #background for a CT image is -1024, not 0 which is for masks.
            cropped_array = np.pad(cropped_array, ((pad_width[0], 0), (pad_width[1], 0), (pad_width[2], 0)), mode="constant", constant_values = [(-1024,-1024), (-1024,-1024), (-1024,-1024)])
        else:
            cropped_array = np.pad(cropped_array, ((pad_width[0], 0), (pad_width[1], 0), (pad_width[2], 0)), mode="constant")
    elif cropped_array.shape != (cropping_size*2,cropping_size*2,cropping_size*2) and sub_zero == False:
        print(f"c-array shape: {cropped_array.shape}")
    print(cropped_array)
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
for file in os.listdir(niftypath):
    #runs over all files and crops/pads them to the correct size
    print(file)
    counter+=0.5#patient 1 has an rt and a ct so need to do two files for every iteration
    index = np.floor(counter)
    index = index.astype(int)
    CoM_index = CoMs[index]
    print(f"CoM:{CoM_index}")

    image = sitk.ReadImage(os.path.join(niftypath, file))
    array = sitk.GetArrayFromImage(image)

    cropped_array = cropping(array, CoM_index, cropping_size, file)

    cropped_image = sitk.GetImageFromArray(cropped_array)
    cropped_image.SetDirection(image.GetDirection())
    cropped_image.SetSpacing(image.GetSpacing())
    cropped_image.SetOrigin(image.GetOrigin())
    sitk.WriteImage(cropped_image, f"{outputpath}/{file}.nii")
    
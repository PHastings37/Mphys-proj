"""
an updated resampler that can handle masks, takes in a mask or dicom structure and returns a resampled image of 512x512x256 with spacing of 113 OR however 
you want to define it. 

adapted from Rorys code MASK_resample.py https://github.com/rfarwell/MPhys

"""

import rt_utils
import sys
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy
from scipy import stats
import os

#Global variables
reader = sitk.ImageSeriesReader()
filepath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED"
outputpath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_NIFTY"
Output_Spacing = [0.9765625, 0.9765625, 3.0] 

#a change2
def resample_volume(volume, interpolator, def_pix_val):
    #function to resample an image passed in, values passed in depend on CT or rtstruct being passed.

    new_size = [512, 512, 134] #dimensions chosen based on images in data set, all images being 512x512 then 512 being the next power of 2 after largest file
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(Output_Spacing)
    resample.SetDefaultPixelValue(def_pix_val)
    return resample.Execute(volume)

def permute_axes(volume) :
   #This function permutes the axes of the input volume.
    
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder([1,2,0])

    return permute.Execute(volume)



#iterate over all patients for rt and ct and save a nii of both.
for filename in os.listdir(filepath):
    print(filename)
    if "-SEG" in filename:
        #dont want to look at seg files here
        continue
    
    
    if "-CT" in filename:
        
        dcm_paths = reader.GetGDCMSeriesFileNames(os.path.join(filepath, filename))
        reader.SetFileNames(dcm_paths)
        image = sitk.ReadImage(dcm_paths)
        dicom = resample_volume(image, sitk.sitkLinear, -1024)
        #dicom = dicom.astype(float)
        sitk.WriteImage(dicom, f"{os.path.join(outputpath, filename)}.nii")

    elif "-RT" in filename:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path="/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-001-CT", 
            rt_struct_path="/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-001-RTSTRUCT/3-2.dcm"
        )           

        # Getting arrays for all the masks for the determined ROIs
        mask_3d_Lung_Right = rtstruct.get_roi_mask_by_name("Lung-Right") 
        mask_3d_Lung_Left = rtstruct.get_roi_mask_by_name("Lung-Left")
        mask_3d_GTV_1 = rtstruct.get_roi_mask_by_name("GTV-1")
        mask_3d_spinal_cord = rtstruct.get_roi_mask_by_name("Spinal-Cord")

        # Setting what the desired mask is (for the case of a tumour we out GTV-1)
        #easiest to view if looking at lungs, so good sanity check
        mask_3d = mask_3d_Lung_Right + mask_3d_Lung_Left

        #Converting this array from boolean to binary
        mask_3d = mask_3d + 1
        mask_3d = mask_3d - 1

        mask_3d = mask_3d.astype(np.float32)
        mask_3d_image = sitk.GetImageFromArray(mask_3d)
        
        #sitk.WriteImage(mask_3d_image, f"{outputpath}/testrt.nii")
        mask_3d_image = permute_axes(mask_3d_image)
        mask_3d_image.SetSpacing(dicom.GetSpacing())
        mask_3d_image.SetDirection(dicom.GetDirection())
        mask_3d_image.SetOrigin(dicom.GetOrigin())
        mask_3d_image_resampled = resample_volume(mask_3d_image, sitk.sitkNearestNeighbor, 0)

        sitk.WriteImage(mask_3d_image_resampled, f"{os.path.join(outputpath, filename)}.nii")
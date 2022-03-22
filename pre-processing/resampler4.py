"""
This code will use the ResampleImageFilter function to resample both the DICOM and Mask for a 
relevant DICOM series. Also outputs both the DICOM and masks as .nii files which can be used 
in 'worldmatch' to check that the mask and CT line up.
BUT QUICKER
Rory Farwell : Last Edited (17/11/2021) (dd/mm/yyyy)
"""

#========================== IMPORTING LIBRARIES =================================================
from genericpath import getsize
import rt_utils
import sys
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy
from scipy import stats
import os

from MASK_resample_single import DICOM_resampled, DICOM_series_path
reader = sitk.ImageSeriesReader()
#================================================================================================

#========================== DEFINING RESAMPLING VARIABLES =======================================
Output_Spacing = [1, 1, 1]
new_size = [512, 512, 512]
#================================================================================================

#========================== DEFINING FUNCTIONS ==================================================

def resample_volume(volume, interpolator, default_pixel_value) :
    '''
    This function resample a volume to size 512 x 512 x 512 with spacing 1 x 1 x 1.
    It will be used in the resampling functions for both the DICOM series and the mask images.
    To ensure they overlap well when put into a program such as Worldmatch the direction and origin of both
    resampled volumes are set to the direction and origin of the original DICOM series.
    Rory Farwell and Patrick Hastings (14/11/2021)
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(DICOM.GetDirection())
    resample.SetOutputOrigin(DICOM.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(default_pixel_value)

    return resample.Execute(volume)

def permute_axes(volume, permutation_order) :
    """
    This function permutes the axes of the input volume.
    It will be used on the mask because SimpleITK seems to flip the axes
    at some stage in this process.
    Patrick Hastings and Rory Farwell (16/11/2021) 
    """
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder(permutation_order)

    return permute.Execute(volume)

def resample_DICOM(patient_number, interpolator = sitk.sitkLinear, default_pixel_value = -1024) :
    """
    This function will do the whole resampling process on the DICOM series and will make use
    of the earlier defined resample_volume function.
    Rory Farwell and Patrick Hastings (14/11/2021)
    """
    DICOM_write_path = "/mnt/d/resampled_niftys" + str('{0:03}'.format(patient_number)) + '-CT.nii'

    DICOM_paths = reader.GetGDCMSeriesFileNames(DICOM_series_path)
    reader.SetFileNames(DICOM_paths)
    global DICOM #Means that DICOM will be defined globally
    DICOM = sitk.ReadImage(DICOM_paths)
    print(' DICOM size is: ' + str(DICOM.GetSize()))
    DICOM_resampled = resample_volume(DICOM, interpolator, default_pixel_value) #DICOM_resampled is an Image/Object not an array
    print(' resampled DICOM size is: ' + str(DICOM_resampled.GetSize()))

    return sitk.WriteImage(DICOM_resampled, DICOM_write_path)

def resample_MASKS(patient_number, interpolator = sitk.sitkNearestNeighbor, default_pixel_value = 0) :
    """
    This function will perform the whole resampling process on a mask produced from an RTSTRUCT
    file and will make use of the earlier defined resample_volume function.
    Rory Farwell and Patrick Hastings (14/11/2021)
    """
    GTV_1_MASK_write_path = "/mnt/d/resampled_niftys" + str('{0:03}'.format(patient_number)) + '-GTV-1.nii'
    ALL_GTV_MASK_write_path = "/mnt/d/resampled_niftys" + str('{0:03}'.format(patient_number)) + '-ALL_GTV.nii'
    
    #==============================================================
    rtstruct = RTStructBuilder.create_from(DICOM_series_path, RTSTRUCT_path) # Telling the code where to get the DICOMs and RTSTRUCT from

    GTV_1_mask_3d = np.zeros(DICOM.GetSize(), dtype = bool)
    ALL_GTV_mask_3d = np.zeros(DICOM.GetSize(), dtype = bool)

    print(ROIs) #Printing to check the for loop below is doing the right thing.

    for ROI in ROIs :
        if "GTV" in ROI or "gtv" in ROI :
            mask_3d_temp = rtstruct.get_roi_mask_by_name(str(ROI))
            ALL_GTV_mask_3d = ALL_GTV_mask_3d + mask_3d_temp
            print(ROI)
        if ROI == "GTV-1" :
            mask_3d_temp = rtstruct.get_roi_mask_by_name(str(ROI))
            GTV_1_mask_3d = GTV_1_mask_3d + mask_3d_temp
    
    GTV_1_mask_3d = GTV_1_mask_3d.astype(np.float32) #Converting this array from boolean to float so that it can be converted to .nii file
    ALL_GTV_mask_3d = ALL_GTV_mask_3d.astype(np.float32) #Converting this array from boolean to float so that it can be converted to .nii file
    
    GTV_1_mask_3d_image = sitk.GetImageFromArray(GTV_1_mask_3d) #Converting array to an image
    ALL_GTV_mask_3d_image = sitk.GetImageFromArray(ALL_GTV_mask_3d) #Converting array to an image
   
    GTV_1_mask_3d_image = permute_axes(GTV_1_mask_3d_image, [1,2,0]) #permuting the axes because SimpleITK changes the axes ordering
    ALL_GTV_mask_3d_image = permute_axes(ALL_GTV_mask_3d_image, [1,2,0]) #permuting the axes because SimpleITK changes the axes ordering
    
    GTV_1_mask_3d_image.SetSpacing(DICOM.GetSpacing())
    GTV_1_mask_3d_image.SetDirection(DICOM.GetDirection())
    GTV_1_mask_3d_image.SetOrigin(DICOM.GetOrigin())

    ALL_GTV_mask_3d_image.SetSpacing(DICOM.GetSpacing())
    ALL_GTV_mask_3d_image.SetDirection(DICOM.GetDirection())
    ALL_GTV_mask_3d_image.SetOrigin(DICOM.GetOrigin())

    ALL_GTV_mask_3d_image_resampled = resample_volume(ALL_GTV_mask_3d_image, interpolator, default_pixel_value)
    GTV_1_mask_3d_image_resampled = resample_volume(GTV_1_mask_3d_image, interpolator, default_pixel_value)
    
    
    sitk.WriteImage(GTV_1_mask_3d_image_resampled, GTV_1_MASK_write_path)
    sitk.WriteImage(ALL_GTV_mask_3d_image_resampled, ALL_GTV_MASK_write_path)

    return

def opening_test(patient_number) :
    """
    Tries to open both the DICOM series and RTSTRUCT
    """

#=====================================================================================================================
number_of_iterations = 10
filenumbers = np.arange(number_of_iterations)
filenumbers = filenumbers + 1


for i in filenumbers :
    DICOM_series_path = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-" + str('{0:03}'.format(i) + '-CT')
    RTSTRUCT_initial_path = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-" + str('{0:03}'.format(i) + '-RTSTRUCT')
    files_in_RTSTRUCT_folder = os.listdir("/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-" + str('{0:03}'.format(i)) + '-RTSTRUCT')
    RTSTRUCT_filename = files_in_RTSTRUCT_folder[0]
    RTSTRUCT_read_filename = str(RTSTRUCT_filename)
    RTSTRUCT_path = RTSTRUCT_initial_path + '/' + RTSTRUCT_read_filename
    
    try :
        """
        Try to read both the RTSTRUCT and DICOM series.
        """
        rtstruct = RTStructBuilder.create_from(DICOM_series_path, RTSTRUCT_path)
        DICOM_paths = reader.GetGDCMSeriesFileNames(DICOM_series_path)
        reader.SetFileNames(DICOM_paths)
        DICOM = sitk.ReadImage(DICOM_paths)  
        
        ROIs = rtstruct.get_roi_names()
    
        for ROI in ROIs :
            str(ROI)
            if "pre-op" in ROI :
                print('The ROIs for this patient included the gtv-preop so has been skipped')
                continue

        resample_DICOM(patient_number = i, interpolator = sitk.sitkLinear, default_pixel_value = -1024)
        resample_MASKS(patient_number = i, interpolator = sitk.sitkNearestNeighbor, default_pixel_value = 0)

        print('Completed writing files for LUNG1-' + str('{0:03}'.format(i)) +'.')       

    except :
        """
        If unable to read in both RTSTRUCT and DICOM series.
        """
        print('Unable to read either RTSTRUCT or DICOM series for LUNG1-' + str('{0:03}'.format(i)) + 'so it has been skipped.')
        continue # if the RTSTRUCT or DICOM can't be opened then this 'i' is skipped

print(len(os.listdir("/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED")))
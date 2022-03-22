"""
This file will read in a sorted tree of dicoms holding all the patient image and return resampled nifty files

"""

import SimpleITK as sitk
import os
import nibabel as nib

#global variables
filepath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED" #note this is filepath to non resampled data
niftypath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_NIFTY" #path to resampled nifty files


def resample_volume(volume, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 3]):
    #reads in image where the volume is determined by the pixel spacing passed in
    #new spacing has been decided as these values as these are the integres closest to the majority of slices.

    new_size = [512, 512, 512] #dimensions chosen based on images in data set, all images being 512x512 then 512 being the next power of 2 after largest file
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([1, 1, 3])
    resample.SetDefaultPixelValue(-1024)

    return resample.Execute(volume)

# rtname = "LUNG1-001-RTSTRUCT"
# rtpath = os.path.join(filepath, rtname)
# print(rtpath)
# reader = sitk.ImageSeriesReader()

# img = nib.load(f"{niftypath}/rtstructtest.nii")
# data = img.get_fdata()
# print(data)

# print(data.shape)


# print(volume)

# img = nib.load(f"{niftypath}/rtstructtest.nii")
reader = sitk.ImageSeriesReader()
counter = 0
for filename in os.listdir(filepath): 
    if "-CT" in filename:
        print(filename)
        
        dcm_paths =reader.GetGDCMSeriesFileNames(os.path.join(filepath,filename))  
        reader.SetFileNames(dcm_paths)
        volume = sitk.ReadImage(dcm_paths)
        print(volume)
        x = resample_volume(volume)
        counter += 1
        print(x.GetSize())
        print(x.GetSpacing())

        #print(x)
        sitk.WriteImage(x, f"{os.path.join(niftypath, filename)}.nii")

    elif "-RTSTRUCT" in filename:
        print(filename)
        img = nib.load(os.path.join(niftypath, "rtstructtest.nii"))
        data = img.get_fdata()
        print(data.shape)
        data_img = sitk.GetImageFromArray(data)
        output = resample_volume(data_img, interpolator=sitk.sitkNearestNeighbor)
        print(output.GetSize())
        print(output.GetSpacing())


        break

        


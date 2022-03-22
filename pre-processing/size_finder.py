"""
This file looks through a dicom tree and pulls out the size of all the patient files. This is just for the purpose of determining what new size you want to resize to.
"""

import SimpleITK as sitk
import numpy as np

filepath = "/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED"
counter = 0 #Used to check that the code is running

#Defining the start of the path name to where my files are (this depends on your path and the
# naming scheme you used when using dicomsort.py)
directory = '/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-'

# defining the start of the path name of the output file

output_file = '/mnt/c/Users/Patrick/Documents/MPHYS_DATA_SORTED/LUNG1-'


#defining the number of iterations the for loop will perform. +1 is due to the range function
number_of_iterations = 422+1

biggest_size = 0
biggest_size_label = 0
size =[]
spacing = []
reader = sitk.ImageSeriesReader()

for x in range(1, number_of_iterations) :
    _3_digit_x = '{0:03}'.format(x) #formats the 'x' to the form 'yzx' e.g. 1 -> 001
                                    # so that it fits the format of the naming scheme used
                                    # e.g. LUNG1-001-CT
    
    directory_full = directory + str(_3_digit_x) + '-CT/' #   This line will change depending on the naming scheme that you have used
    dicom_names = reader.GetGDCMSeriesFileNames(directory_full)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    size.append((image.GetSize()))
    spacing.append((image.GetSpacing()))
    print(x)
    if x == 3:
        break
    else: 
        continue


size = np.vstack(size)
spacing = np.vstack(spacing)
"""
This code produces has two functions:
    - Produce a .nii file for all the DICOM series in your chosen source directory
    - Tell you which series has the largest number of slices (useful knowledge for preprocessing) 
      This then lets you add slices of air to the non-largest series in order to be fed into a CNN.
      This is because the CNN will expect to read-in data with the same dimensions.
You may wish to only run one of them in order to save time.

Rory Farwell (21/10/2021)
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~IMPORTING FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import print_function
import SimpleITK as sitk
import sys
import os
reader = sitk.ImageSeriesReader()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEFINING VARAIABLES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
counter = 0 #Used to check that the code is running

"""Defining the start of the path name to where my files are (this depends on your path and the
   naming scheme you used when using dicomsort.py)"""
directory = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-'

""" defining the start of the path name of the output file """
output_file = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_nifti_test/LUNG1-'


"""defining the number of iterations the for loop will perform. +1 is due to the range function"""
number_of_iterations = 422+1 #I want to make this so that the program can read the number of series in the chosen directory
#                             rather than requiring user input.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for x in range(1, number_of_iterations) :
    _3_digit_x = '{0:03}'.format(x) #formats the 'x' to the form 'yzx' e.g. 1 -> 001
                                    # so that it fits the format of the naming scheme used
                                    # e.g. LUNG1-001-CTUnknownStudyID
    output_file_full = output_file + _3_digit_x +'.nii' # the '+.nii' ensures the output file is a NIfTI
    directory_full = directory + str(_3_digit_x) + '-CTUnknownStudyID/'
    dicom_names = reader.GetGDCMSeriesFileNames(directory_full)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    print("Image size:", size[0], size[1], size[2])
    print("Writing image:", output_file_full)
    sitk.WriteImage(image, output_file_full) #writes the output file to output_file_full
    counter += 1
    print(counter) # Checking that the code is running and its progress

    #The two lines below would open the file in ImageJ (Fiji)
    """if ("SITK_NOSHOW" not in os.environ):
            sitk.Show(image, "Dicom Series")"""
    
  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ THE CODE BELOW WILL TELL YOU THE LARGEST SERIES SIZE IN YOUR DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note : It is currently specified for our data but in the future may be made more general

counter = 0 #Used to check that the code is running

#Defining the start of the path name to where my files are (this depends on your path and the
# naming scheme you used when using dicomsort.py)
directory = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-'

# defining the start of the path name of the output file
output_file = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_nifti/LUNG1-'


#defining the number of iterations the for loop will perform. +1 is due to the range function
number_of_iterations = 422+1

biggest_size = 0
biggest_size_label = 0

for x in range(1, number_of_iterations) :
    _3_digit_x = '{0:03}'.format(x) #formats the 'x' to the form 'yzx' e.g. 1 -> 001
                                    # so that it fits the format of the naming scheme used
                                    # e.g. LUNG1-001-CTUnknownStudyID
    output_file_full = output_file + _3_digit_x +'.nii' # the '+.nii' ensures the output file is a NIfTI
    directory_full = directory + str(_3_digit_x) + '-CTUnknownStudyID/' #   This line will change depending on the naming scheme that you have used
    dicom_names = reader.GetGDCMSeriesFileNames(directory_full)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    if size[2] > biggest_size :
        biggest_size = size[2]
        biggest_size_label = x
    counter += 1
    print(counter) # Checking that the code is running and its progress

print(biggest_size)
print(biggest_size_label)
print("The largest depth in the NSCLC-Radiomics data set is 297 from LUNG1-" + str('{0:03}'.format(biggest_size_label))) 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
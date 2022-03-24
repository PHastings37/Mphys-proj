"""
This file contains functions that will imported in the training file.

Rory Farwell and Patrick Hastings 22/3/2022
"""

import numpy as np
import random


def equalise_array_lengths(array_1, array_2) :
  """
  This functions takes in the arguments of two lists and makes sure they are returned as the same length.

  Rory Farwell 02/12/2021
  """
  # output_array = []

  if len(array_1) > len(array_2) :
    array_1 = array_1[:len(array_2)]
  elif len(array_1) < len(array_2) :
    array_2 = array_2[:len(array_1)]

  """
  These should be used if we want to test something very quickly. 
  Limits the data set to total length 20 (10 alive and 10 dead)
  """
  # array_1 = array_1[:10]
  # array_2 = array_2[:10]

  return (array_1, array_2)

def remove_same_elements(small_array, long_array) :
  """
  For use in the context, all the elements in small_array come from long_array.
  This function will remove all of the elements used in small_array from_long_array.  
  """
  for element in small_array :
    long_array.remove(element)
  return long_array

def create_subgroup(input_array, original_array_length, desired_percentage) :
  """
  This function outputs a subgroup array (e.g. training array) using a specified output array name,
  input array and percentage length
  """
  desired_length = int(original_array_length * desired_percentage)
  output_array = random.sample(input_array, desired_length)
  return output_array

def open_metadata(metadata_filepath) :
    """
    Opens the metadata file using the globall defined variables 'project_folder' and 'clinical_data_filename'.

    Returns patient_IDs which will be used for checking the data is shuffled
    Returns time_markers which will be used for checking patient status at a specified timepoint
    Returns dead_statuses.

    Rory Farwell and Patrick Hastings 08/02/2022
    """
    metadata = np.genfromtxt(metadata_filepath, comments = '%', dtype="str", delimiter=",")
    print(f"Length of metadata array is {len(metadata)}")

    # Retrieve data from metadata file
    patient_IDs = metadata[:,0] # selecting patient IDs from the csv file
    time_markers = metadata[:,8] # selecting the day of the last patient review from the csv file
    dead_statuses = metadata[:,9] # selecting the dead status on the last review day

    time_markers = time_markers.astype(np.float32) # converting to float
    dead_statuses = dead_statuses.astype(np.float32) # converting to float

    return patient_IDs, time_markers, dead_statuses

def patient_status_on_check_day(check_day, dead_statuses, patient_IDs, time_markers) :
    """
    Changes patient status according to patient status on the check day.
    There are 4 possibilities here:
    1. Dead: timepoint < check_day -> Dead on check day [1 -> 1]
    2. Dead: timepoint > check_day -> Alive on check day [1 -> 0]
    3. Alive: timepoint > check_day -> Alive on check day [0 -> 0]
    4. Alive: timepoint < check_day -> no info on current status -> right-censored data

    Rory Farwell and Patrick Hastings 08/02/2022
    """

    dead_counter = 0
    alive_counter = 0
    no_info_counter = 0
    dead_patient_array = []
    alive_patient_array = []

    for i in range(len(dead_statuses)) :
        temp_patient_ID = patient_IDs[i]
        temp_dead_status = dead_statuses[i]
        temp_time_marker = time_markers[i]
        if temp_dead_status == 1 : #if the patient is dead
            if temp_time_marker < check_day :# situation 1
                dead_patient_array.append([temp_patient_ID, 1])
                dead_counter += 1
                continue
            elif temp_time_marker > check_day : # situation 2
                alive_patient_array.append([temp_patient_ID, 0])
                alive_counter += 1
                continue
        elif temp_dead_status == 0 : #if the patient is alive
            if temp_time_marker < check_day : # situation 4
                no_info_counter += 1
                continue
            elif temp_time_marker > check_day :
                alive_patient_array.append([temp_patient_ID, 0])
                alive_counter += 1
                continue

    # Printing the results of this loop (the number of dead and alive patients at the check day)
    print(f"Number of patients dead after {check_day} days: {dead_counter}")
    print(f"Number of patients alive counter after {check_day} days: {alive_counter}")
    print(f"Number of right-censored data when using a check day of {check_day} days: {no_info_counter}")

    return dead_patient_array, alive_patient_array

def equalise_arrays(array_1, array_2) :
    """
    Equalise the length arrays 1 and 2 to the length of the shortest of the two.

    Rory Farwell and Patrick Hastings 08/02/2022
    """
    new_array_1 = equalise_array_lengths(array_1, array_2)[0]
    new_array_2 = equalise_array_lengths(array_1, array_2)[1]
    print(f"The alive and dead arrays have been sorted (randomly) so that they are both of length {len(new_array_1)}, {len(new_array_2)}")

    return new_array_1, new_array_2

def create_final_datasets(train_patients_dead, train_patients_alive,
    test_patients_dead, test_patients_alive, validate_patients_dead, validate_patients_alive) :
    """
    Combines the dead and alive arrays of each training, validation and testing data sets to produce the final lists.
    And shuffles them.

    Rory Farwell and Patrick Hastings 08/02/2022
    """
    outcomes_train = train_patients_dead + train_patients_alive
    outcomes_test = test_patients_dead + test_patients_alive
    outcomes_validate = validate_patients_dead + validate_patients_alive
    
    random.shuffle(outcomes_train)
    random.shuffle(outcomes_test)
    random.shuffle(outcomes_validate)

    print(f'Length of shuffled outcomes_train: {len(outcomes_train)}') 
    print(f'Length of shuffled outcomes_validate: {len(outcomes_validate)}')
    print(f'Length of shuffled outcomes_test: {len(outcomes_test)}')

    return outcomes_train, outcomes_validate, outcomes_test
"""
This code trains, validates and tests a custom binary classfiying CNN.

The inputs to the network are 264 x 264 x 264 textured masks of NSCLC pre-treatment CT scans.

Rory Farwell and Patrick Hastings 08/02/2022

"""
#====================================================================
#======================= IMPORTING FUNCTIONS ========================
#====================================================================
import os
from click import open_file
print(f'Running {__file__}')

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import sklearn

from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import Module
from torch.nn import Conv3d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch import nn
from torch import reshape
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.io import read_image
from torch.optim import Adam
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import zoom, rotate
import sys
import time
import itertools
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Import_Functions.Tensorboard_writer import customWriter
from Import_Functions.Results_Class import results
import Import_Functions.SystemArguments as SysArg
import Import_Functions.ClinicalDataProcessing as CDP
import Import_Functions.train_and_valid_loops as Loops
import Import_Functions.ImageDataset_Class as ImageDataset_Class
import Import_Functions.ResNet as RN
import Import_Functions.ConvNet as ConvNet

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from colorama import Fore

num_epochs, user_choice_of_check_day, plot_folder_path, network_filepath, plot_filename = SysArg.get_system_arguments(sys.argv)
print(num_epochs, user_choice_of_check_day, plot_folder_path, network_filepath)
print(Fore.CYAN + "############# USERINPUTS #############")
print(f"Number of epochs = {num_epochs}")
print(f"Check day = {365*user_choice_of_check_day} days.")
print(f"Plot folder path: {plot_folder_path}")
print(f"Filepath of best performing network: {network_filepath}" + Fore.RESET)

plot_date = time.strftime("%Y_%m_%d")
plot_time = time.strftime("%H_%M")
#====================================================================
#=================== SELECT DEVICE ==================================
#====================================================================

# Connect to cuda:1 if it is available (which it should be)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
print(f'Using {device} device')

project_folder = "/home/ptrickhastings37_gmail_com/data/rory_and_pat_data/"
clinical_data_filename = "382_metadata.csv"
print(os.path.join(project_folder, clinical_data_filename))


#====================================================================
#=================== DEFINING FUNCTIONS =============================
#====================================================================

def convert_to_one_hot_labels(images, labels) :
    """
    This function converts the labels to one-hot labels so that they will work with the BCEwithLogitsLoss
    """
    hot_labels = torch.empty((images.shape[0], 2))
    
    for index in range(len(labels)) :
        if labels[index] == 0 :
            hot_labels[index,0] = 1
            hot_labels[index,1] = 0
        elif labels[index] == 1 :
            hot_labels[index, 0] = 0
            hot_labels[index, 1] = 1
    
    return hot_labels

def save_loss_plots():
  new_avg_train_loss = avg_train_loss
  new_avg_valid_loss = avg_valid_loss

  epochs = np.array(range(epoch_counter)) + 1
  loss_plot = plt.figure()
  plt.xticks(fontsize = 20)
  plt.yticks(fontsize = 20)
  loss_plot.set_size_inches(20,10)
  plt.plot(epochs, new_avg_train_loss, label = 'Average training loss', linewidth = 7.0)
  plt.plot(epochs, new_avg_valid_loss, label = 'Average validation loss', linewidth = 7.0)
  plt.legend(loc = 'best', prop={'size': 20})
  plt.ylabel('Average Loss', fontsize = 20)
  plt.xlabel('Epoch Number', fontsize = 20)
  plt.savefig(f'{plot_folder_path}{plot_time}_{plot_filename}_epoch_{epoch_counter}')
  print(f'The loss plot has been saved in: {plot_folder_path}{plot_date}/{plot_time}_{plot_filename}_epoch_{epoch_counter}')

  return

#====================================================================
#=================  CLASS DEFINITIONS ===============================
#====================================================================
transform = transforms.Compose(
    [transforms.ToTensor() ] #added 13/12/2021 to normalize the inputs. THIS NORMALIZES to mean = 0 and std = 1
)    
        
# model = ConvNet.CNN().to(device) # Send the CNN to the device
model = RN.generate_model(10, device)
#====================================================================
#=================== DEFIINING VARIABLES ============================
#====================================================================

check_day = 365 * user_choice_of_check_day # This is defining the timeframe for which our CNN will consider the binary output (in days) 

# sanity check to check progress
counter = 0 


# Creating empty arrays that will be appended to later
# These will contain the patient ID and dead status (on the check day).
training_array = []
testing_array = []
validation_array = []

#====================================================================
#=================== HYPER PARAMETER DEFINITION =====================
#====================================================================
batch_size = 4
learning_rate = 0.001

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

#====================================================================
#=================== MAIN CODE ======================================
#====================================================================

metadata_filepath = os.path.join(project_folder, clinical_data_filename)
print(f'metadata_filepath: {metadata_filepath}')
patient_IDs, time_markers, dead_statuses = CDP.open_metadata(metadata_filepath)
dead_patient_array, alive_patient_array = CDP.patient_status_on_check_day(check_day, dead_statuses, patient_IDs, time_markers)

#  Shuffle both arrays to ensure a random selection of patient data which will be input to the CNN
random.shuffle(dead_patient_array)
random.shuffle(alive_patient_array)

# Equalising the length of the 'dead' and 'alive' arrays so that we can ensure optimum training proportions
new_dead_patient_array, new_alive_patient_array = CDP.equalise_arrays(dead_patient_array, alive_patient_array)
equalised_array_length = len(new_alive_patient_array)

train_patients_dead = CDP.create_subgroup(new_dead_patient_array, equalised_array_length, 0.7)
train_patients_alive = CDP.create_subgroup(new_alive_patient_array, equalised_array_length, 0.7)

new_dead_patient_array = CDP.remove_same_elements(train_patients_dead, new_dead_patient_array)
new_alive_patient_array = CDP.remove_same_elements(train_patients_alive, new_alive_patient_array)

test_patients_dead = CDP.create_subgroup(new_dead_patient_array, equalised_array_length, 0.15)
test_patients_alive = CDP.create_subgroup(new_alive_patient_array, equalised_array_length, 0.15)

new_dead_patient_array = CDP.remove_same_elements(test_patients_dead, new_dead_patient_array)
new_alive_patient_array = CDP.remove_same_elements(test_patients_alive, new_alive_patient_array)

validate_patients_dead = CDP.create_subgroup(new_dead_patient_array, equalised_array_length, 0.15)
validate_patients_alive = CDP.create_subgroup(new_alive_patient_array, equalised_array_length, 0.15)

new_dead_patient_array = CDP.remove_same_elements(validate_patients_dead, new_dead_patient_array)
new_alive_patient_array = CDP.remove_same_elements(validate_patients_alive, new_alive_patient_array)

print(f"After separation into training, validation and testing arrays the number of unused data is {len(new_dead_patient_array) + len(new_alive_patient_array)}. If not then something has gone wrong.")

outcomes_train, outcomes_validate, outcomes_test = CDP.create_final_datasets(train_patients_dead, train_patients_alive,
    test_patients_dead, test_patients_alive, validate_patients_dead, validate_patients_alive)

open_file = open("testing_data_list.pkl", "wb")
pickle.dump(outcomes_test, open_file)
open_file.close()

training_data = ImageDataset_Class.ImageDataset(outcomes_train, os.path.join(project_folder, "textured_masks"), transform = transform)
validation_data = ImageDataset_Class.ImageDataset(outcomes_validate, os.path.join(project_folder, "textured_masks"), transform = transform)


train_dataloader = DataLoader(training_data, batch_size = 4, shuffle = True)
validation_dataloader = DataLoader(validation_data, batch_size = 4, shuffle = True)

summary(model, (1,160,160,160), batch_size = 4)
patient = ""

#============================ TRAINING AND VALIDATION LOOP ==========
writer = customWriter(project_folder, 4, 0, 1)
writer.epoch = 0
n_total_steps = len(train_dataloader)
train_loss = []
valid_loss = []
avg_train_loss = np.empty(0)
avg_valid_loss = np.empty(0)
all_training_losses = []
epoch_counter = 0
minimum_average_validation_loss = 100000

for epoch in range(num_epochs):
    #send train loss and val loss to tb from here?
    writer.epoch = epoch
    epoch_validation_targets = []
    epoch_validation_predictions = []
    epoch_counter += 1
    avg_train_loss = np.append(avg_train_loss, Loops.training_loop(epoch, model, train_dataloader, device, writer, criterion, optimizer, all_training_losses,
    num_epochs, n_total_steps, train_loss))
    epoch_average_validation_loss = Loops.validation_loop(epoch, model, validation_dataloader, device, criterion, epoch_validation_targets, 
    epoch_validation_predictions, writer)
    avg_valid_loss = np.append(avg_valid_loss, epoch_average_validation_loss)
    if epoch_average_validation_loss < minimum_average_validation_loss :
      minumum_average_validation_loss = epoch_average_validation_loss
      torch.save(model.state_dict(), f'{network_filepath}_epoch{epoch+1}')
      epoch_save_number = epoch+1
    scheduler.step(avg_valid_loss[epoch])
    # print(f"epoch_validation_targets = {epoch_validation_targets}")
    # print(f"epoch_validation_predictions = {epoch_validation_predictions}")
    epoch_results = results(epoch_validation_targets, epoch_validation_predictions)
    conf_mat = epoch_results.confusion_matrix()
    print(f'(TP, TN, FP, FN): {epoch_results.evaluate_results()}')
    # print(type(conf_mat))
    if (epoch+1)%5 == 0:
      writer.plot_confusion_matrix(conf_mat, ["alive","dead"])
      save_loss_plots()

print('FINISHED TRAINING')
print(Fore.CYAN + f"The best performing network from this training run was saved"
          + f"to {network_filepath}_epoch{epoch_save_number}" + Fore.RESET)
writer.close()
"""
This code contains the testing loop for the our ResNet model.
This code will load in the list of the unseen patient data and
the network state parameters of the best performing model during the 
training section. This code will then train the network on this unseen
data.

Rory Farwell and Patrick Hastings 24/03/2022
"""
import os
import sys
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
import Import_Functions.ResNet as RN
import Import_Functions.testing_loop as loop
# Using GPU 1, which is the one that has been allocated to us
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
print(f'Using {device} device')

#Defining some filepaths
project_folder = "/home/ptrickhastings37_gmail_com/data/rory_and_pat_data/"
clinical_data_filename = "382_metadata.csv"
print(f"Clinical data filepath: {os.path.join(project_folder, clinical_data_filename)}")

#Defining some functions and classes
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

transform = transforms.Compose(
    [transforms.ToTensor()] #added 13/12/2021 to normalize the inputs. THIS NORMALIZES to mean = 0 and std = 1
)

def window_and_level(image, level = -600, window = 1500) :
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld

class ImageDataset(Dataset) :
  def __init__(self, annotations, img_dir, transform = transform, target_transform = None, shift_augment = False, rotate_augment = False, scale_augment = False, flip_augment = False) :
    self.img_labels = annotations
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    self.shifts = shift_augment
    self.rotations = rotate_augment
    self.flips = flip_augment
    self.scales = scale_augment
    print(self.img_labels)

  def __len__(self) :
    return len(self.img_labels)

  def __getitem__(self,idx) :
    img_path = os.path.join(self.img_dir, self.img_labels[idx][0] + "-GTV-1.nii" )
    image_sitk = sitk.ReadImage(img_path)
    # ID = self.img_labels[idx][0]
    # print(f'ID: {ID}')
    image = sitk.GetArrayFromImage(image_sitk)
    label = self.img_labels[idx][1]

    # Augmentations
    if self.shifts:
      mx_x, mx_yz = 10, 10 
      # find shift values
      cc_shift, ap_shift, lr_shift = random.randint(-mx_x,mx_x), random.randint(-mx_yz,mx_yz), random.randint(-mx_yz,mx_yz)
      # pad for shifting into
      image = np.pad(image, pad_width=((mx_x,mx_x),(mx_yz,mx_yz),(mx_yz,mx_yz)), mode='constant', constant_values=-1024) # original is zero but appears to work better with -1024 (HU of air)
      # crop to complete shift
      image = image[mx_x+cc_shift:160+mx_x+cc_shift, mx_yz+ap_shift:160+mx_yz+ap_shift, mx_yz+lr_shift:160+mx_yz+lr_shift]

    if self.rotations and random.random() < 0.5 : # normal is 0.5
      roll_angle = np.clip(np.random.normal(loc=0,scale=3), -15, 15)
      # print(f'Rotation by angle {roll_angle} applied.')
      #print(roll_angle)
      image = self.rotation(image, roll_angle, rotation_plane=(1,2))

    if self.scales and random.random() < 0.5 : # normal is 0.5
      # same here -> zoom between 80-120%
      scale_factor = np.clip(np.random.normal(loc=1.0,scale=0.05), 0.7, 1.3)
      # print(f'Scaled by factor {scale_factor}.')
      image = self.scale(image, scale_factor)
    
    if self.flips and random.random() < 0.5 : # normal is 0.5
        # print(f'Left-right flip applied')
        image = np.flipud(image)
    
    image = window_and_level(image)

    if self.transform :
      image = self.transform(image)
    if self.target_transform :
      label = self.target_transform(label)
    return image,label
  
  def rotation(self, image, rotation_angle, rotation_plane):
      # rotate the image or mask using scipy rotate function
      order, cval = (3, -1024)
      return rotate(input=image, angle=rotation_angle, axes=rotation_plane, reshape=False, order=order, mode='constant', cval=cval)
    
  def scale(self, image, scale_factor):
      # scale the image or mask using scipy zoom function
      order, cval = (3, -1024)
      height, width, depth = image.shape
      zheight = int(np.round(scale_factor*height))
      zwidth = int(np.round(scale_factor*width))
      zdepth = int(np.round(scale_factor*depth))
      # zoomed out
      if scale_factor < 1.0:
          new_image = np.full_like(image, cval)
          ud_buffer = (height-zheight) // 2
          ap_buffer = (width-zwidth) // 2
          lr_buffer = (depth-zdepth) // 2
          new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth, lr_buffer:lr_buffer+zdepth] = zoom(input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
          return new_image
      elif scale_factor > 1.0:
          new_image = zoom(input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
          ud_extra = (new_image.shape[0] - height) // 2
          ap_extra = (new_image.shape[1] - width) // 2
          lr_extra = (new_image.shape[2] - depth) // 2
          new_image = new_image[ud_extra:ud_extra+height, ap_extra:ap_extra+width, lr_extra:lr_extra+depth]
          return new_image
      return image

#############################################################
######################## MAIN CODE ##########################
#############################################################

if len(sys.argv) != 2:
    print("Correct usage is: python Binary_Classifier_Testing.py <file path of the saved network>")
    sys.exit(1)

# Open the file
open_file = open("testing_data_list.pkl", "rb")
outcomes_test = pickle.load(open_file)
open_file.close()
# print(outcomes_test)

test_data = ImageDataset_Class.ImageDataset(outcomes_test, os.path.join(project_folder, "textured_masks"), transform = transform, target_transform = None, shift_augment = False, rotate_augment = False, scale_augment = False, flip_augment = False)
test_dataloader = DataLoader(test_data, batch_size = 4, shuffle = False)

model = RN.generate_model(10, device)
model.load_state_dict(torch.load(sys.argv[1]))

# print("were here")
testing_targets = []
testing_predictions = []
testing_accuracy = loop.testing_loop(model, test_dataloader, device, testing_targets, testing_predictions)

testing_results = results(testing_targets, testing_predictions)
print(f"Targets: {testing_targets}")
print(f"Predictions: {testing_predictions}")
print(f'(TP, TN, FP, FN): {testing_results.evaluate_results()}')
print(f'Accuracy on testing set = {testing_accuracy:.1f}%')


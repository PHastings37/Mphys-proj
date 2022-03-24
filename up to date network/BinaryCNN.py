"""
This code trains, validates and tests a custom binary classfiying CNN.

The inputs to the network are 264 x 264 x 264 textured masks of NSCLC pre-treatment CT scans.

Rory Farwell and Patrick Hastings 08/02/2022

"""
#====================================================================
#======================= IMPORTING FUNCTIONS ========================
#====================================================================

# Un hash below if on Google Colab
# !pip install torch torchvision
# !pip install opencv-contrib-python
# !pip install scikit-learn
# !pip install SimpleITK

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

if len(sys.argv) < 4 :
  print("Error: User inputs are wrong.")
  print("The correct usage is: '/content/gdrive/MyDrive/University/Year_4_Sem_2/BinaryClassifier.py' <Number of epochs> <Check day>  <Do you want to use full dataset> <Plot save name.png>")

#NUMBER OF EPOCHS
num_epochs = int(sys.argv[1])

#CHECK DAY
user_choice_of_check_day = float(sys.argv[2])

#WHETHER TO USE FULL DATASET OR PARTIAL DATASET
if sys.argv[3].lower() == 'full' :
  print('You chose to run the full dataset')
  full_dataset_choice = True
elif sys.argv[3].lower() == 'partial' :
  print('You chose to run the partial dataset')
  full_dataset_choice = False
else:
  print('Error: Your input for whether to use the full dataset must be "True" or "False".')
  sys.exit(1)

#FILENAME
# Create a folder at path "folder path" if one does not already exist
plot_filename = sys.argv[4]
print(plot_filename)
plot_date = time.strftime("%Y_%m_%d")
plot_time = time.strftime("%H_%M_%S")
#plot_folder_path = f"/home/ptrickhastings37_gmail_com/data/rory_and_pat_results/loss_plots/{plot_date}/"
plot_folder_path = f"/home/ptrickhastings37_gmail_com/data/pat_results/{plot_date}/"
if not os.path.exists(plot_folder_path):
  os.makedirs(plot_folder_path)

#====================================================================
#=================== COLAB SPECIFIC CODE ============================
#====================================================================

# from google.colab import drive
# drive.mount('/content/gdrive')
# project_folder = "/content/gdrive/My Drive/Mphys project"
# clinical_data_filename = "cancerdatasem2(40p).csv"
# print(os.path.join(project_folder, clinical_data_filename))

#====================================================================
#=================== VSCode SPECIFIC CODE ===========================
#====================================================================
project_folder = "/home/ptrickhastings37_gmail_com/data/rory_and_pat_data/"
clinical_data_filename = "382_metadata.csv"
print(os.path.join(project_folder, clinical_data_filename))


#====================================================================
#=================== SELECT DEVICE ==================================
#====================================================================

# Connect to GPU is available
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
print(f'Using {device} device')
# /content/gdrive/MyDrive/MPhys/Data/COLAB-Clinical-Data.csv
# Specify project folder location

if full_dataset_choice == False:
  print("You chose to use the partial dataset")
  project_folder = "/content/gdrive/MyDrive/MPhys/Data/"
  clinical_data_filename = "COLAB-Clinical-Data.csv"
  print(os.path.join(project_folder, clinical_data_filename))
elif full_dataset_choice == True:
  print("You chose to use the full dataset")
  project_folder = "/home/ptrickhastings37_gmail_com/data/rory_and_pat_data/"
  clinical_data_filename = "382_metadata.csv"
  print(os.path.join(project_folder, clinical_data_filename))


#====================================================================
#=================== DEFINING FUNCTIONS =============================
#====================================================================

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

  # array_1 = array_1[:20]
  # array_2 = array_2[:20]

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

def open_metadata() :
    """
    Opens the metadata file using the globall defined variables 'project_folder' and 'clinical_data_filename'.

    Returns patient_IDs which will be used for checking the data is shuffled
    Returns time_markers which will be used for checking patient status at a specified timepoint
    Returns dead_statuses.

    Rory Farwell and Patrick Hastings 08/02/2022
    """
    metadata_file = os.path.join(project_folder, clinical_data_filename)
    print(f'metadata_file path: {metadata_file}')
    metadata = np.genfromtxt(metadata_file, comments = '%', dtype="str", delimiter=",")
    print(f"Length of metadata array is {len(metadata)}")

    # Retrieve data from metadata file
    patient_IDs = metadata[:,0] # selecting patient IDs from the csv file
    time_markers = metadata[:,8] # selecting the day of the last patient review from the csv file
    dead_statuses = metadata[:,9] # selecting the dead status on the last review day

    time_markers = time_markers.astype(np.float32) # converting to float
    dead_statuses = dead_statuses.astype(np.float32) # converting to float

    return patient_IDs, time_markers, dead_statuses

def patient_status_on_check_day(check_day) :
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

def create_final_datasets() :
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

def training_loop():
    epoch_train_loss = 0 # will be used for plotting testing vs validation loss curves
    n_training_samples = 0
    print(f'Training for epoch {epoch+1}')
    print("=============================")

    model.train()

    

    for i, (images, labels, patient) in enumerate(train_dataloader):
        images = reshape(images, (images.shape[0], 1, 160,160,160))
        images = images.float()
        
        #writer.plot_batch(labels[i], images[i])
        hot_labels = convert_to_one_hot_labels(images, labels)
        
        images = images.to(device)
        hot_labels = hot_labels.to(device)

        #print(f"tag:{patient}")
        pat_temp = patient.pop(0)
        pat_temp_list = list(pat_temp)
        for j in range(len(pat_temp_list)):

          writer.plot_tumour(pat_temp_list[j], images[j])
        

        #forward pass
        outputs = model(images)
       
        # print (outputs)
        loss = criterion(outputs, hot_labels)
        

        #backwards pass
        optimizer.zero_grad() #clears gradients before performing backpropagation
        loss.backward()
        optimizer.step()

        # Add the number of images in this batch to n_training_samples which will
        # be used when calculating the average loss per image in the training set
        n_training_samples += labels.shape[0]

        # Updating the total training loss of this epoch
        all_training_losses.append(loss.item())
        epoch_train_loss += loss.item()

        if (i+1)%5 == 0 :
            print(f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            print(f"outputs: {outputs}")

    # Append the train_loss list with the total training loss for this epoch
    train_loss.append(epoch_train_loss)

    #Append the avg_train_loss list with the average training loss of this epoch
    avg_train_loss = epoch_train_loss/n_training_samples
    print(f"Average training loss list: {avg_train_loss}")

    print(f"Training loss array at end of epoch {epoch + 1}: {train_loss}. Total number of images used = {n_training_samples}.")
    print(f"Finished training for epoch {epoch + 1}")
    
    writer.plot_scalar("Train Loss", avg_train_loss)
    return avg_train_loss


def validation_loop() :
    print(f'Validation for epoch {epoch + 1}')
    print('=================================')

    model.eval()

    with torch.no_grad(): # ensuring gradients are not calculated during the validation loop
        valid_epoch_loss = 0
        n_valid_correct = 0
        n_valid_samples = 0
        for images, labels, patient in validation_dataloader :
            images = reshape(images, (images.shape[0],1 ,160,160,160))
            images = images.float()
            hot_labels = convert_to_one_hot_labels(images, labels)

            images = images.to(device)
            hot_labels = hot_labels.to(device)
            outputs = model(images)

            # calculate loss of validation set
            loss = criterion(outputs, hot_labels)
            valid_epoch_loss += loss.item()

            # max returns (value, index) 
            _,predictions = torch.max(outputs, 1)
            _,targets = torch.max(hot_labels, 1)
            #print(f'predictions: {predictions}')
            #print(f'targets: {targets}')
            #print(f'correct in this batch: {(predictions == targets).sum().item()}')
            n_valid_samples += labels.shape[0]
            n_valid_correct += (predictions == targets).sum().item()
            #print(f'n_correct = {n_correct}. n_samples = {n_samples}')

            labels_numpy = labels.numpy()
            for index in range(labels_numpy.size):
              epoch_validation_targets.append(labels_numpy[index])
            
            predictions_numpy = predictions.cpu().numpy()
            for index in range(predictions_numpy.size):
              epoch_validation_predictions.append(predictions_numpy[index])


        avg_valid_loss = valid_epoch_loss/n_valid_samples
        #valid_loss.append(valid_epoch_loss)
        acc = (100*n_valid_correct)/n_valid_samples
        print(f'Accuracy on validation set for epoch {epoch+1} = {acc:.1f}%')
        print(f'Loss on validation set = {valid_epoch_loss}')

        print(f'Finished validation for epoch {epoch+1}')
        print('=============================================')
        
    writer.plot_scalar("Validation Loss", avg_valid_loss)
    writer.plot_scalar("Validation Accuracy", acc)
    return avg_valid_loss

def testing_loop():
  print("---- Currently testing the network on unseen data ----")
  model.eval()

  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    counter = 0
    for images, labels, patient in test_dataloader :
      # counter+=1
      # print(counter)
      images = images = reshape(images, (images.shape[0],1,160,160,160))
      images = images.float()
      hot_labels = convert_to_one_hot_labels(images, labels)

      images = images.to(device)
      hot_labels = hot_labels.to(device)
      outputs = model(images)
      # max returns (value, index) 
      _,predictions = torch.max(outputs, 1)
      _,targets = torch.max(hot_labels,1)
      #print(f'predictions: {predictions}')
      #print(f'targets: {targets}')
      n_samples += hot_labels.shape[0]
      n_correct += (predictions == targets).sum().item()
      #print(f'n_correct = {n_correct}. n_samples = {n_samples}')
    
    acc = (100*n_correct)/n_samples
    return acc

def window_and_level(image, level = -700, window = 1000) :
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld


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

"""
Custom tensorboard writer class
From Donal McSweeney
"""
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

class customWriter(SummaryWriter):
    def __init__(self, log_dir, batch_size, epoch, num_classes):
        super(customWriter, self).__init__()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_classes = num_classes
        self.train_loss = []
        self.val_loss = []
        self.class_loss = {n: [] for n in range(num_classes+1)}
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def reset_losses(self):
        self.train_loss, self.val_loss, self.class_loss = [], [], {
            n: [] for n in range(self.num_classes+1)}

    def plot_batch(self, tag, images):
        """
        Plot batches in grid
​
        Args: tag = identifier for plot (string)
              images = input batch (torch.tensor)
        """
        img_grid = torchvision.utils.make_grid(images, nrow=self.batch_size // 2)
        self.add_image(tag, img_grid)

    
    def plot_tumour(self, tag, image):
        
        fig = plt.figure(figsize=(24, 24))
        image=image.cpu()
        image=image.detach().numpy()
        image=image[0,:,:,:]
        image=image[:,:,80]
        ax = fig.add_subplot()
        #print(f"tag:{tag}")
        ax.imshow(image.T, cmap="viridis")
        ax.set_title("tumour")
        self.add_figure(str(tag), fig)

    def plot_scalar(self, name, value):
        self.add_scalar(name, value, self.epoch)

    def plot_confusion_matrix(self, cm, class_names):
      #function taken from https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
      print(type(cm))
      figure = plt.figure(figsize=(8,8))
      plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
      plt.title("Confusion Matrix")
      plt.colorbar()
      tick_marks = np.arange(len(class_names))
      plt.xticks(tick_marks, class_names, rotation=45)
      plt.yticks(tick_marks, class_names)

      #Normalize confusion matrix
      cm = np.around(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis],decimals=2)

      # Use white text if squares are dark; otherwise black.
      threshold = cm.max() / 2.

      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      self.add_figure(f"Confusion Matrix at epoch {self.epoch}", figure)

    def plot_histogram(self, tag, prediction):
        print('Plotting histogram')
        fig = plt.figure(figsize=(24, 24))
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, yticks=[], label='histogram')
            pred_norm = (prediction[idx, 0]-prediction[idx, 0].min())/(
                prediction[idx, 0].max()-prediction[idx, 0].min())
            ax.hist(pred_norm.cpu().flatten(), bins=100)
            ax.set_title(
                f'Prediction histogram @ epoch: {self.epoch} - idx: {idx}')
        self.add_figure(tag, fig)

    def per_class_loss(self, prediction, target, criterion, alpha=None):
        # Predict shape: (4, 1, 512, 512)
        # Target shape: (4, 1, 512, 512)
        #pred, target = prediction.cpu().numpy(), target.cpu().numpy()
        pred, target = prediction, target
        for class_ in range(self.num_classes + 1):
            class_pred, class_tgt = torch.where(
                target == class_, pred, torch.tensor([0], dtype=torch.float32).cuda()),  torch.where(target == class_, target, torch.tensor([0], dtype=torch.float32).cuda())

            #class_pred, class_tgt = pred[target == class_], target[target == class_] 
            if alpha is not None:
                loss = criterion(class_pred, class_tgt, alpha)
                #bce_loss, dice_loss = criterion(class_pred, class_tgt, alpha)
            else:
                loss = criterion(class_pred, class_tgt)
                #bce_loss, dice_loss = criterion(class_pred, class_tgt)
            #loss = bce_loss + dice_loss
            self.class_loss[class_].append(loss.item())

    def write_class_loss(self):
        for class_ in range(self.num_classes+1):
            self.add_scalar(f'Per Class loss for class {class_}', np.mean(self.class_loss[class_]), self.epoch)


class results :
    def __init__(self, expected, predicted) :
        self.expected = expected
        self.predicted = predicted

    def confusion_matrix(self):
        print(sklearn.metrics.confusion_matrix(self.expected, self.predicted))
        return sklearn.metrics.confusion_matrix(self.expected, self.predicted)
    
    def evaluate_results(self):
        self.true_positive_counter = 0
        self.true_negative_counter = 0
        self.false_positive_counter = 0
        self.false_negative_counter = 0
        for i in range(len(self.expected)) :
            if self.expected[i] == 1 and self.predicted[i] == 1 :
                self.true_positive_counter += 1
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> true positive')
            elif self.expected[i] == 0 and self.predicted[i] == 0 :
                self.true_negative_counter += 1
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> true negative')
            elif self.expected[i] == 0 and self.predicted[i] == 1 :
                self.false_positive_counter += 1 
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> false positive')
            elif self.expected[i] == 1 and self.predicted[i] == 0 :
                self.false_negative_counter += 1 
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> false negative')
        return self.true_positive_counter, self.true_negative_counter, self.false_positive_counter, self.false_negative_counter

transform = transforms.Compose(
    [transforms.ToTensor() ] #added 13/12/2021 to normalize the inputs. THIS NORMALIZES to mean = 0 and std = 1
)


class ImageDataset(Dataset) :
  def __init__(self, annotations, img_dir, transform = transform, target_transform = None, shift_augment = True, rotate_augment = True, scale_augment = True, flip_augment = True) :
    self.img_labels = annotations
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    self.shifts = shift_augment
    self.rotations = rotate_augment
    self.flips = flip_augment
    self.scales = scale_augment

  def __len__(self) :
    return len(self.img_labels)

  def __getitem__(self,idx) :
    img_path = os.path.join(self.img_dir, self.img_labels[idx][0] + "-GTV-1.nii" )
    image_sitk = sitk.ReadImage(img_path)
    # ID = self.img_labels[idx][0]
    # print(f'ID: {ID}')
    image = sitk.GetArrayFromImage(image_sitk)
    label = self.img_labels[idx][1]
    if self.transform :
      image = self.transform(image)
    if self.target_transform :
      label = self.target_transform(label)
    patient = []
    patient.append(self.img_labels[idx][0])
    #print(f"patient ID: {self.img_labels[idx][0]}")


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
    return image,label,patient
  
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
    
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.layer5 = nn.Conv3d(block_inplanes[3] * block.expansion, n_classes, 1, 1)
        self.avgpool2 = nn.AvgPool3d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.avgpool(x)

        #x = x.view(-1, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs).to(device)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

class CNN(nn.Module):

    def __init__(self):
      super(CNN, self).__init__()
      self.conv1 = nn.Conv3d(1,32,2,2)
      self.pool = nn.MaxPool3d(2,2)
      self.avg_pool = nn.AvgPool3d(10)
      self.conv2 = nn.Conv3d(32,128,2,2)
      self.conv3 = nn.Conv3d(128,64,1,1)
      self.conv4 = nn.Conv3d(64,16,1,1)
      self.conv5 = nn.Conv3d(16,2,1,1)

    # Defining the forward pass  (NIN method)  
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.avg_pool(self.conv5(x))
        x = x.view(-1,2)
        return x
        
#model = CNN().to(device) # Send the CNN to the device
model = generate_model(10)
#====================================================================
#=================== DEFIINING VARIABLES ============================
#====================================================================

check_day = 365 * 1.5 # This is defining the timeframe for which our CNN will consider the binary output (in days) 

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

patient_IDs, time_markers, dead_statuses = open_metadata()
dead_patient_array, alive_patient_array = patient_status_on_check_day(check_day)

#  Shuffle both arrays to ensure a random selection of patient data which will be input to the CNN
random.shuffle(dead_patient_array)
random.shuffle(alive_patient_array)

# Equalising the length of the 'dead' and 'alive' arrays so that we can ensure optimum training proportions
new_dead_patient_array, new_alive_patient_array = equalise_arrays(dead_patient_array, alive_patient_array)
equalised_array_length = len(new_alive_patient_array)

train_patients_dead = create_subgroup(new_dead_patient_array, equalised_array_length, 0.7)
train_patients_alive = create_subgroup(new_alive_patient_array, equalised_array_length, 0.7)

new_dead_patient_array = remove_same_elements(train_patients_dead, new_dead_patient_array)
new_alive_patient_array = remove_same_elements(train_patients_alive, new_alive_patient_array)

test_patients_dead = create_subgroup(new_dead_patient_array, equalised_array_length, 0.15)
test_patients_alive = create_subgroup(new_alive_patient_array, equalised_array_length, 0.15)

new_dead_patient_array = remove_same_elements(test_patients_dead, new_dead_patient_array)
new_alive_patient_array = remove_same_elements(test_patients_alive, new_alive_patient_array)

validate_patients_dead = create_subgroup(new_dead_patient_array, equalised_array_length, 0.15)
validate_patients_alive = create_subgroup(new_alive_patient_array, equalised_array_length, 0.15)

new_dead_patient_array = remove_same_elements(validate_patients_dead, new_dead_patient_array)
new_alive_patient_array = remove_same_elements(validate_patients_alive, new_alive_patient_array)

print(f"After separation into training, validation and testing arrays the number of unused data is {len(new_dead_patient_array) + len(new_alive_patient_array)}. If not then something has gone wrong.")

outcomes_train, outcomes_validate, outcomes_test = create_final_datasets()

training_data = ImageDataset(outcomes_train, os.path.join(project_folder, "textured_masks"), transform = transform)
validation_data = ImageDataset(outcomes_validate, os.path.join(project_folder, "textured_masks"), transform = transform)
test_data = ImageDataset(outcomes_test, os.path.join(project_folder, "textured_masks"), transform = transform) 


train_dataloader = DataLoader(training_data, batch_size = 4, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 4, shuffle = False)
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

for epoch in range(num_epochs):
    #send train loss and val loss to tb from here?
    writer.epoch = epoch
    epoch_validation_targets = []
    epoch_validation_predictions = []
    epoch_counter += 1
    avg_train_loss = np.append(avg_train_loss, training_loop())
    avg_valid_loss = np.append(avg_valid_loss, validation_loop())
    scheduler.step(avg_valid_loss[epoch])
    print(f"epoch_validation_targets = {epoch_validation_targets}")
    print(f"epoch_validation_predictions = {epoch_validation_predictions}")
    epoch_results = results(epoch_validation_targets, epoch_validation_predictions)
    conf_mat = epoch_results.confusion_matrix()
    print(f'(TP, TN, FP, FN): {epoch_results.evaluate_results()}')
    print(type(conf_mat))
    if (epoch+1)%5 == 0:
      writer.plot_confusion_matrix(conf_mat, ["alive","dead"])
    save_loss_plots()

print('FINISHED TRAINING')
print(f'All training batch losses = {all_training_losses}')
print(f'Training losses = {train_loss}')
print(f'Average training losses = {avg_train_loss}')
print(f'Validation losses = {avg_valid_loss}')

#===================== PLOT LOSS CURVES =============================
# new_avg_train_loss = avg_train_loss
# new_avg_valid_loss = avg_valid_loss

# epochs = np.array(range(num_epochs)) + 1
# loss_plot = plt.figure()
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# loss_plot.set_size_inches(20,10)
# plt.plot(epochs, new_avg_train_loss, label = 'Average training loss', linewidth = 7.0)
# plt.plot(epochs, new_avg_valid_loss, label = 'Average validation loss', linewidth = 7.0)
# loss_plot.legend(loc = 'best', prop={'size': 20})
# plt.ylabel('Average Loss', fontsize = 20)
# plt.xlabel('Epoch Number', fontsize = 20)
# plt.savefig(f'{plot_folder_path}{plot_filename}')
# print(f'The loss plot has been saved in: {plot_folder_path}{plot_filename}')

#===================== TESTING LOOP =================================
testing_accuracy = testing_loop()
print(f'Accuracy on testing set = {testing_accuracy:.1f}%')
writer.close()

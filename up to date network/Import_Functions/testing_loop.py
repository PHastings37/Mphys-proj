"""
This code includes the function for the testing loop that will
be imported into the testing code.

Rory Farwell and Patrick Hastings 24/03/2022
"""

import torch
from torch import reshape

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

def grad_cam_write(model, index, model_path, test_dataloader, writer, medcam, device):

    layer = f'layer{index}'
    model.load_state_dict(torch.load(model_path))
    model = medcam.inject(model, output_dir="medcam_test", 
    save_maps=True, layer=layer, replace=True)
    #print(medcam.get_layers(model))
    model.eval()
    


def testing_loop(model, test_dataloader, device, testing_targets, testing_predictions, writer, model_path, cam_save_path):
  print("---- Currently testing the network on unseen data ----")
  model.eval()
  import nibabel as nib
  import numpy as np
  import matplotlib.pyplot as plt
  from pytorch_grad_cam import GradCAM
  from pytorch_grad_cam.utils.image import show_cam_on_image
  from medcam import medcam

  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    counter = 0
    for images, labels, patients in test_dataloader :
      
      print(images.shape)
      print('===================')
      print(f'patient={patients[0][0]}')

      image = reshape(images, (images.shape[0],1,180,180,180))
      image = image.float()
      hot_labels = convert_to_one_hot_labels(image, labels)

      image = image.to(device)
      hot_labels = hot_labels.to(device)
      outputs = model(image)
      # max returns (value, index) 
      _,predictions = torch.max(outputs, 1)
      _,targets = torch.max(hot_labels,1)
      print(f'predictions:{predictions}')
      print(f'targets:{targets}')
      #print(f'predictions: {predictions}')
      #print(f'targets: {targets}')
      n_samples += hot_labels.shape[0]
      n_correct += (predictions == targets).sum().item()
      #print(f'n_correct = {n_correct}. n_samples = {n_samples}')

      labels_numpy = labels.numpy()

      for index in range(labels_numpy.size) :
            testing_targets.append(labels_numpy[index])

      predictions_numpy = predictions.cpu().numpy()

      for index in range(predictions_numpy.size) :
            testing_predictions.append(predictions_numpy[index])
      if predictions == targets:
        guess_status = "Correct"
      else:
        guess_status = "Incorrect"
      writer.plot_gradcam(images, patients, labels, model_path, model, device, guess_status, cam_save_path)
      acc = (100*n_correct)/n_samples
     
    
      
    
    return acc
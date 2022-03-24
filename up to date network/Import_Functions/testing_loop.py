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

# def testing_loop(model, test_dataloader, device, testing_targets, testing_predictions):
#   print("---- Currently testing the network on unseen data ----")
#   model.eval()

#   with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     counter = 0
#     for images, labels in test_dataloader :
#         # counter+=1
#         # print(counter)
#         images = reshape(images, (images.shape[0],1 ,160,160,160))
#         images = images.float()
#         hot_labels = convert_to_one_hot_labels(images, labels)

#         images = images.to(device)
#         hot_labels = hot_labels.to(device)
#         outputs = model(images)
#         # max returns (value, index) 
#         _,predictions = torch.max(outputs, 1)
#         _,targets = torch.max(hot_labels,1)
#         #print(f'predictions: {predictions}')
#         #print(f'targets: {targets}')
#         n_samples += hot_labels.shape[0]
#         n_correct += (predictions == targets).sum().item()
#         #print(f'n_correct = {n_correct}. n_samples = {n_samples}')

#         labels_numpy = labels.numpy()
#         # print(f"labels_numpy = {labels_numpy}")

#         for index in range(labels_numpy.size) :
#             testing_targets.append(labels_numpy[index])
        
#             # print(f"epoch_validation_targets = {epoch_validation_targets}")

#         predictions_numpy = predictions.cpu().numpy()
#         for index in range(predictions_numpy.size) :
#             testing_predictions.append(predictions_numpy[index])

#     acc = (100*n_correct)/n_samples

#     return acc

def testing_loop(model, test_dataloader, device, testing_targets, testing_predictions):
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

      labels_numpy = labels.numpy()

      for index in range(labels_numpy.size) :
            testing_targets.append(labels_numpy[index])

      predictions_numpy = predictions.cpu().numpy()
      for index in range(predictions_numpy.size) :
            testing_predictions.append(predictions_numpy[index])
    
    acc = (100*n_correct)/n_samples
    return acc
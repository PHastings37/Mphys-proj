"""
This file includes the training and validation loop functions that will be imported into the main network code.

Rory Farwell and Patrick Hastings 22/03/2022
"""
import torch 
from torch import reshape

def training_loop(epoch, model, train_dataloader, device, writer, criterion, optimizer, all_training_losses,
    num_epochs, n_total_steps, train_loss):
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
            # print(f"outputs: {outputs}")

    # Append the train_loss list with the total training loss for this epoch
    train_loss.append(epoch_train_loss)

    #Append the avg_train_loss list with the average training loss of this epoch
    avg_train_loss = epoch_train_loss/n_training_samples
    print(f"Average training loss list: {avg_train_loss}")

    print(f"Training loss array at end of epoch {epoch + 1}: {train_loss}. Total number of images used = {n_training_samples}.")
    print(f"Finished training for epoch {epoch + 1}")
    
    writer.plot_scalar("Train Loss", avg_train_loss)
    return avg_train_loss

def validation_loop(epoch, model, validation_dataloader, device, criterion, epoch_validation_targets, 
    epoch_validation_predictions, writer) :
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
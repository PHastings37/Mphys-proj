"""
This file contains the tensorboard writer class that will be imported into the main network code

Rory Farwell and Patrick Hastings 22/03/2022
"""

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
import itertools
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from medcam import medcam
import nibabel as nib
from torch.utils.data import DataLoader

class customWriter(SummaryWriter):
    def __init__(self, log_dir, batch_size, epoch, num_classes, test_run):
        super(customWriter, self).__init__()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_classes = num_classes
        self.train_loss = []
        self.val_loss = []
        self.class_loss = {n: [] for n in range(num_classes+1)}
        self.test_run = test_run
        
    
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
        if self.test_run == True:
            return
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
        if self.test_run == True:
            return
        
        self.add_scalar(name, value, self.epoch)
        

    def plot_confusion_matrix(self, cm, class_names):
        #function taken from https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
        if self.test_run == True:
            return
        
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
      

    def plot_gradcam(self, layer, test_dataloader, model_path, model, device):
        print(f"layer{layer}")
        image, label, pid = next(iter(test_dataloader))
        filename = pid[0][0]
        image = image[None].to(device, torch.float)
        attn = model(image)

        attn = np.squeeze(attn.cpu().numpy())
        img = np.squeeze(image.cpu().numpy())
        print(img.shape, attn.shape)
        slice_num = 80

        fig, ax = plt.subplots(1,1, figsize=(10,10))
        img1 = nib.Nifti1Image(img, np.eye(4))
        img2 = nib.Nifti1Image(attn, np.eye(4))
        img1.header.get_xyzt_units()
        img1.to_filename(f"{layer}_mask.nii")
        img2.header.get_xyzt_units()
        img2.to_filename(f"{layer}_attn.nii")
        im = img[..., slice_num]
        attn = attn[..., slice_num]

        print(pid)
        print(attn.max(), attn.min())
        ax.imshow(im, cmap='gray')
        ax.imshow(attn, cmap='jet', alpha=0.5)
        filename="./test0.png"
        fig.savefig(filename)   
        self.add_figure(f"{layer}_{pid}", fig)

    def plot_histogram(self, tag, prediction):
        if self.test_run == True:
            return

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

''' 
    utils 

    - device
    - path
    - functions
'''

## import
import os
import random
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


## device
def get_device():
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device


## path
def get_data_model_path(root_path):
    data_path = os.path.join(root_path, 'dataset')
    model_path = os.path.join(root_path, 'model')

    return data_path, model_path


## function

# set random seed (for reproducibility)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# image read
def img_read(src, file):
    img_path = os.path.join(src, file)
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY) # grayscale
    return img

# image read/plot
def img_read_plot(src, file):
    img_path = os.path.join(src, file)
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY) # grayscale

    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return img


# plot loss history
def plot_loss(loss_history, logy=False, title=None):

    if logy:
        plt.semilogy(loss_history)
    else:
        plt.plot(loss_history)

    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training data'], loc=0)
    plt.show()


## metric
# RMSE
def plot_rmse(rmse_tr_hist, rmse_val_hist, title=None):

    plt.plot(rmse_tr_hist)
    plt.plot(rmse_val_hist)

    if title is not None:
        plt.title(title)

    plt.ylabel('RMSE')
    plt.xlabel('Epochs')
    plt.legend(['Training data', 'Validation data'], loc=0)
    plt.show()

# MAPE
def plot_mape(mape_tr_hist, mape_val_hist, title=None):

    plt.plot(mape_tr_hist)
    plt.plot(mape_val_hist)

    if title is not None:
        plt.title(title)
        
    plt.ylabel('MAPE')
    plt.xlabel('Epochs')
    plt.legend(['Training data', 'Validation data'], loc=0)
    plt.show()
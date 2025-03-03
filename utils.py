''' 
    utils 

    - device
    - path
    - functions
'''

# import
import os
import random
import numpy as np
import torch

from sklearn.model_selection import train_test_split

import cv2
import matplotlib.pyplot as plt


''' setting '''
# device
def get_device():
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device

# path
def get_path(root_path):
    data_path = os.path.join(root_path, 'data')
    output_path = os.path.join(root_path, 'output')

    return data_path, output_path

# random seed (for reproducibility)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# settings
def set_all(root_path):
    device = get_device()
    data_path, output_path = get_path(root_path)
    set_seed()

    return device, data_path, output_path



''' image '''
# image read
def img_read(src, file):
    img_path = os.path.join(src, file)
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY) # grayscale
    return img



''' load dataset '''
# load
def load_data(data_path, dataset_name):
    src = os.path.join(data_path, dataset_name)
    files = os.listdir(src)

    X,Y = [],[]

    for file in files:
        X.append(img_read(src, file))
        Y.append(float(file[:-4]))      # label <- file name

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# split + @
def split_data(X, Y, test_size, device, flatten=False, scaler=None):
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X,Y, test_size=test_size, random_state=1, shuffle=True)

    # reshape
    if flatten:
        X_tr = X_tr.reshape(-1, np.prod(X_tr.shape[1:])) # (N, 56, 56) -> (N, 56*56)ㄴ
        X_ts = X_ts.reshape(-1, np.prod(X_ts.shape[1:]))
    else:
        X_tr = np.expand_dims(X_tr, axis=1) # (N, 56, 56) -> (N, 1, 56, 56) : channel 추가
        X_ts = np.expand_dims(X_ts, axis=1)

    Y_tr = np.expand_dims(Y_tr, axis=1) # (N,) -> (N, 1)
    Y_ts = np.expand_dims(Y_ts, axis=1)

    # normalization (Y:0~1)
    if scaler:
        Y_tr = scaler.fit_transform(Y_tr)
        Y_ts = scaler.transform(Y_ts)

    # convert to Tensor
    X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(device)
    X_ts = torch.tensor(X_ts, dtype=torch.float32).to(device)
    Y_ts = torch.tensor(Y_ts, dtype=torch.float32).to(device)

    return X_tr, X_ts, Y_tr, Y_ts

# check
def check_data(X, Y, nplot=5):
    fig = plt.figure(figsize=(15,5))

    for i in range(nplot):
        ax = fig.add_subplot(1, nplot+1, i+1)
        ax.imshow(X[i,:,:], cmap='gray')
        ax.set_title(f"Y={Y[i]:.2f}")
        ax.axis('off')
    plt.show()



''' train/test '''
# training (loss only)
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# evaluating (loss only)
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()

    return total_loss / len(test_loader)


# # training
# def train(model, train_loader, criterion, optimizer, scaler=None):
#     model.train()

#     loss_tr = 0.0
#     rmse_tr = 0.0
#     mape_tr = 0.0

#     for x, y in train_loader:
#         output = model(x)
#         loss = criterion(output, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # metrics
#         loss_tr += loss.item()

#         # inverse transform (for calc metrics)
#         if scaler:
#             output = output.cpu().detach().numpy().reshape(1,-1)
#             output = scaler.inverse_transform(output)
#             output = torch.from_numpy(output).clone().detach()

#             y = y.cpu().detach().numpy().reshape(1,-1)
#             y = scaler.inverse_transform(y)
#             y = torch.from_numpy(y).clone().detach()

#         rmse_tr += mean_squared_error(output, y, squared=False).item()
#         mape_tr += mean_absolute_percentage_error(output, y).item() * 100

#     avg_loss_tr = loss_tr / len(train_loader)
#     avg_rmse_tr = rmse_tr / len(train_loader)
#     avg_mape_tr = mape_tr / len(train_loader)

#     return avg_loss_tr, avg_rmse_tr, avg_mape_tr


# # evaluating
# def evaluate(model, test_loader, criterion, scaler=None):
#     model.eval()

#     loss_val = 0.0
#     rmse_val = 0.0
#     mape_val = 0.0

#     with torch.no_grad():
#         for x, y in test_loader:
#             output = model(x)
#             loss = criterion(output, y)

#             # metrics
#             loss_val += loss.item()

#             # inverse transform (for calc metrics)
#             if scaler:
#                 output = output.cpu().detach().numpy().reshape(1,-1)
#                 output = scaler.inverse_transform(output)
#                 output = torch.from_numpy(output).clone().detach()

#                 y = y.cpu().detach().numpy().reshape(1,-1)
#                 y = scaler.inverse_transform(y)
#                 y  = torch.from_numpy(y).clone().detach()

#             rmse_val += mean_squared_error(output, y, squared=False).item()
#             mape_val += mean_absolute_percentage_error(output, y).item() * 100

#         avg_loss_val = loss_val / len(test_loader)
#         avg_rmse_val = rmse_val / len(test_loader)
#         avg_mape_val = mape_val / len(test_loader)

#     return avg_loss_val, avg_rmse_val, avg_mape_val



''' plot history'''
# metrics hist
def plot_hist(tr_hist, val_hist=None, title=None, legend=None): # loss, RMSE, MAPE ...

    plt.semilogy(tr_hist)
    if val_hist:
        plt.semilogy(val_hist)

    if title:
        plt.title(title)
    
    if legend:
        plt.legend(legend)

    plt.show()











''' metric for torchmetrics vs sklearn '''
# torchmetrics
# in:   Tensor
# out:  Tensor (Tensor.item() => float)
'''
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_percentage_error, r2_score

dnn.train()
total_loss = 0.0
rmse_tr = 0.0
mape_tr = 0.0

for x, y in train_loader:
    optimizer.zero_grad()
    output = dnn(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    # inverse transformation (for calc metrics)
    output = output.cpu().detach().numpy().reshape(1,-1)    # tensor -> numpy
    output = scaler.inverse_transform(output)               # inverse transform
    output = torch.from_numpy(output).clone().detach()      # numpy -> tensor
    
    y = y.cpu().detach().numpy().reshape(1,-1)
    y = scaler.inverse_transform(y)
    y = torch.from_numpy(y).clone().detach()

    rmse_tr += mean_squared_error(output, y, squared=False).item() # only Tensor / item() : tensor -> float
    mape_tr += 100*mean_absolute_percentage_error(output, y).item()
'''

# sklearn
# in:   MatrixLike (Tensor, numpy)
# out:  float
'''
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

dnn.train()
total_loss = 0.0
rmse_tr = 0.0
mape_tr = 0.0

for x, y in train_loader:
    optimizer.zero_grad()
    output = dnn(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    # inverse transformation (for calc metrics)
    output = output.cpu().detach().numpy().reshape(1,-1)    # tensor -> numpy
    output = scaler.inverse_transform(output)               # inverse transform
    
    y = y.cpu().detach().numpy().reshape(1,-1)
    y = scaler.inverse_transform(y)

    rmse_tr += mean_squared_error(output, y).item()         # both Tensor/numpy
    mape_tr += 100*mean_absolute_percentage_error(output, y).item()
'''
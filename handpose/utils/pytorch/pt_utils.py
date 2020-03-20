import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset


def conv1_len_out(len_in, kernel_size, stride=1, padding=0, dilation=1):
    len_out = (len_in+2*padding-dilation*(kernel_size-1)-1)/stride + 1
    return int(len_out)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def evaluate(model, criterion, loader, device):

    model.eval()   # Set model to evaluate mode

    is_tuple = False
    if type(loader.dataset[0]) is tuple:
        is_tuple = True

    num_samples = 0
    corrects = 0
    loss = 0.0
    with torch.no_grad():
        for samples in loader:
            if is_tuple:
                data, target = samples[0].to(device), samples[1].to(device)
            else:
                data, target = samples["data"].to(device), samples["target"].to(device)

            batch_size = target.size(0)
            output = model(data)
            batch_loss = criterion(output, target)
            _, predicted = torch.max(output, 1)

            num_samples += batch_size
            loss += batch_loss.item() * batch_size
            corrects += (predicted == target).sum().item()

    loss /= num_samples
    acc = float(corrects) / num_samples

    return loss, acc

def evaluate_v0(model, criterion, loader, device):

    model.eval()   # Set model to evaluate mode

    num_samples = 0
    corrects = 0
    loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)

            output = model(data)
            batch_loss = criterion(output, target)
            _, predicted = torch.max(output, 1)

            num_samples += batch_size
            loss += batch_loss.item() * batch_size
            corrects += (predicted == target).sum().item()

    loss /= num_samples
    acc = float(corrects) / num_samples

    return loss, acc

def predict(data, batch_size, model, device):

    model.eval()   # Set model to evaluate mode
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         shuffle=False, num_workers=32, pin_memory=True)

    proba = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)

            output = model(batch_data)
            p =  torch.exp(output)
            proba.append(p.cpu().numpy())

    proba = np.concatenate(proba, axis=0)

    return proba


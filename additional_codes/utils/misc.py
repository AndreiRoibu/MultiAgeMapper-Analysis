import os
import numpy as np
import torch.nn as nn

def create_folder(path):

    if not os.path.exists(path):
        os.mkdir(path)

def mae(predicted, actual):
    
    absolute_error = np.abs(np.subtract(predicted, actual))

    if absolute_error.shape[0] == 1:
        return absolute_error
    else:
        return np.mean(absolute_error)

def my_KLDivLoss(prediction, target):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """

    prediction.cpu()
    target.cpu()

    loss_func = nn.KLDivLoss(reduction='sum')
    target += 1e-16
    target_shape = target.shape[0]
    loss = loss_func(prediction, target) / target_shape
    return loss
import torch
import numpy as np



####Shift mode operators#####
def A_torch_withshift(x, Phi, shift_func, shift_back_func):
    """
    Forward operator, both x and Phi have been shifted
    :param x:
    :param Phi:
    :param shift_back_func: Reverse shift operator
    :return:
    """
    y = torch.sum(x * Phi, dim=1)  # cassi_measurement
    return y
def At_torch_withshift(y,Phi,shift_func,shift_back_func):
    """
    :param y: [y1,y2] two observations concatenated vertically
    :param Phi: shifted Phi
    :param shift_func: forward shift operator
    :return:
    """
    y = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = y * Phi
    return x

####Unshift mode operators#####
def A_torch(x, Phi, shift_func, shift_back_func):
    """
    Dual-camera forward operator, both x and Phi have been shifted
    :param x:
    :param Phi:
    :return:
    """
    x_shift = shift_func(x)  # Compute using the CUDA version of shift_func
    y = torch.sum(x_shift * Phi, dim=1)  # cassi_measurement
    return y
def At_torch(y,Phi,shift_func,shift_back_func):
    """
    :param y: [y1,y2] two observations concatenated vertically
    :param Phi: shifted Phi
    :param shift_func: forward shift operator
    :return:
    """
    y = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)

    x = y * Phi
    x = shift_back_func(x)
    return x

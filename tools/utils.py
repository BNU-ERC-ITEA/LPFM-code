import cv2
import numpy as np
import os
from pathlib import Path
import torch
from tools.ssim_torch import ssim
import time
import functools
import logging
import os

join = os.path.join

def omake(path):
    if os.path.isdir(path):
        os.makedirs(path,exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path),exist_ok=True)
    return path

def joinc(*args)->str:
    # Create folder and join path
    file = join(*args)
    omake(file)
    return file
    return file
def glob(folder:str,pattern:str="*.mat")->list:
    return [str(p) for p in Path(folder).glob(pattern)]
def rglob(folder:str,pattern:str="*.mat")->list:
    return [str(p) for p in Path(folder).rglob(pattern)]



def tictoc(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

import datetime
def get_logger(log_name='my_logger',file_name='train.log'):
    logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='## %Y-%m-%d %H:%M:%S')
    log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(file_name, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(file_handler)

    now = datetime.datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"begining at {formatted_time}")
    return logger

def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*255).round()
    ref = (ref*255).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
def torch_rmse(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: (C, H, W)
    return: scalar
    """
    assert pred.shape == gt.shape
    return torch.sqrt(torch.mean((pred - gt) ** 2))
def torch_ergas(pred: torch.Tensor, gt: torch.Tensor, eps=1e-8):
    """
    pred, gt: (C, H, W)
    return: scalar
    """
    assert pred.shape == gt.shape
    C = gt.shape[0]

    # per-band RMSE
    rmse_c = torch.sqrt(torch.mean((pred - gt) ** 2, dim=(1, 2)))  # (C,)
    mean_c = torch.mean(gt, dim=(1, 2))                            # (C,)

    ergas = 100 * torch.sqrt(torch.mean((rmse_c / (mean_c + eps)) ** 2))
    return ergas

def torch_sam(pred: torch.Tensor, gt: torch.Tensor, eps=1e-8):
    """
    pred, gt: (C, H, W)
    return: scalar (radian)
    """
    assert pred.shape == gt.shape

    # (H, W, C)
    pred = pred.permute(1, 2, 0)
    gt   = gt.permute(1, 2, 0)

    dot = torch.sum(pred * gt, dim=-1)
    norm_pred = torch.norm(pred, dim=-1)
    norm_gt   = torch.norm(gt, dim=-1)

    cos = dot / (norm_pred * norm_gt + eps)
    cos = torch.clamp(cos, -1.0, 1.0)

    sam = torch.mean(torch.acos(cos))
    return sam
def touint8img(img):
    img = img/np.max(img)
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

import pdb
pdb_flag = False
def setpdb():
    if pdb_flag:
        pdb.set_trace()
    else:
        pass

import math
def psnr_block(ref, img,normto1=False):
    h,w,c = img.shape
    if normto1:
        for i in range(c):
            img[:, :, i] = img[:, :, i] / np.max(img[:, :, i])
    img = (img * 256).round()
    ref = (ref * 256).round()

    psnr = 0
    for i in range(c):
        mse = np.mean((img[ :, :,i] - ref[ :, :,i]) ** 2)
        psnr += 10 * math.log10((255 * 255) / mse)
    return psnr / c

def psnr_torch(ref, img):
    """
    Assume it is a 4D tensor
    :param ref:
    :param img:
    :return:
    """
    img = img[0]
    ref = ref[0]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * math.log10((255 * 255) / mse)
    return psnr / nC
def torch2np(x:torch.Tensor):
    x = x.cpu().detach().numpy()
    if len(x.shape) == 3:
        x = np.transpose(x,(1,2,0))
    if len(x.shape) == 4:
        x = np.transpose(x, (0,2, 3, 1))
    return x
def numpy2torch(x:np.ndarray):
    if len(x.shape) == 2:
        x = torch.tensor(x)
        return torch.tensor(x).unsqueeze(0).unsqueeze(0)
    if len(x.shape) == 3:
        x = torch.tensor(x)
        x = np.transpose(x, (2,0,1))
        return torch.tensor(x).unsqueeze(0)
    if len(x.shape) == 4:
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.tensor(x)
        return x
    else:
        return torch.tensor(x)

def solver_wrapper(func):
    def wrapped_func(self, *args, **kwargs):
        tic = time.time()
        ret =  func(self, *args, **kwargs)
        toc = time.time()
        cost_time = toc-tic
        itr = self.itr if hasattr(self,"itr") else -1
        print(f"{func.__name__}_cost={cost_time},iter_number={itr}")
        if hasattr(self,"cost_time"):
            self.cost_time = cost_time
        return ret
    return wrapped_func
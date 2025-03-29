import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
    
    def forward(self, input, target):
        return F.l1_loss(input, target)

class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, input, target):
        return self.loss(input, target)

class LossSmoothL1(nn.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, input, target):
        return self.loss(input, target)

class LossCrossEntropy(nn.Module):
    def __init__(self):
        super(LossCrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        return self.loss(input, target)

class L1LossMu(nn.Module):
    def __init__(self, mu=5000):
        super(L1LossMu, self).__init__()
        self.mu = mu
    
    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return F.l1_loss(mu_pred, mu_label)

def compute_losses(data, endpoints, params):
    #loss = {}
    if params['loss_type'] == "l1_loss_mu":
        criterion = L1LossMu()
        pred = endpoints["p"]
        label = data["label"]
        loss = criterion(pred, label)
    else:
        raise NotImplementedError
    return loss

def batch_psnr(img, imclean, data_range):
    Img = img.cpu().numpy().astype(np.float32)
    Iclean = imclean.cpu().numpy().astype(np.float32)
    #print('a', Img.shape)
    #print(Iclean.shape)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i], Img[i], data_range=data_range)
        #print('a', peak_signal_noise_ratio(Iclean[i], Img[i], data_range=data_range))
    return psnr / Img.shape[0]

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor(img)
    imclean = range_compressor(imclean)
    return batch_psnr(img, imclean, data_range)

def compute_metrics(data, endpoints, manager):
    metrics = {}
    pred = endpoints['p']
    label = data['label']
    psnr = batch_psnr(pred, label, data_range=1.0)
    #psnr_mu = batch_psnr_mu(pred, label, data_range=1.0)
    metrics['psnr'] = torch.tensor(psnr)
    #metrics['psnr_mu'] = 0#torch.tensor(psnr_mu)
    #metrics_1 = torch.tensor(psnr)
    #metrics_2 = torch.tensor(psnr_mu)
    return metrics

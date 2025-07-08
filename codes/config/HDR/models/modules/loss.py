import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys


class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')
    
    # Input images must be in range [0, 1] and shape (N, C, H, W)
    def ssim_loss(self, img1, img2):
    	return 1 - ssim(img1, img2, data_range=1.0, size_average=True)

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        
        loss_ssim = self.ssim_loss(predict, target)
        composite_loss = loss + 0.05 * loss_ssim 

        if self.is_weighted and weights is not None:
            composite_loss = weights * composite_loss

        return composite_loss.mean()



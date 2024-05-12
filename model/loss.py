import torch
from torch import nn
from torch.nn import functional as F
from hparams import hparams as hps
import numpy as np
import time

class MercuryNetLoss(nn.Module):
    def __init__(self):
        super(MercuryNetLoss, self).__init__()

    def forward(self, model_output, targets):
        model_output = model_output[:, 10:80, :]
        targets = targets[:, 10:80, :]
        target_f0, target_voiced, target_amp = targets[:,:,0], targets[:,:,1], targets[:,:,2]
        output_f0, output_voiced, output_amp = model_output[:,:,0], model_output[:,:,1], model_output[:,:,2]
        output_f0 = output_f0 * target_voiced
        target_f0 = target_f0 * target_voiced
        output = F.mse_loss(output_f0, target_f0) + F.mse_loss(output_voiced, target_voiced) + F.mse_loss(output_amp, target_amp)
        return output
    
class HumanReadableLoss(nn.Module):
    def __init__(self):
        super(HumanReadableLoss, self).__init__()
    
    def forward(self, model_output, targets):
        targets = targets.detach()
        model_output = model_output.detach()

        nan_to_num(targets, 1e-12)
        
        target_f0, target_voiced, target_amp = targets[:,:,0], targets[:,:,1], targets[:,:,2]
        output_f0, output_voiced, output_amp = model_output[:,:,0], model_output[:,:,1], model_output[:,:,2]
        
        # only care about f0 when it's voiced
        target_f0[target_voiced <= 0] = 1e-12
        output_f0[target_voiced <= 0] = 1e-12
   
        target_f0[target_f0 <= 0] = 1e-12
        target_amp[target_amp <= 0] = 1e-12
        output_f0[output_f0 <= 0] = 1e-12
        output_amp[output_amp <= 0] = 1e-12
        
        # in octaves
        target_f0 = torch.log(target_f0) / np.log(2)
        output_f0 = torch.log(output_f0) / np.log(2)

        # in decibels
        target_amp = 10 * torch.log10(target_amp)
        output_amp = 10 * torch.log10(output_amp)
        
        # equalize baselines (only measure relative f0 change)
        first_voiced_idx = torch.min(torch.where(target_voiced == 1)[1])
        target_f0 -= target_f0[:,first_voiced_idx].unsqueeze(1)
        output_f0 -= output_f0[:,first_voiced_idx].unsqueeze(1)

        f0_err = (output_f0 - target_f0)**2
        voiced_err = (target_voiced - output_voiced)**2
        amp_err = (target_amp - output_amp)**2

        all_stats = [(np.mean(err), np.quantile(err, 0.5), np.quantile(err, 0.90), np.quantile(err, 0.95), np.max(err))
                     for err in (f0_err.cpu().numpy() , voiced_err.cpu().numpy() , amp_err.cpu().numpy() )]
     
        return all_stats
    
    

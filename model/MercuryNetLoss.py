import torch
from torch import nn
from torch.nn import functional as F
from hparams import hparams as hps

def nan_to_num(x, replacement = 0):
    nan_mask = torch.isnan(x)
    x[nan_mask] = replacement

class MercuryNetLoss(nn.Module):
    def __init__(self):
        super(MercuryNetLoss, self).__init__()

    def forward(self, model_output, targets):
        nan_to_num(targets, 1e-12)
        
        target_f0, target_voiced, target_amp = targets[:,:,0], targets[:,:,1], targets[:,:,2]
        output_f0, output_voiced, output_amp = model_output[:,:,0], model_output[:,:,1], model_output[:,:,2]
        
        # only care about f0 when it's voiced (so there's a valid ground truth)
        target_f0[target_voiced <= 0] = 1e-12
        output_f0[target_voiced <= 0] = 1e-12
   
        target_f0[target_f0 <= 0] = 1e-12
        target_amp[target_amp <= 0] = 1e-12
        output_f0[output_f0 <= 0] = 1e-12
        output_amp[output_amp <= 0] = 1e-12
        
        target_f0 = torch.log(target_f0)
        output_f0 = torch.log(output_f0)                
        target_amp = torch.log(target_amp)
        output_amp = torch.log(output_amp)
        
        # equalize baselines (only measure relative f0 change) - using large number as "zero" to avoid negative log
        first_voiced_idx = torch.min(torch.where(target_voiced == 1)[1])
        target_f0 -= target_f0[:,first_voiced_idx].unsqueeze(1)
        output_f0 -= output_f0[:,first_voiced_idx].unsqueeze(1)

        loss = 0
        loss += hps.f0_penalty * F.mse_loss(output_f0, target_f0)
        loss += hps.voiced_penalty * F.mse_loss(target_voiced, output_voiced)
        loss += hps.amp_penalty * F.mse_loss(target_amp, output_amp)
        print(loss.detach().cpu(), F.mse_loss(output_f0, target_f0).detach().cpu(), F.mse_loss(target_voiced,      output_voiced).detach().cpu(), F.mse_loss(target_amp, output_amp).detach().cpu())

        if torch.sum(torch.isnan(loss)) != 0:
            print(model_output)
            print(targets)
            print(torch.sum(torch.isnan(target_f0)))
            print(torch.sum(torch.isnan(output_f0)))
            print(torch.sum(torch.isnan(target_amp)))
            print(torch.sum(torch.isnan(output_amp)))
            print(torch.sum(torch.isnan(target_voiced)))
            print(torch.sum(torch.isnan(output_voiced)))
     
        return loss
    
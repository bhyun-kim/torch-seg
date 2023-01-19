


import torch
import torch.nn as nn

from library import LossRegistry


@LossRegistry.register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module):
    """
    """
    def __init__(self,
                weight=None,
                reduction='mean',
                ignore_idx=255):
        super().__init__()

        if weight: 
            weight = torch.tensor(weight)
        
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_idx, reduction='none')

        self.reduction = reduction
        self.ignore_idx = ignore_idx
        
    def forward(self, outputs, targets):
        
        loss = self.loss(outputs, targets)

        if self.reduction == 'mean':
            num_element = targets.numel() - (targets == self.ignore_idx).sum().item()
            loss = loss.sum() / num_element

        return loss
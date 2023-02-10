from copy import deepcopy

import torch
import torch.nn as nn

from tools.library import LossRegistry


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
        
        self.loss = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_idx, 
            reduction='sum'
            )

        self.reduction = reduction
        self.ignore_idx = torch.tensor(ignore_idx)
        
    def forward(self, outputs, targets):

        num_element = targets.numel() - (targets == self.ignore_idx).sum().item()
        
        loss = self.loss(outputs, targets)

        if self.reduction == 'mean':
            
            # loss = loss.sum() / num_element
            loss = loss / num_element

        return loss
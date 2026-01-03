import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for extreme values. 
                   Weight = 1 + alpha * |target|
            reduction: 'mean' or 'sum'
        """
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # Calculate standard squared error
        mse = (pred - target) ** 2
        
        # Calculate weight based on target magnitude (absolute deviation from mean since data is normalized)
        # Assuming data is normalized to ~N(0, 1) or [-1, 1] centered at 0.
        weight = 1 + self.alpha * torch.abs(target)
        
        loss = mse * weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalMSELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        """
        Focal Mean Squared Error Loss.
        Args:
            alpha: Scaling factor.
            gamma: Focusing parameter. Higher gamma focuses more on hard examples (large errors).
        """
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        loss = (error ** 2) * (1 + self.alpha * (error ** self.gamma))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction = "mean", epsilon = 1e-8):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        pt = pt.clamp(min=self.epsilon, max=1 - self.epsilon)
        F_loss = self.alpha*(1-pt)**self.gamma * BCE_loss

        # Return the loss (mean, sum, or none depending on the reduction type)
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
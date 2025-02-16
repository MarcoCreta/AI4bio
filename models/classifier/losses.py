import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        :param alpha: A tensor of size (C,) where C is the number of classes. It can represent the class weights.
        :param gamma: Focal loss focusing parameter (default is 2)
        :param reduction: Type of reduction to apply to the loss ('mean', 'sum', or 'none')
        """
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        :param inputs: Raw logits from the model (before applying sigmoid or softmax)
        :param targets: Ground truth labels (can be multi-hot encoded for multi-label problems)
        """
        # For multi-label tasks, we apply sigmoid to get probabilities
        # For multi-class tasks, use softmax
        probs = torch.sigmoid(inputs) if targets.size(1) > 1 else torch.softmax(inputs, dim=1)

        # Compute BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # If alpha is provided, make sure it has the same size as the number of classes

        alpha = self.alpha

        # Compute pt: probability of the true class (or the class we care about)
        pt = torch.exp(-BCE_loss)

        # Apply focal loss formula
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        # Return the loss (mean, sum, or none depending on the reduction type)
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

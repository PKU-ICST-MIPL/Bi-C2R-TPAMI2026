import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def KnowledgeDistillation(scores, target_scores, T=2., return_score=False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    # log_scores_norm = torch.cat([F.log_softmax(score / T, dim=1) for score in scores], dim=1)
    # targets_norm = torch.cat([F.softmax(target_score / T, dim=1) for target_score in target_scores], dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)
        # log_scores_norm = log_scores_norm[:, :target_scores.size(1)]

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    kd_loss_unnorm = -(targets_norm * log_scores_norm)
    kd_loss_unnorm = kd_loss_unnorm.sum(dim=1)  # --> sum over classes
    kd_loss_unnorm = kd_loss_unnorm.mean()  # --> average over batch

    # normalize
    kd_loss = kd_loss_unnorm * T ** 2

    if not return_score:
        return kd_loss
    else:
        return kd_loss, targets_norm.max(dim=1)[0].mean()

# ------------------------------------------------------------------------
# Modified from UniMoCo (https://github.com/dddzg/unimoco)
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Definition of the Supervised Contrastive Loss and Causal Losses
"""
from torch import nn
import torch
import torch.nn.functional as F


class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class CausalConsistencyLoss(nn.Module):
    """
    Causal Consistency Loss: Ensures the model's predictions are consistent 
    between original and counterfactual samples.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_original, logits_cf):
        """
        Calculates the consistency loss between original and counterfactual predictions.

        Args:
            logits_original: Predictions for original samples [B, C]
            logits_cf: Predictions for counterfactual samples [B, C]
        """
        # Use KL divergence to measure the difference in predictive distributions
        p_original = F.softmax(logits_original / self.temperature, dim=1)
        p_cf = F.softmax(logits_cf / self.temperature, dim=1)

        # Calculate KL divergence
        loss_kl = F.kl_div(
            torch.log(p_cf + 1e-10),
            p_original,
            reduction='batchmean'
        )

        return loss_kl


class CausalInvarianceLoss(nn.Module):
    """
    Causal Invariance Loss: Ensures predictions on counterfactual samples 
    remain invariant to the original samples.
    """

    def __init__(self, mode='js', temperature=1.0):
        super().__init__()
        self.mode = mode  # 'kl' or 'js'
        self.temperature = temperature

    def forward(self, logits_orig, logits_cf):
        """
        Calculates the causal invariance loss between original and counterfactual predictions.

        Args:
            logits_orig: Predictions for original samples [B, C]
            logits_cf: Predictions for counterfactual samples [B, C]
        """
        p_orig = F.softmax(logits_orig / self.temperature, dim=1)
        p_cf = F.softmax(logits_cf / self.temperature, dim=1)

        if self.mode == 'kl':
            # KL Divergence: KL(p_cf||p_orig)
            loss = F.kl_div(torch.log(p_cf + 1e-10), p_orig, reduction='batchmean')

        elif self.mode == 'js':
            # JS Divergence: 0.5*KL(p_orig||p_mix) + 0.5*KL(p_cf||p_mix)
            p_mix = 0.5 * (p_orig + p_cf)
            loss_1 = F.kl_div(torch.log(p_mix + 1e-10), p_orig, reduction='batchmean')
            loss_2 = F.kl_div(torch.log(p_mix + 1e-10), p_cf, reduction='batchmean')
            loss = 0.5 * (loss_1 + loss_2)

        return loss

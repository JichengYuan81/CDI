# ------------------------------------------------------------------------
# Modified from UniMoCo (https://github.com/dddzg/unimoco)
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Definition of the Supervised Contrastive Loss
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
    """因果一致性损失：确保模型对原始样本和反事实样本的预测一致"""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_original, logits_cf):
        """
        计算原始预测和反事实预测之间的一致性损失

        logits_original: 原始样本的预测 [B, C]
        logits_cf: 反事实样本的预测 [B, C]
        """
        # 使用KL散度衡量预测分布的差异
        p_original = F.softmax(logits_original / self.temperature, dim=1)
        p_cf = F.softmax(logits_cf / self.temperature, dim=1)

        # 计算KL散度
        loss_kl = F.kl_div(
            torch.log(p_cf + 1e-10),
            p_original,
            reduction='batchmean'
        )

        return loss_kl


class CausalInvarianceLoss(nn.Module):
    """因果不变性损失：确保对反事实样本的预测与原始样本一致"""

    def __init__(self, mode='js', temperature=1.0):
        super().__init__()
        self.mode = mode  # 'kl'或'js'
        self.temperature = temperature

    def forward(self, logits_orig, logits_cf):
        """
        计算原始预测和反事实预测之间的因果不变性损失

        Args:
            logits_orig: 原始样本的预测 [B, C]
            logits_cf: 反事实样本的预测 [B, C]
        """
        p_orig = F.softmax(logits_orig / self.temperature, dim=1)
        p_cf = F.softmax(logits_cf / self.temperature, dim=1)

        if self.mode == 'kl':
            # KL散度: KL(p_cf||p_orig)
            loss = F.kl_div(torch.log(p_cf + 1e-10), p_orig, reduction='batchmean')

        elif self.mode == 'js':
            # JS散度: 0.5*KL(p_orig||p_mix) + 0.5*KL(p_cf||p_mix)
            p_mix = 0.5 * (p_orig + p_cf)
            loss_1 = F.kl_div(torch.log(p_mix + 1e-10), p_orig, reduction='batchmean')
            loss_2 = F.kl_div(torch.log(p_mix + 1e-10), p_cf, reduction='batchmean')
            loss = 0.5 * (loss_1 + loss_2)

        return loss


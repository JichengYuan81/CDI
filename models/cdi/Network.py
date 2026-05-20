import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from .attention import SelfAttention
from .causal import CausalSCM  # 新增导入


class MYNET(nn.Module):

    def __init__(self, args, mode=None, trans=1):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder_q = resnet20(num_classes=self.args.moco_dim)
            self.encoder_k = resnet20(num_classes=self.args.moco_dim)
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder_q = resnet18(False, args, num_classes=self.args.moco_dim)  # pretrained=False
            self.encoder_k = resnet18(False, args, num_classes=self.args.moco_dim)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder_q = resnet18(True, args, num_classes=self.args.moco_dim)
            self.encoder_k = resnet18(True, args,
                                      num_classes=self.args.moco_dim)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes * trans, bias=False)

        self.K = self.args.moco_k
        self.m = self.args.moco_m
        self.T = self.args.moco_t

        if self.args.mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(),
                                              self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(),
                                              self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.args.moco_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

        # Add SelfAttention module
        self.attention = SelfAttention(self.num_features)

        # 添加因果SCM模块
        self.causal_module = CausalSCM(feature_dim=self.num_features,
                                       z_dim=args.causal_z_dim if hasattr(args, 'causal_z_dim') else 128,
                                       class_dim=args.causal_class_dim if hasattr(args, 'causal_class_dim') else 64,
                                       style_dim=args.causal_style_dim if hasattr(args, 'causal_style_dim') else 64)

        # 添加用于因果特征的分类器
        self.causal_classifier = nn.Linear(args.causal_class_dim if hasattr(args, 'causal_class_dim') else 64,
                                           self.args.num_classes * trans, bias=False)

        # 因果损失函数的权重
        self.causal_weight = args.causal_weight if hasattr(args, 'causal_weight') else 0.5

    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess):
        """
        Momentum update of the key encoder
        """
        if base_sess:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        else:
            for k, v in self.encoder_q.named_parameters():
                if k.startswith('fc') or k.startswith('layer4') or k.startswith('layer3'):
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[k].data * self.m + v.data * (
                            1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains:]
            self.label_queue[ptr:] = labels[:batch_size - remains]
            self.label_queue[:remains] = labels[batch_size - remains:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T  # this queue is feature queue
            self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward_metric(self, x):
        if self.training:
            x, _, causal_outputs = self.encode_q(x)
            class_features = causal_outputs['class_features']

            # 根据模式选择正确的分类方式
            if 'cos' in self.mode:
                # 对原始特征的分类
                orig_logits = F.linear(F.normalize(x, p=2, dim=-1),
                                       F.normalize(self.fc.weight, p=2, dim=-1))
                orig_logits = self.args.temperature * orig_logits

                # 对因果类别特征的分类
                causal_logits = F.linear(F.normalize(class_features, p=2, dim=-1),
                                         F.normalize(self.causal_classifier.weight, p=2, dim=-1))
                causal_logits = self.args.temperature * causal_logits

                # 组合两种logits
                # x = orig_logits + causal_logits
                x = orig_logits + self.causal_weight * causal_logits

            elif 'dot' in self.mode:
                orig_logits = self.fc(x)
                causal_logits = self.causal_classifier(class_features)
                x = orig_logits + self.causal_weight * causal_logits

            # 返回组合后的logits以及因果模块输出（用于计算损失）
            return x, causal_outputs

        else:
            x, _ = self.encode_q(x)
            if 'cos' in self.mode:
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
                x = self.args.temperature * x
            elif 'dot' in self.mode:
                x = self.fc(x)

            return x

    def encode_q(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        # 应用自注意力
        x = self.attention(x)

        # 应用因果模块进行干预
        causal_outputs = self.causal_module(x, training=self.training)

        # 在训练时返回因果模块的输出，测试时直接返回类别特征
        if self.training:
            return x, y, causal_outputs
        else:
            # 在测试时使用投影后的类别特征增强原始特征
            class_features_projected = causal_outputs['class_features_projected']
            # 通过残差连接方式融合原始特征和投影后的类别特征
            # x = x + class_features_projected
            x = x + self.causal_weight * class_features_projected
            return x, y

    def encode_k(self, x):
        x, y = self.encoder_k(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        x = self.attention(x)

        # 在测试时对key也应用因果处理以保持一致性
        if not self.training:
            causal_outputs = self.causal_module(x, training=False)
            class_features_projected = causal_outputs['class_features_projected']
            # x = x + class_features_projected
            x = x + self.causal_weight * class_features_projected

        return x, y

    def forward(self, im_cla, im_q=None, im_k=None, labels=None, im_q_small=None, base_sess=True,
                last_epochs_new=False):
        if self.mode != 'encoder':
            if im_q == None:
                if self.training:
                    x, causal_outputs = self.forward_metric(im_cla)
                    return x, causal_outputs
                else:
                    x = self.forward_metric(im_cla)
                    return x
            else:
                b = im_q.shape[0]

                # 获取分类logits和因果输出
                if self.training:
                    logits_classify, causal_outputs = self.forward_metric(im_cla)
                else:
                    logits_classify = self.forward_metric(im_cla)

                # 处理查询图像
                if self.training:
                    _, q, _ = self.encode_q(im_q)  # 忽略因果输出，仅使用MoCo特征
                else:
                    _, q = self.encode_q(im_q)

                q = nn.functional.normalize(q, dim=1)
                feat_dim = q.shape[-1]
                q = q.unsqueeze(1)  # bs x 1 x dim

                if im_q_small is not None:
                    if self.training:
                        _, q_small, _ = self.encode_q(im_q_small)  # 修改这里，正确解包3个返回值
                    else:
                        _, q_small = self.encode_q(im_q_small)

                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1, 1)
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim), self.queue.clone().detach()])
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim), self.queue.clone().detach()])

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda()
                # find same label images from label queue
                # for the query with -1, all
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets], dim=1)
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1], dim=0)
                labels_small = labels.repeat_interleave(repeats=self.args.num_crops[1], dim=0)

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)

                # 返回值根据训练/测试状态调整
                if self.training:
                    return logits_classify, logits_global, logits_small, targets_global, targets_small, causal_outputs
                else:
                    return logits_classify, logits_global, logits_small, targets_global, targets_small

        elif self.mode == 'encoder':
            # 编码器模式
            if self.training:
                x, _, _ = self.encode_q(im_cla)  # 忽略因果输出
            else:
                x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, transform, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            data, _ = self.encode_q(data)
            data.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list) * m, self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m)

    def update_fc_avg(self, data, labels, class_list, m):
        new_fc = []
        for class_index in class_list:
            for i in range(m):
                index = class_index * m + i
                data_index = (labels == index).nonzero().squeeze(-1)
                embedding = data[data_index]
                proto = embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
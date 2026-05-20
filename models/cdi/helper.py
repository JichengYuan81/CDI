# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from losses import SupContrastive, CausalInvarianceLoss


def base_train(model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
    tl = Averager()
    tl_joint = Averager()
    tl_moco = Averager()
    tl_moco_global = Averager()
    tl_moco_small = Averager()
    tl_causal = Averager()
    tl_recon = Averager()
    tl_inv = Averager()  # 新增：因果不变性损失
    ta = Averager()

    # 创建因果不变性损失函数
    inv_criterion = CausalInvarianceLoss(mode='kl', temperature=args.distill_temp)

    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, single_labels = [_ for _ in batch]
        b, c, h, w = data[1].shape
        original = data[0].cuda(non_blocking=True)
        data[1] = data[1].cuda(non_blocking=True)
        data[2] = data[2].cuda(non_blocking=True)
        single_labels = single_labels.cuda(non_blocking=True)
        if len(args.num_crops) > 1:
            data_small = data[args.num_crops[0] + 1].unsqueeze(1)
            for j in range(1, args.num_crops[1]):
                data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
            data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
        else:
            data_small = None

        data_classify = transform(original)
        data_query = transform(data[1])
        data_key = transform(data[2])
        data_small = transform(data_small)
        m = data_query.size()[0] // b
        joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

        joint_preds, logits_global, logits_small, target_global, target_small, causal_outputs = model(
            im_cla=data_classify, im_q=data_query, im_k=data_key,
            labels=joint_labels, im_q_small=data_small)

        loss_moco_global = criterion(logits_global, target_global)
        loss_moco_small = criterion(logits_small, target_small)
        loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

        joint_preds = joint_preds[:, :args.base_class * m]
        joint_loss = F.cross_entropy(joint_preds, joint_labels)

        recon_loss = F.mse_loss(causal_outputs['reconstructed_features'],
                                causal_outputs['original_features'])

        c_flat = causal_outputs['class_features'].view(b * m, -1)
        s_flat = causal_outputs['style_features'].view(b * m, -1)
        c_norm = F.normalize(c_flat, p=2, dim=1)
        s_norm = F.normalize(s_flat, p=2, dim=1)
        orthogonal_loss = torch.abs(torch.mm(c_norm, s_norm.t())).mean()

        # 生成反事实样本特征
        original_features = causal_outputs['original_features']
        cf_features, _, _ = model.module.causal_module.counterfactual(
            original_features, joint_labels, alpha=args.counterfactual_alpha)

        # 使用反事实特征进行前向传播
        model.module.mode = 'encoder'  # 临时切换到编码器模式
        features_cf = model.module.get_logits(cf_features, model.module.fc.weight)
        features_orig = model.module.get_logits(original_features, model.module.fc.weight)
        model.module.mode = args.base_mode  # 恢复模式

        # 计算因果不变性损失
        inv_loss = inv_criterion(features_orig, features_cf)

        # 总因果损失
        causal_loss = recon_loss + 0.1 * orthogonal_loss

        # 总损失
        # loss = joint_loss + loss_moco + args.causal_weight * causal_loss + args.counterfactual_weight * inv_loss
        loss = joint_loss + args.causal_weight * causal_loss + args.counterfactual_weight * inv_loss

        # 准确率计算
        acc = count_acc(joint_preds, joint_labels)

        # 记录损失值
        tl.add(loss.item())
        tl_joint.add(joint_loss.item())
        tl_moco_global.add(loss_moco_global.item())
        tl_moco_small.add(loss_moco_small.item())
        tl_moco.add(loss_moco.item())
        tl_causal.add(causal_loss.item())
        tl_recon.add(recon_loss.item())
        tl_inv.add(inv_loss.item())  # 新增：记录不变性损失
        ta.add(acc)

        # 显示进度
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total={:.4f},joint={:.4f},moco={:.4f},'
            'causal={:.4f},inv={:.4f},acc={:.4f}'.format(
                epoch, lrc, loss.item(), joint_loss.item(), loss_moco.item(),
                causal_loss.item(), inv_loss.item(), acc))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 返回值包括新增的不变性损失
    return (tl.item(), tl_joint.item(), tl_moco.item(), tl_moco_global.item(),
            tl_moco_small.item(), tl_causal.item(), tl_recon.item(), tl_inv.item(), ta.item())


def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class * m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class * m] = proto_list

    return model


def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class * m]

            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va
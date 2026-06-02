from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import os
import pickle

from losses import SupContrastive, CausalInvarianceLoss


def base_train(model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
    tl = Averager()
    tl_joint = Averager()
    tl_moco = Averager()
    tl_moco_global = Averager()
    tl_moco_small = Averager()
    tl_causal = Averager()
    tl_recon = Averager()
    tl_inv = Averager()  

    # Create causal invariance loss function
    inv_criterion = CausalInvarianceLoss(mode='kl', temperature=args.distill_temp)

    model = model.train()
    tqdm_gen = tqdm(trainloader)
    
    # Collect style features for storage during the last epoch
    style_features_list = []
    
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

        # Generate counterfactual sample features
        original_features = causal_outputs['original_features']
        cf_features, _, _ = model.module.causal_module.counterfactual(
            original_features, joint_labels, mode=args.counterfactual_mode, alpha=args.counterfactual_alpha)

        # Forward pass using counterfactual features
        model.module.mode = 'encoder'  # Temporarily switch to encoder mode
        features_cf = model.module.get_logits(cf_features, model.module.fc.weight)
        features_orig = model.module.get_logits(original_features, model.module.fc.weight)
        model.module.mode = args.base_mode  # Restore mode

        # Calculate causal invariance loss
        inv_loss = inv_criterion(features_orig, features_cf)

        # Total causal loss
        causal_loss = recon_loss + 0.1 * orthogonal_loss

        # Total loss
        loss = joint_loss + args.causal_weight * causal_loss + args.counterfactual_weight * inv_loss

        # Collect style features if it is the last epoch
        if epoch == args.epochs_base - 1:
            with torch.no_grad():
                style_features_list.append(causal_outputs['style_features'].cpu())

        # Calculate accuracy
        acc = count_acc(joint_preds, joint_labels)

        # Record loss values
        tl.add(loss.item())
        tl_joint.add(joint_loss.item())
        tl_moco_global.add(loss_moco_global.item())
        tl_moco_small.add(loss_moco_small.item())
        tl_moco.add(loss_moco.item())
        tl_causal.add(causal_loss.item())
        tl_recon.add(recon_loss.item())
        tl_inv.add(inv_loss.item()) 
        ta.add(acc)

        # Display progress
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {},total={:.4f},joint={:.4f},'
            'causal={:.4f},inv={:.4f},acc={:.4f}'.format(
                epoch, loss.item(), joint_loss.item(),
                causal_loss.item(), inv_loss.item(), acc))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save style features at the last epoch
    if epoch == args.epochs_base - 1 and len(style_features_list) > 0:
        # Concatenate all style features
        all_style_features = torch.cat(style_features_list, dim=0)
        
        # Create save directory
        style_features_dir = os.path.join(args.save_path, 'style_features')
        os.makedirs(style_features_dir, exist_ok=True)
        
        # Save style features
        style_features_path = os.path.join(style_features_dir, 'base_style_features.pkl')
        with open(style_features_path, 'wb') as f:
            pickle.dump(all_style_features, f)
        
        print(f"Base class style features saved to: {style_features_path}")
        print(f"Shape of saved style features: {all_style_features.shape}")

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


def update_fc_ft(trainloader, data_transform, model, m, session, args):
    # incremental finetuning with style feature sampling
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session
    new_fc = nn.Parameter(
        torch.rand(args.way * m, model.module.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.module.fc.weight[old_class * m: new_class * m, :].data)

    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.module.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                     {'params': model.module.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new}, ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                 {'params': model.module.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                 {'params': model.module.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new}],
                                momentum=0.9, dampening=0.9, weight_decay=0)

    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                      {'params': model.module.encoder_q.fc.parameters(), 'lr': 0.01 * args.lr_new},
                                      {'params': model.module.encoder_q.layer3.parameters(), 'lr': 0.02 * args.lr_new}],
                                     weight_decay=0)

    criterion = SupContrastive().cuda()
    
    # Load base class style features
    style_features_path = os.path.join(args.save_path, 'style_features', 'base_style_features.pkl')
    base_style_features = None
    
    if os.path.exists(style_features_path):
        try:
            with open(style_features_path, 'rb') as f:
                base_style_features = pickle.load(f).cuda()
            print(f"Successfully loaded base style features: {base_style_features.shape}")
        except Exception as e:
            print(f"Failed to load base style features: {e}")
            base_style_features = None
    else:
        print(f"Base style feature file does not exist: {style_features_path}")
    
    # Create causal invariance loss function
    inv_criterion = CausalInvarianceLoss(mode='kl', temperature=args.distill_temp)

    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, single_labels = [_ for _ in batch]
                b, c, h, w = data[1].shape
                origin = data[0].cuda(non_blocking=True)
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
                    
                data_classify = data_transform(origin)
                data_query = data_transform(data[1])
                data_key = data_transform(data[2])
                data_small = data_transform(data_small)
                joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

                old_fc = model.module.fc.weight[:old_class * m, :].clone().detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                
                # Extract features and obtain causal decoupling results
                model.module.mode = 'encoder'
                features, _ = model.module.encode_q(data_classify)
                features.detach()
                
                # Get causal decoupling outputs
                causal_outputs = model.module.causal_module(features, training=True)
                new_class_features = causal_outputs['class_features']  # Causal features of novel classes
                
                # Calculate base loss
                logits = model.module.get_logits(features, fc)
                joint_loss = F.cross_entropy(logits, joint_labels)
                
                # Perform style feature sampling and calculate causal invariance loss if base style features exist
                causal_inv_loss = torch.tensor(0.0).cuda()
                if base_style_features is not None and new_class_features.size(0) > 0:
                    # Multiple counterfactual generation
                    batch_size = new_class_features.size(0)
                    num_base_styles = base_style_features.size(0)
                    
                    multiplier = 8
                    sample_size = min(batch_size * multiplier, num_base_styles)
                    
                    # Sample batch_size * multiplier base style features
                    sampled_indices = torch.randperm(num_base_styles)[:sample_size]
                    sampled_style_features = base_style_features[sampled_indices]
                    
                    # Expand novel class features to match sampled style features
                    expanded_class_features = new_class_features.repeat_interleave(multiplier, dim=0)
                    
                    # Ensure style feature quantities match
                    if sample_size < batch_size * multiplier:
                        # Repeat style features to match
                        repeat_factor = (batch_size * multiplier + sample_size - 1) // sample_size
                        sampled_style_features = sampled_style_features.repeat(repeat_factor, 1)[:batch_size * multiplier]
                    
                    # Generate multiple counterfactual samples
                    reconstructed_features = model.module.causal_module.decode(
                        expanded_class_features,    
                        sampled_style_features     
                    )
                    
                    # Calculate multiple counterfactual predictions
                    reconstructed_logits = model.module.get_logits(reconstructed_features, fc)
                    
                    # Expand original predictions
                    expanded_logits = logits.repeat_interleave(multiplier, dim=0)
                    
                    # Calculate causal invariance loss
                    causal_inv_loss = inv_criterion(expanded_logits, reconstructed_logits)
                
                # Restore mode
                model.module.mode = args.base_mode
                
                # Calculate standard MoCo loss
                _, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query,
                                                                      im_k=data_key, labels=joint_labels,
                                                                      im_q_small=data_small, base_sess=False,
                                                                      last_epochs_new=(epoch == args.epochs_new - 1))
                
                loss_moco_global = criterion(output_global, target_global)
                loss_moco_small = criterion(output_small, target_small)
                loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small
                
                # Total loss: add causal invariance loss
                loss = joint_loss + 0.1 * causal_inv_loss  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    model.module.fc.weight.data[old_class * m: new_class * m, :].copy_(new_fc.data)


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
        
    print('Epoch {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

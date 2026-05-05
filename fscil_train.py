from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from losses import SupContrastive
from augmentations import fantasy


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        if args.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[args.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0

        # 根据数据集动态设置因果模块维度
        if args.dataset == 'cifar100':
            args.causal_z_dim = 32
            args.causal_class_dim = 16
            args.causal_style_dim = 16
        else:  # mini_imagenet或cub200
            args.causal_z_dim = 128
            args.causal_class_dim = 64
            args.causal_style_dim = 64

        self.model = MYNET(self.args, mode=self.args.base_mode, trans=self.num_trans)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # 在训练开始时就创建保存目录
        os.makedirs(self.args.save_path, exist_ok=True)

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)

            # 增量会话前创建预训练模型
            if session > 0:
                # 创建教师模型作为单独实例
                self.pre_model = MYNET(self.args, mode=self.args.base_mode, trans=self.num_trans)

                # 针对best_model_dict处理"module."前缀问题
                if list(self.best_model_dict.keys())[0].startswith('module.'):
                    # 创建新的状态字典，去除"module."前缀
                    new_state_dict = {}
                    for k, v in self.best_model_dict.items():
                        name = k[7:] if k.startswith('module.') else k  # 去除'module.'前缀
                        new_state_dict[name] = v
                    pre_model_state_dict = new_state_dict
                else:
                    pre_model_state_dict = self.best_model_dict

                # 加载处理后的状态字典
                self.pre_model.load_state_dict(pre_model_state_dict, strict=True)
                self.pre_model = self.pre_model.cuda()
                # 设置模式为评估模式
                self.pre_model.eval()

            if session == 0:  # load base class train img label

                train_set.multi_train = True
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                criterion = SupContrastive()
                criterion = criterion.cuda()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, tl_causal, tl_recon, tl_inv, ta = base_train(
                        self.model, trainloader,
                        criterion, optimizer,
                        scheduler, epoch,
                        self.transform, args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, self.transform, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        # 保存模型前确保目录存在
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        os.makedirs(os.path.dirname(save_model_dir), exist_ok=True)
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,train_loss:%.5f,joint_loss:%.5f,moco_loss:%.5f,moco_global:%.5f,'
                        'moco_small:%.5f,causal:%.5f,recon:%.5f,inv:%.5f,train_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small,
                            tl_causal, tl_recon, tl_inv, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    train_set.multi_train = False
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.transform, self.model,
                                                 args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, self.transform, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode

                self.model.eval()
                train_transform = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), self.transform, session)
                if args.incft:
                    trainloader.dataset.transform = train_transform
                    train_set.multi_train = True
                    update_fc_ft(trainloader, self.transform, self.model, self.num_trans, session, args)

                tsl, tsa = test(self.model, testloader, 0, self.transform, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                os.makedirs(os.path.dirname(save_model_dir), exist_ok=True)
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        # 移除可能存在的引号
        mode = mode.replace("'", "")
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)
        self.args.save_path = self.args.save_path + f'-fantasy_{self.args.fantasy}'
        self.args.save_path = self.args.save_path + '-alpha_%.2f-beta_%.2f-causal_%.2f' % (
            self.args.alpha, self.args.beta, self.args.causal_weight)
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        # 使用os.makedirs创建目录，exist_ok=True 确保目录存在时不会报错
        os.makedirs(self.args.save_path, exist_ok=True)

        # 添加反事实训练相关的路径信息，先检查属性是否存在
        if hasattr(self.args, 'use_counterfactual') and self.args.use_counterfactual:
            self.args.save_path = self.args.save_path + '-CF_w%.2f_a%.2f' % (
                self.args.counterfactual_weight, self.args.counterfactual_alpha)

        return None

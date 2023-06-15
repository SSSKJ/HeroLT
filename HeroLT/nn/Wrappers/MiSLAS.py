from . import BaseModel
from ...configs.MiLAS import _C as config
from ...configs.MiLAS import update_config
from ..Models import MiSLAS_resnet
from ..Models import MiSLAS_resnet_cifar
from ..Models import MiSLAS_resnet_places
from ...tools.loaders import *
from ...tools import LearnableWeightScaling, LabelAwareSmoothing
from ...utils import AverageMeter, ProgressMeter, calibration, mixup_data, mixup_criterion

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

import os
import random
import numpy as np
import math

class MiSLAS(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'MiSLAS',
            dataset_name = dataset,
            base_dir = base_dir)
        

        super().load_config()

        self.ngpus_per_node = torch.cuda.device_count()

    def __load_config(self):

        config_path = f'{self.base_dir}/{self.model_name}/{self.dataset_name}'
        self.s1_config = update_config(config, f'{config_path}/s1_config.yaml')
        self.s2_config = update_config(config, f'{config_path}/s2_config.yaml')

    def __init_model(self, config, phase):

        if self.dataset_name == 'cifar10_lt' or self.dataset_name == 'cifar100_lt':
            self.model = getattr(MiSLAS_resnet_cifar, config.backbone)()
            self.classifier = getattr(MiSLAS_resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)
            self.block = None

        elif self.dataset_name == 'imagenet_lt' or self.dataset_name == 'inatural2018_lt':
            self.model = getattr(MiSLAS_resnet, config.backbone)()
            self.classifier = getattr(MiSLAS_resnet, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
            self.block = None

        elif self.dataset_name == 'places_lt':
            self.model = getattr(MiSLAS_resnet_places, config.backbone)(pretrained=True)
            self.classifier = getattr(MiSLAS_resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
            self.block = getattr(MiSLAS_resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d)

        if phase == 's2':

            self.lws_model = LearnableWeightScaling(num_classes=config.num_classes)

    def __init_optimizer_and_scheduler(self, config, phase):

        if phase == 's1':

            if self.dataset_name == 'places':
                self.optimizer = torch.optim.SGD([{"params": self.block.parameters()},
                                            {"params": self.classifier.parameters()}], config.lr,
                                            momentum=config.momentum,
                                            weight_decay=config.weight_decay)
            else:
                self.optimizer = torch.optim.SGD([{"params": self.model.parameters()},
                                            {"params": self.classifier.parameters()}], config.lr,
                                            momentum=config.momentum,
                                            weight_decay=config.weight_decay)
                
        else:

            self.optimizer = torch.optim.SGD([{"params": self.classifier.parameters()},
                                {'params': self.lws_model.parameters()}], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

            
    def __init_criterion(self, config, phase):

        if phase == 's1':
            self.criterion = nn.CrossEntropyLoss().cuda(config.gpu)

        else:
            self.criterion = LabelAwareSmoothing(cls_num_list=self.dataset.cls_num_list, smooth_head=config.smooth_head,
                                    smooth_tail=config.smooth_tail).cuda(config.gpu)

    
    ## 直接在这里写两个stage
    def train(self, phase):

        ## stage 1 training
        if phase in ['s1', 'all']:

            if self.s1_config.deterministic:
                seed = 0
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                random.seed(seed)
                np.random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if self.s1_config.dist_url == "env://" and self.s1_config.world_size == -1:
                self.s1_config.world_size = int(os.environ["WORLD_SIZE"])

            self.s1_config.distributed = self.s1_config.world_size > 1 or self.s1_config.multiprocessing_distributed

            if self.s1_config.multiprocessing_distributed:
                # Since we have ngpus_per_node processes per node, the total world_size
                # needs to be adjusted accordingly
                self.s1_config.world_size = self.ngpus_per_node * self.s1_config.world_size
                # Use torch.multiprocessing.spawn to launch distributed processes: the
                # main_worker process function
                mp.spawn(self.__main_worker, nprocs=self.ngpus_per_node, args=(self.s1_config, 's1'))
            else:
                # Simply call main_worker function
                self.__main_worker(self.s1_config, 's1')
        
        if phase in ['s2', 'all']:

            ## stage 2 training
            if self.s2_config.deterministic:
                seed = 0
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                random.seed(seed)
                np.random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if self.s2_config.dist_url == "env://" and self.s2_config.world_size == -1:
                self.s2_config.world_size = int(os.environ["WORLD_SIZE"])

            self.s2_config.distributed = self.s2_config.world_size > 1 or self.s2_config.multiprocessing_distributed

            if self.s2_config.multiprocessing_distributed:
                # Since we have ngpus_per_node processes per node, the total world_size
                # needs to be adjusted accordingly
                self.s2_config.world_size = self.ngpus_per_node * self.s2_config.world_size
                # Use torch.multiprocessing.spawn to launch distributed processes: the
                # main_worker process function
                mp.spawn(self.__main_worker, nprocs=self.ngpus_per_node, args=(self.s2_config, 's2'))
            else:
                # Simply call main_worker function
                self.__main_worker(self.s2_config, 's2')

    def __main_worker(self, config, phase):

        model_dir = f'{self.base_dir}/outputs/{self.model_name}/{self.dataset_name}/'

        best_acc1 = 0
        its_ece = 100

        if config.gpu is not None:
            print("Use GPU: {} for training".format(config.gpu))

        if config.distributed:
            if config.dist_url == "env://" and config.rank == -1:
                config.rank = int(os.environ["RANK"])
            if config.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                config.rank = config.rank * self.ngpus_per_node + config.gpu
            dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                    world_size=config.world_size, rank=config.rank)
            
        self.__init_model(config, phase)

        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif config.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if config.gpu is not None:
                torch.cuda.set_device(config.gpu)
                self.model.cuda(config.gpu)
                self.classifier.cuda(config.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                config.batch_size = int(config.batch_size / self.ngpus_per_node)
                config.workers = int((config.workers + self.ngpus_per_node - 1) / self.ngpus_per_node)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config.gpu])
                self.classifier = torch.nn.parallel.DistributedDataParallel(self.classifier, device_ids=[config.gpu])
                if self.dataset_name == 'places_lt':
                    self.block.cuda(config.gpu)
                    self.block = torch.nn.parallel.DistributedDataParallel(self.block, device_ids=[config.gpu])
            else:
                self.model.cuda()
                self.classifier.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
                self.classifier = torch.nn.parallel.DistributedDataParallel(self.classifier)
                if self.dataset_name == 'places_lt':
                    self.block.cuda()
                    self.block = torch.nn.parallel.DistributedDataParallel(self.block)

        elif config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            self.model = self.model.cuda(config.gpu)
            self.classifier = self.classifier.cuda(config.gpu)
            if self.dataset_name == 'places_lt':
                self.block.cuda(config.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.classifier = torch.nn.DataParallel(self.classifier).cuda()
            if self.dataset_name == 'places_lt':
                self.block = torch.nn.DataParallel(self.block).cuda()

        # optionally resume from a checkpoint
        if config.resume:
            if os.path.isfile(config.resume):
                print("=> loading checkpoint '{}'".format(config.resume))
                if config.gpu is None:
                    checkpoint = torch.load(config.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(config.gpu)
                    checkpoint = torch.load(config.resume, map_location=loc)
                # config.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if config.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(config.gpu)
                self.model.load_state_dict(checkpoint['state_dict_model'])
                self.classifier.load_state_dict(checkpoint['state_dict_classifier'])
                print("=> loaded checkpoint '{}' (epoch {})"
                            .format(config.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(config.resume))

        self.load_data()

        train_loader = self.dataset.train_instance
        val_loader = self.dataset.eval
        if config.distributed:
            train_sampler = self.dataset.dist_sampler

        self.__init_optimizer_and_scheduler(config, phase)
        self.__init_criterion(config, phase)

        for epoch in range(config.num_epochs):
            if config.distributed:
                train_sampler.set_epoch(epoch)

            self.__adjust_learning_rate(epoch, config, phase)

            if self.dataset_name != 'places_lt':
                self.block = None
            # train for one epoch

            eval(f'self.__{phase}_train_model')(train_loader, epoch, config)

            # evaluate on validation set
            acc1, ece = self.__validate(val_loader, config)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                its_ece = ece
            print('Best Prec@1: %.3f%% ECE: %.3f%%\n' % (best_acc1, its_ece))

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                        and config.rank % self.ngpus_per_node == 0):
                if self.dataset_name == 'places_lt':
                    self.__save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': self.model.state_dict(),
                        'state_dict_classifier': self.classifier.state_dict(),
                        'state_dict_block': self.block.state_dict(),
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)

                else:
                    self.__save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': self.model.state_dict(),
                        'state_dict_classifier': self.classifier.state_dict(),
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)

    def load_data(self):
        # Data loading code
        if self.dataset_name == 'cifar10_lt':
            self.dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                                batch_size=config.batch_size, num_works=config.workers)

        elif self.dataset_name == 'cifar100_lt':
            self.dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                                batch_size=config.batch_size, num_works=config.workers)

        elif self.dataset_name == 'places_lt':
            self.dataset = Places_LT(config.distributed, root=config.data_path,
                                batch_size=config.batch_size, num_works=config.workers)

        elif self.dataset_name == 'imagenet_lt':
            self.dataset = ImageNet_LT(config.distributed, root=config.data_path,
                                batch_size=config.batch_size, num_works=config.workers)

        elif self.dataset_name == 'inatural2018_lt':
            self.dataset = iNatural2018(config.distributed, root=config.data_path,
                            batch_size=config.batch_size, num_works=config.workers)

    def __s1_train_model(self, train_loader, epoch, config):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        if self.dataset_name == 'places_lt':
            self.model.eval()
            self.block.train()
        else:
            self.model.train()
        self.classifier.train()

        training_data_num = len(train_loader.dataset)
        end_steps = int(training_data_num / train_loader.batch_size)

        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break

            if torch.cuda.is_available():
                images = images.cuda(config.gpu, non_blocking=True)
                target = target.cuda(config.gpu, non_blocking=True)

            if config.mixup is True:
                images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
                if self.dataset_name == 'places_lt':
                    with torch.no_grad():
                        feat_a = self.model(images)
                    feat = self.block(feat_a.detach())
                    output = self.classifier(feat)
                else:
                    feat = self.model(images)
                    output = self.classifier(feat)
                loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
            else:
                if self.dataset_name == 'places_lt':
                    with torch.no_grad():
                        feat_a = self.model(images)
                    feat = self.block(feat_a.detach())
                    output = self.classifier(feat)
                else:
                    feat = self.model(images)
                    output = self.classifier(feat)

                loss = self.criterion(output, target)

            acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if i % config.print_freq == 0:
            #     progress.display(i, logger)

    def __s2_train_model(self, train_loader, epoch, config):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        training_data_num = len(train_loader.dataset)
        end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))
        progress = ProgressMeter(
            end_steps,
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode

        if config.dataset == 'places':
            self.model.eval()
            if config.shift_bn:
                self.block.train()
            else:
                self.block.eval()
        else:
            if config.shift_bn:
                self.model.train()
            else:
                self.model.eval()
        self.classifier.train()

        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break

            if torch.cuda.is_available():
                images = images.cuda(config.gpu, non_blocking=True)
                target = target.cuda(config.gpu, non_blocking=True)

            if config.mixup is True:
                images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
                with torch.no_grad():
                    if config.dataset == 'places':
                        feat = self.block(self.model(images))
                    else:
                        feat = self.model(images)
                output = self.classifier(feat.detach())
                output = self.lws_model(output)
                loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
            else:
                # compute output
                with torch.no_grad():
                    if config.dataset == 'places':
                        feat = self.block(self.model(images))
                    else:
                        feat = self.model(images)
                output = self.classifier(feat.detach())
                output = self.lws_model(output)
                loss = self.criterion(output, target)

            acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if i % config.print_freq == 0:
            #     progress.display(i, logger)

    def __validate(self, val_loader, config):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Eval: ')

        # switch to evaluate mode
        self.model.eval()
        if self.dataset_name == 'places_lt':
            self.block.eval()
        self.classifier.eval()
        class_num = torch.zeros(config.num_classes).cuda()
        correct = torch.zeros(config.num_classes).cuda()

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                if config.gpu is not None:
                    images = images.cuda(config.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(config.gpu, non_blocking=True)

                # compute output
                feat = self.model(images)
                if self.dataset_name == 'places_lt':
                    feat = self.block(feat)
                output = self.classifier(feat)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                _, predicted = output.max(1)
                target_one_hot = F.one_hot(target, config.num_classes)
                predict_one_hot = F.one_hot(predicted, config.num_classes)
                class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
                correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

                prob = torch.softmax(output, dim=1)
                confidence_part, pred_class_part = torch.max(prob, dim=1)
                confidence = np.append(confidence, confidence_part.cpu().numpy())
                pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
                true_class = np.append(true_class, target.cpu().numpy())

                # if i % config.print_freq == 0:
                #     progress.display(i, logger)

            acc_classes = correct / class_num
            head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100

            med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
            tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
            print('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

            cal = calibration(true_class, pred_class, confidence, num_bins=15)
            print('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

        return top1.avg, cal['expected_calibration_error'] * 100


    def __save_checkpoint(state, is_best, model_dir):
        filename = model_dir + '/current.pth.tar'
        torch.save(state, filename)
        # if is_best:
        #     shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


    def __adjust_learning_rate(self, epoch, config, phase):
        
        if phase == 's1':
            """Sets the learning rate"""
            if config.cos:
                lr_min = 0
                lr_max = config.lr
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
            else:
                epoch = epoch + 1
                if epoch <= 5:
                    lr = config.lr * epoch / 5
                elif epoch > 180:
                    lr = config.lr * 0.01
                elif epoch > 160:
                    lr = config.lr * 0.1
                else:
                    lr = config.lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if phase == 's2':

            lr_min = 0
            lr_max = config.lr
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))

            for idx, param_group in enumerate(self.optimizer.param_groups):
                if idx == 0:
                    param_group['lr'] = config.lr_factor * lr
                else:
                    param_group['lr'] = 1.00 * lr

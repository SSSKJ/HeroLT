from HeroLT.utils import source_import
from HeroLT.nn.Wrappers import BaseModel
from utils import *
from Schedulers import CosineAnnealingLRWarmup
from utils.logger import Logger

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import pickle
import numpy as np
from tqdm import tqdm
import math
import higher

class BALMS(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            test_mode: bool = False,
            ) -> None:
        
        super().__init__(
            model_name = 'BALM',
            dataset = dataset,
            base_dir = base_dir)
        
        self.__load_config()

        self.test_mode = test_mode

        self.training_opt = self.config['training_opt']

        self.logger = Logger(self.base_dir, self.model_name, self.dataset_name)
        self.logger.log(f'Log will be saved to {self.base_dir}/logs/{self.model_name}_{self.dataset_name}.log')

        self.meta_sample = None
        self.learner = None
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = self.config['shuffle'] if 'shuffle' in self.config else False
        
        # Initialize model
        self.__init_model()


    def __init_optimizer_and_scheduler(self, optim_params):

        # Initialize model optimizer and scheduler
        self.logger.log('Initializing model optimizer.')
        self.scheduler_params = self.training_opt['scheduler_params']
        self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)

        optimizer = optim.SGD(optim_params)
        if self.config['coslr']:
            self.logger.log("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        elif self.config['coslrwarmup']:
            self.logger.log("===> Using coslrwarmup eta_min={}, warmup_epochs={}".format(
                self.config['endlr'],self.config['warmup_epochs']))
            scheduler = CosineAnnealingLRWarmup(
                optimizer=optimizer,
                T_max=self.training_opt['num_epochs'],
                eta_min=self.config['endlr'],
                warmup_epochs=self.config['warmup_epochs'],
                base_lr=self.config['base_lr'],
                warmup_lr=self.config['warmup_lr']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def __init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())

            self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                self.logger.log('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.__init_optimizer_and_scheduler(optim_params)
            else:
                self.criterion_optimizer = None


    def __init_model(self):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        if self.meta_sample:
            # init meta optimizer
            self.optimizer_meta = torch.optim.Adam(self.learner.parameters(),
                                                   lr=self.training_opt['sampler'].get('lr', 0.01))

        self.logger.log("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            # model_args = list(val['params'].values())
            # model_args.append(self.test_mode)
            model_args = val['params']
            model_args.update({'test': self.test_mode})

            self.networks[key] = source_import(def_file).create_model(**model_args)
            if 'KNNClassifier' in type(self.networks[key]).__name__:
                # Put the KNN classifier on one single GPU
                self.networks[key] = self.networks[key].cuda()
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                self.logger.log('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False
                    # self.logger.log('  | ', param_name, param.requires_grad)

            if self.meta_sample and key!='classifier':
                # avoid adding classifier parameters to the optimizer,
                # otherwise error will be raised when computing higher gradients
                continue

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']})
    

    def train(self):

        if self.__training_data is None:

            self.load_data(train = True)

        self.data = self.__training_data

        # Initialize optimizer and scheduler
        self.__init_optimizer_and_scheduler(self.model_optim_params_list)
        self.__init_criterions()
        if self.memory['init_centroids']:
            self.criterions['FeatureLoss'].centroids.data = self.__centroids_cal(self.data['train_plain'])

        # Compute epochs from iterations
        if self.training_opt.get('num_iterations', False):
            self.training_opt['num_epochs'] = math.ceil(self.training_opt['num_iterations'] / len(self.data['train']))
        if self.config.get('warmup_iterations', False):
            self.config['warmup_epochs'] = math.ceil(self.config['warmup_iterations'] / len(self.data['train']))

        # If using steps for training, we need to calculate training steps 
        # for each epoch based on actual number of training data instead of 
        # oversampled data number 
        self.logger.log('Using steps for training.')
        self.training_data_num = len(self.data['train'].dataset)
        self.epoch_steps = int(self.training_data_num  \
                                / self.training_opt['batch_size'])
        

        if self.memory['init_centroids']:
            self.criterions['FeatureLoss'].centroids.data = \
                self.__centroids_cal(self.data['train_plain'])
            
        # When training the network
        self.logger.log('Phase: train')

        self.logger.log('Do shuffle??? --- ', self.do_shuffle)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            for model in self.networks.values():
                model.train()

            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # Iterate over dataset
            total_preds = []
            total_labels = []

            for step, (inputs, labels, indexes) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.__shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                    if self.meta_sample:
                        # do inner loop
                        self.__meta_forward(inputs, labels, verbose=step % self.training_opt['display_step'] == 0)
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.__batch_forward(inputs, labels, 
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.__batch_loss(labels)
                    self.__batch_backward()

                    # Tracking predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)

                        self.logger.log('Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_feature: %.3f' 
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf) if minibatch_loss_perf else '',
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (minibatch_acc))

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf,
                            'feat': minibatch_loss_feat
                        }


                # Update priority weights if using PrioritizedSampler
                # if self.training_opt['sampler'] and \
                #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
                if hasattr(self.data['train'].sampler, 'update_weights'):
                    if hasattr(self.data['train'].sampler, 'ptype'):
                        ptype = self.data['train'].sampler.ptype 
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)
                    # ws = logits2score(self.logits.detach(), labels)
                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)
                    # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)

            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.__eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data['train'].sampler, 'reset_priority'):
                ws = get_priority(self.data['train'].sampler.ptype,
                                  self.total_logits.detach(),
                                  self.total_labels)
                self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())


            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            
            self.logger.log('===> Saving checkpoint')
            self.__save_latest(epoch)

        self.logger.log('Training Complete.')

        self.logger.log('est validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
        # Save the best model and best centroids if calculated
        self.__save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        # Test on the test set
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        self.logger.log('Done')

    

    def load_data(
            self,
            force: bool = False):

        super().load_data()

        self.logger.log('Loading data for Training')
        self.logger.log(self.config)

        dataset = self.dataset_name.lower()

        if not self.test_mode:

            if self.__training_data is not None and not force:

                return self.__training_data

            sampler_defs = self.training_opt['sampler']
            if sampler_defs:
                def_file = sampler_defs['def_file']
                if sampler_defs['type'] == 'ClassAwareSampler':
                    sampler_dic = {
                        'sampler': source_import(f'{self.base_dir}/tools/{def_file}.py').get_sampler(),
                        'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
                    }
                elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                            'ClassPrioritySampler']:
                    sampler_dic = {
                        'sampler': source_import(f'{self.base_dir}/tools/{def_file}.py').get_sampler(),
                        'params': {k: v for k, v in sampler_defs.items() \
                                if k not in ['type', 'def_file']}
                    }
                elif sampler_defs['type'] == 'MetaSampler':  # Add option for Meta Sampler
                    self.learner = source_import(f'{self.base_dir}/tools/{def_file}.py').get_learner()(
                        num_classes=self.training_opt['num_classes'],
                        init_pow=sampler_defs.get('init_pow', 0.0),
                        freq_path=sampler_defs.get('freq_path', None)
                    ).cuda()
                    sampler_dic = {
                        'batch_sampler': True,
                        'sampler': source_import(f'{self.base_dir}/tools/{def_file}.py').get_sampler(),
                        'params': {'meta_learner': self.learner, 'batch_size': self.training_opt['batch_size']}
                    }
            else:
                sampler_dic = None

            splits = ['train', 'train_plain', 'val']
            if dataset not in ['inatural2018', 'imagenet_lt']:
                splits.append('test')
            self.__training_data = {x: TDE_loader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}',
                                            dataset=dataset, phase=x, 
                                            batch_size=self.training_opt['batch_size'],
                                            sampler_dic=sampler_dic,
                                            num_workers=self.training_opt['num_workers'],
                                            cifar_imb_ratio=self.training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in self.training_opt else None)
                    for x in splits}

            if sampler_defs and sampler_defs['type'] == 'MetaSampler':   # todo: use meta-sampler
                cbs_sampler_dic = {
                        'sampler': source_import(f'{self.base_dir}/tools/ClassAwareSampler.py').get_sampler(),
                        'params': {'is_infinite': True}
                    }
                # use Class Balanced Sampler to create meta set
                self.__training_data['meta'] = TDE_loader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}',
                                            dataset=dataset, phase='train' if 'CIFAR' in dataset else 'val',
                                            batch_size=sampler_defs.get('meta_batch_size', self.training_opt['batch_size'], ),
                                            sampler_dic=cbs_sampler_dic,
                                            num_workers=self.training_opt['num_workers'],
                                            cifar_imb_ratio=self.training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in self.training_opt else None,
                                            meta=True)
                self.meta_sample = True
                # init meta learner and meta set
                assert self.learner is not None
                self.meta_data = iter(self.__training_data['meta'])

        else:

            self.logger.log('Under testing phase, we load training data simply to calculate \
                training data number for each class.')

            if dataset == 'inatural2018':
                splits = ['train', 'val']
                self.test_split = 'val'
            else:
                splits = ['train', 'val', 'test']
                self.test_split = 'test'
            if dataset == 'imageNet-lt':
                splits = ['train', 'val']
                self.test_split = 'val'
            splits.append('train_plain')

            data = {x: TDE_loader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}',
                                            dataset=dataset, phase=x,
                                            batch_size=self.training_opt['batch_size'],
                                            sampler_dic=None, 
                                            test_open=False,
                                            num_workers=self.training_opt['num_workers'],
                                            shuffle=False,
                                            cifar_imb_ratio=self.training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in self.training_opt else None)
                    for x in splits}
            
            self.__testing_data = data
            self.data = data
        

    def __batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                    torch.cat([self.centroids] * self.num_gpus)
                else:
                    self.centroids = None

            if self.centroids is not None:
                centroids_ = torch.cat([self.centroids] * self.num_gpus)
            else:
                centroids_ = self.centroids

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, centroids_)

    def __batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def __batch_loss(self, labels):
        self.loss = 0

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
            self.loss_perf *=  self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat
    
    def __shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def __meta_forward(self, inputs, labels, verbose=False):
        # take a meta step in the inner loop
        self.learner.train()
        self.model_optimizer.zero_grad()
        self.optimizer_meta.zero_grad()
        with higher.innerloop_ctx(self.networks['classifier'], self.model_optimizer) as (fmodel, diffopt):
            # obtain the surrogate model
            features, _ = self.networks['feat_model'](inputs)
            train_outputs, _ = fmodel(features.detach())
            loss = self.criterions['PerformanceLoss'](train_outputs, labels, reduction='none')
            loss = self.learner.forward_loss(loss)
            diffopt.step(loss)

            # use the surrogate model to update sample rate
            val_inputs, val_targets, _ = next(self.meta_data)
            val_inputs = val_inputs.cuda()
            val_targets = val_targets.cuda()
            features, _ = self.networks['feat_model'](val_inputs)
            val_outputs, _ = fmodel(features.detach())
            val_loss = F.cross_entropy(val_outputs, val_targets, reduction='mean')
            val_loss.backward()
            self.optimizer_meta.step()

        self.learner.eval()

        if verbose:
            # log the sample rates
            num_classes = self.learner.num_classes
            prob = self.learner.fc[0].weight.sigmoid().squeeze(0)
            s = 'Unnormalized Sample Prob: '
            interval = 1 if num_classes < 10 else num_classes // 10
            for i in range(0, num_classes, interval):
                s += 'class{}={:.3f}, '.format(i, prob[i].item())
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            s += ', Max Mem: {:.0f}M'.format(max_mem_mb)
            self.logger.log(s)

    def __eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)
        
        # Calculate normal prediction accuracy
        rsl = {'train_all':0., 'train_many':0., 'train_median':0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds*2, mixup_labels1+mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1-mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        self.logger.log('\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n')

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False):

        self.logger.log('Phase: %s' % (phase))
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.__batch_forward(inputs, labels, 
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())

        if get_feat_only:
            typ = 'feat'
            if phase == 'train_plain':
                name = 'train{}_all.pkl'.format(typ)
            elif phase == 'test':
                name = 'test{}_all.pkl'.format(typ)
            elif phase == 'val':
                name = 'val{}_all.pkl'.format(typ)

            fname = os.path.join(self.training_opt['log_dir'], name)
            self.logger.log('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                             'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all),
                            },
                            f, protocol=4) 
            return 
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            self.logger.log('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1], 
                                 self.data['train'],
                                 acc_per_cls=True)
        # Top-1 accuracy and additional string
        s = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']
        
        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}

        if phase == 'val':
            self.logger.log(s)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            self.logger.log(*s)
            self.logger.log(*acc_str)
        
        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl
            
    def __centroids_cal(self, data, save_all=False):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        self.logger.log('Calculating centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all, idxs_all = [], [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                # Save features if requried
                if save_all:
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(idxs.numpy())
        
        if save_all:
            fname = os.path.join(self.training_opt['log_dir'], 'feats_all.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids
    
    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_pretrained_model(self):

        model_dir = f'{self.output_path}/'

        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        
        self.logger.log('Validation on the best model.')
        self.logger.log('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():
            # if not self.test_mode and key == 'classifier':
            if not self.test_mode and \
                'DotProductClassifier' in self.config['networks'][key]['def_file']:
                # Skip classifier initialization 
                self.logger.log('Skiping classifier initialization')
                continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    
    def __save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)
        
    def __save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)
        

    


        

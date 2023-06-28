from .BaseModel import BaseModel
from ..Dataloaders.TDEDataLoader import TDEDataLoader
from ..Samplers.ClassAwareSampler import ClassAwareSampler
from ..Schedulers.WarmupMultiStepLR import WarmupMultiStepLR
from ...utils.logger import Logger
from ...utils import source_import, torch2numpy, mic_acc_cal, get_priority, shot_acc, weighted_mic_acc_cal, weighted_shot_acc, F_measure, print_grad_norm

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import IncrementalPCA

import os
import copy
import pickle
from tqdm import tqdm

class TDE(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            test_mode: bool = False,
            ) -> None:
        
        super().__init__(
            model_name = 'TDE',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()
        self.test_mode = test_mode

        self.networks = None
        self.__training_data = None
        self.__testing_data = None

        self.logger = Logger(self.base_dir, self.model_name, self.dataset_name)
        self.logger.log.info(f'Log will be saved to {self.base_dir}/logs/{self.model_name}_{self.dataset_name}.log')

        self.training_opt = self.config['training_opt']

        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = self.config['shuffle'] if 'shuffle' in self.config else False

        # init moving average
        self.embed_mean = torch.zeros(int(self.training_opt['feature_dim'])).numpy()
        self.mu = 0.9
        
        # Initialize model
        self.__init_models()

        # apply incremental pca
        self.apply_pca = ('apply_ipca' in self.config) and self.config['apply_ipca']
        if self.apply_pca:
            self.logger.log.info('==========> Apply Incremental PCA <=======')
            self.pca = IncrementalPCA(n_components=self.config['num_components'], batch_size=self.training_opt['batch_size'])


    def __init_models(self, optimizer=True):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_dict = {}
        self.model_optim_named_params = {}

        self.logger.log.info("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():
            # Networks
            def_file = val['def_file']
            model_args = val['params']
            model_args.update({'test': self.test_mode})
            model_args.update({'dataset': self.dataset_name, 'log_dir': self.output_path, 'logger': self.logger.log})

            self.networks[key] = source_import(f'{self.base_dir}/nn/Models/{def_file}.py').create_model(**model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                self.logger.log.info('Freezing weights of module {}'.format(key))
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except final fc layer
                    if 'fc' not in param_name:
                        param.requires_grad = False
                self.logger.log.info('=====> Freezing: {} | False'.format(key))
            
            if 'fix_set' in val:
                for fix_layer in val['fix_set']:
                    for param_name, param in self.networks[key].named_parameters():
                        if fix_layer == param_name:
                            param.requires_grad = False
                            self.logger.log.info('=====> Freezing: {} | {}'.format(param_name, param.requires_grad))
                            continue


            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_named_params.update(dict(self.networks[key].named_parameters()))
            self.model_optim_params_dict[key] = {'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']}
        
        
    def load_data(
        self,
        force: bool = False,
        test_mode: bool = None
        ):
    
        super().load_data()

        ## todo: need to check if the files have been placed in the right place

        training_opt = self.config['training_opt']
        dataset = self.dataset_name.lower()

        if test_mode is None:

            test_mode = self.test_mode

        if not test_mode:

            if self.__training_data is not None and not force:

                return self.__training_data

            self.logger.log.info('Loading data for Training')
            sampler_defs = training_opt['sampler']
            if sampler_defs and sampler_defs['type'] == 'ClassAwareSampler':
                    sampler_dic = {
                        'sampler': ClassAwareSampler,
                        'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
                    }
            else:
                sampler_dic = None

            splits = ['train', 'val']
            if dataset not in ['inatural2018', 'imagenet_lt']:
                splits.append('test')
            data = {x: TDEDataLoader.load_data(data_root=f'{self.base_dir}/data/CVData/{dataset}',
                                            dataset=dataset, 
                                            phase=x, 
                                            batch_size=training_opt['batch_size'],
                                            logger = self.logger,
                                            sampler_dic=sampler_dic,
                                            num_workers=training_opt['num_workers'],
                                            top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                            cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,)
                    for x in splits}

        else:

            
            if (self.__testing_data is not None) and (not force):

                return self.__testing_data

            self.logger.log.info('Loading data for Testing')
            self.logger.log.info('Under testing phase, we load training data simply to calculate \
                training data number for each class.')

            if dataset == 'inatural2018':
                splits = ['train', 'val']
                self.test_split = 'val'
            else:
                splits = ['train', 'val', 'test']
                self.test_split = 'test'
            if dataset == 'imagenet_lt':
                splits = ['train', 'val']
                self.test_split = 'val'

            data = {x: TDEDataLoader.load_data(data_root=f'{self.base_dir}/data/CVData/{dataset}',
                                    dataset=dataset, 
                                    phase=x,
                                    batch_size=training_opt['batch_size'],
                                    logger = self.logger,
                                    sampler_dic=None, 
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    shuffle=False,
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,)
            for x in splits}
            
            self.__testing_data = data
            self.data = data

    def __init_optimizer_and_scheduler(self, optim_params_dict):
        '''
        seperate backbone optimizer and classifier optimizer
        by Kaihua
        '''
        networks_defs = self.config['networks']
        self.model_optimizer_dict = {}
        self.model_scheduler_dict = {}

        for key, val in networks_defs.items():
            # optimizer
            if 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adam':
                self.logger.log.info('=====> Using Adam optimizer')
                optimizer = optim.Adam([optim_params_dict[key],])
            else:
                self.logger.log.info('=====> Using SGD optimizer')
                optimizer = optim.SGD([optim_params_dict[key],])
            self.model_optimizer_dict[key] = optimizer
            # scheduler
            scheduler_params = val['scheduler_params']
            if scheduler_params['coslr']:
                self.logger.log.info("===> Module {} : Using coslr eta_min={}".format(key, scheduler_params['endlr']))
                self.model_scheduler_dict[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                                    optimizer, self.training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
            elif scheduler_params['warmup']:
                self.logger.log.info("===> Module {} : Using warmup".format(key))
                self.model_scheduler_dict[key] = WarmupMultiStepLR(optimizer, scheduler_params['lr_step'], 
                                                    gamma=scheduler_params['lr_factor'], warmup_epochs=scheduler_params['warm_epoch'])
            else:
                self.model_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer,
                                                                                    step_size=scheduler_params['step_size'],
                                                                                    gamma=scheduler_params['gamma'])

        return
    
    def __init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params']
            loss_args.update({'logger': self.logger.log})

            self.criterions[key] = source_import(f'{self.base_dir}/nn/Loss/{def_file}.py').create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                self.logger.log.info('Initializing criterion optimizer.')
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
    

    def train(self):

        if self.__training_data is None:

            self.load_data(test_mode = False)

        self.data = self.__training_data

        # Initialize model optimizer and scheduler
        self.logger.log.info('Initializing model optimizer.')
        self.__init_optimizer_and_scheduler(self.model_optim_params_dict)
        self.__init_criterions()

        # If using steps for training, we need to calculate training steps 
        # for each epoch based on actual number of training data instead of 
        # oversampled data number 
        self.logger.log.info('Using steps for training.')
        self.training_data_num = len(self.data['train'].dataset)
        self.epoch_steps = int(self.training_data_num  / self.training_opt['batch_size'])

        # When training the network
        self.logger.log.info('-----------------------------------Phase: train-----------------------------------')

        self.logger.log.info(['Force shuffle in training??? --- ', self.do_shuffle])

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            for key, model in self.networks.items():
                # only train the module with lr > 0
                if self.config['networks'][key]['optim_params']['lr'] == 0.0:
                    self.logger.log.info(['=====> module {} is set to eval due to 0.0 learning rate.'.format(key)])
                    model.eval()
                else:
                    model.train()

            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            for key, scheduler in self.model_scheduler_dict.items():
                scheduler.step() 
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # Iterate over dataset
            total_preds = []
            total_labels = []

            # indicate current path
            self.logger.log.info(self.training_opt['log_dir'])
            # print learning rate
            current_lr = self.__show_current_lr()
            current_lr = min(current_lr * 50, 1.0)
            # scale the original mu according to the lr
            if 'CIFAR' not in self.training_opt['dataset']:
                self.mu = 1.0 - (1 - 0.9) * current_lr

            for step, (inputs, labels, indexes) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.__shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.__batch_forward(inputs, labels, phase='train')
                    self.__batch_loss(labels)
                    self.__batch_backward(print_grad=(step % self.training_opt['display_grad_step'] == 0))

                    # Tracking predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_route = self.loss_route.item() \
                            if 'RouteWeightLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)

                        self.logger.log.info('Epoch: [%d/%d]' % (epoch, self.training_opt['num_epochs']))
                        self.logger.log.info('Step: %5d' % (step))
                        self.logger.log.info('Minibatch_loss_route: %.3f' % (minibatch_loss_route) if minibatch_loss_route else '')
                        self.logger.log.info('Minibatch_loss_performance: %.3f' % (minibatch_loss_perf) if minibatch_loss_perf else '',)
                        self.logger.log.info('Minibatch_accuracy_micro: %.3f' % (minibatch_acc))
                        
                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf,
                            'route': minibatch_loss_route,
                        }

                        self.logger.log_loss(loss_info)

                # batch-level: sampler update
                if hasattr(self.data['train'].sampler, 'update_weights'):
                    if hasattr(self.data['train'].sampler, 'ptype'):
                        ptype = self.data['train'].sampler.ptype 
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)

                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)

            # epoch-level: reset sampler weight
            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data['train'].sampler, 'reset_priority'):
                ws = get_priority(self.data['train'].sampler.ptype,
                                  self.total_logits.detach(),
                                  self.total_labels)
                self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())

            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            
            self.logger.log.info('===> Saving checkpoint')
            self.__save_latest(epoch)

        self.logger.log.info('Training Complete.')

        self.logger.log.info('Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
        # Save the best model
        self.__save_model(epoch, best_epoch, best_model_weights, best_acc)

        # Test on the test set
        self._reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        self.logger.log.info('Done')

    def __show_current_lr(self):
        max_lr = 0.0
        for key, val in self.model_optimizer_dict.items():
            lr_set = list(set([para['lr'] for para in val.param_groups]))
            if max(lr_set) > max_lr:
                max_lr = max(lr_set)
            lr_set = ','.join([str(i) for i in lr_set])
            self.logger.log.info('=====> Current Learning Rate of model {} : {}'.format(key, str(lr_set)))
        return max_lr
    
    def __batch_forward(self, inputs, labels=None, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        self.features = self.networks['feat_model'](inputs)

        if self.apply_pca:
            if phase=='train' and self.features.shape[0] > 0:
                self.pca.partial_fit(self.features.cpu().numpy())
            else:
                pca_feat = self.pca.transform(self.features.cpu().numpy())
                pca_feat[:, 0] = 0.0
                new_feat = self.pca.inverse_transform(pca_feat)
                self.features = torch.from_numpy(new_feat).float().to(self.features.device)

        # update moving average
        if phase == 'train':
            self.embed_mean = self.mu * self.embed_mean + self.features.detach().mean(0).view(-1).cpu().numpy()

        # If not just extracting features, calculate logits
        if not feature_ext:
            # cont_eval = 'continue_eval' in self.training_opt and self.training_opt['continue_eval'] and phase != 'train'
            self.logits, self.route_logits = self.networks['classifier'](self.features, labels, self.embed_mean)

    def __batch_backward(self, print_grad=False):
        # Zero out optimizer gradients
        for key, optimizer in self.model_optimizer_dict.items():
            optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # display gradient
        if self.training_opt['display_grad']:
            print_grad_norm(self.model_optim_named_params, self.logger.log, verbose=print_grad)
        # Step optimizers
        for key, optimizer in self.model_optimizer_dict.items():
            optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def __batch_loss(self, labels):
        self.loss = 0

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
            self.loss_perf *=  self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf

        # Apply loss on Route Weights if set up
        if 'RouteWeightLoss' in self.criterions.keys():
            self.loss_route = self.criterions['RouteWeightLoss'](self.route_logits, labels)
            self.loss_route = self.loss_route * self.criterion_weights['RouteWeightLoss']
            # Add Route Weights loss to total loss
            self.loss += self.loss_route

    def __shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y
    
    def eval_with_preds(self, preds, labels):
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
        self.logger.log.info('Training acc Top1: %.3f ' % (rsl['train_all']))
        self.logger.log.info('Many_top1: %.3f' % (rsl['train_many']))
        self.logger.log.info('Median_top1: %.3f' % (rsl['train_median']))
        self.logger.log.info('Low_top1: %.3f' % (rsl['train_low']))

        return rsl

    def eval(self, phase='val', save_feat=False):

        self.logger.log.info('Phase: %s' % (phase))
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []

        # feature saving initialization
        if save_feat:
            self.__saving_feature_with_label_init()

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.__batch_forward(inputs, labels, phase=phase)
                # feature saving update
                if save_feat:
                    self.__saving_feature_with_label_update(self.features, self.logits, labels)

                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_paths = np.concatenate((self.total_paths, paths))

        # feature saving export
        if save_feat:
            self.__saving_feature_with_label_export()

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, theta=self.training_opt['open_threshold'])

        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1], 
                                 self.data['train'],
                                 acc_per_cls=True)
        
        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}
                                 
        # Top-1 accuracy and additional string
        self.logger.log.info('Phase: %s, Evaluation_accuracy_micro_top1: %.3f, Averaged F-measure: %.3f, Many_shot_accuracy_top1: %.3f, Median_shot_accuracy_top1: %.3f, Low_shot_accuracy_top1: %.3f' 
            % (phase, self.eval_acc_mic_top1, self.eval_f_measure, self.many_acc_top1, self.median_acc_top1, self.low_acc_top1),)

        if phase != 'val':
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            self.logger.log.info(*acc_str)
        
        if phase == 'test':
            with open(f'{self.output_path}/cls_accs.pkl', 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl
    
    def __reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_pretrained_model(self):

        model_dir = f'{self.output_path}/'

        if 'CIFAR' in self.training_opt['dataset']:
            # CIFARs don't have val set, so use the latest model
            self.logger.log.info('Validation on the latest model.')
            if not model_dir.endswith('.pth'):
                model_dir = os.path.join(model_dir, 'latest_model_checkpoint.pth')
            self.logger.log.info('Loading model from %s' % (model_dir))
            checkpoint = torch.load(model_dir)          
            model_state = checkpoint['state_dict']
        else:
            self.logger.log.info('Validation on the best model.')
            if not model_dir.endswith('.pth'):
                model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
            self.logger.log.info('Loading model from %s' % (model_dir))
            checkpoint = torch.load(model_dir)          
            model_state = checkpoint['state_dict_best']
        
        for key, model in self.networks.items():
            ##########################################
            # if loading classifier in training:
            #     1. only tuning memory embedding
            #     2. retrain the entire classifier
            ##########################################
            if 'embed' in checkpoint:
                self.logger.log.info('============> Load Moving Average <===========')
                self.embed_mean = checkpoint['embed']
            if not self.test_mode and 'Classifier' in self.config['networks'][key]['def_file']:
                if 'tuning_memory' in self.config and self.config['tuning_memory']:
                    self.logger.log.info('=============== WARNING! WARNING! ===============')
                    self.logger.log.info('========> Only Tuning Memory Embedding  <========')
                    for param_name, param in self.networks[key].named_parameters():
                        # frezing all params only tuning memory_embeding
                        if 'embed' in param_name:
                            param.requires_grad = True
                            self.logger.log.info('=====> Abandon Weight {} in {} from the checkpoints.'.format(param_name, key))
                            if param_name in model_state[key]:
                                del model_state[key][param_name]
                        else:
                            param.requires_grad = False
                        self.logger.log.info('=====> Tuning: {} | {}'.format(str(param.requires_grad).ljust(5, ' '), param_name))
                    self.logger.log.info('=================================================')
                else:
                    # Skip classifier initialization 
                    #self.logger.log.info('================ WARNING! WARNING! ================')
                    self.logger.log.info('=======> Load classifier from checkpoint <=======')
                    #self.logger.log.info('===================================================')
                    #continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            if all([weights[k].sum().item() == x[k].sum().item() for k in weights if k in x]):
                self.logger.log.info('=====> All keys in weights have been loaded to the module {}'.format(key))
            else:
                self.logger.log.info('=====> Error! Error! Error! Error! Loading failure in module {}'.format(key))
            model.load_state_dict(x)
    
    def __save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights,
            'embed': self.embed_mean,
        }

        model_dir = f'{self.output_path}/latest_model_checkpoint.pth'

        torch.save(model_states, model_dir)
        
    def __save_model(self, epoch, best_epoch, best_model_weights, best_acc):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'embed': self.embed_mean,}

        model_dir = f'{self.output_path}/final_model_checkpoint.pth'

        torch.save(model_states, model_dir)

    def __saving_feature_with_label_init(self):
        self.saving_feature_container = []
        self.saving_logit_container = []
        self.saving_label_container = []


    def __saving_feature_with_label_update(self, features, logits, labels):
        self.saving_feature_container.append(features.detach().cpu())
        self.saving_logit_container.append(logits.detach().cpu())
        self.saving_label_container.append(labels.detach().cpu())

    
    def __saving_feature_with_label_export(self):
        eval_features = {'features': torch.cat(self.saving_feature_container, dim=0).numpy(),
                    'labels': torch.cat(self.saving_label_container, dim=0).numpy(),
                    'logits': torch.cat(self.saving_logit_container, dim=0).numpy(),
                    }

        eval_features_dir = f'{self.output_path}/eval_features_with_labels.pth'
        
        torch.save(eval_features, eval_features_dir)
        self.logger.log.info('=====> Features with labels are saved as {}'.format(eval_features_dir))
    

        
        
        
        

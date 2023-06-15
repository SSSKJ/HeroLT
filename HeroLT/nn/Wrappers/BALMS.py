from ...utils import source_import
from . import CVModel
from ..Schedulers import CosineAnnealingLRWarmup
from ..Dataloaders import BALMSDataLoader
from ...utils import torch2numpy, mic_acc_cal, get_priority

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import math
import higher

class BALMS(CVModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            test_mode: bool = False,
            ) -> None:
        
        super().__init__(
            model_name = 'BALM',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()

        self.meta_sample = False
        self.learner = None
        
        # Initialize model
        self.__init_model()


    def __init_model(self):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        if self.meta_sample:
            # init meta optimizer
            self.optimizer_meta = torch.optim.Adam(self.learner.parameters(),
                                                   lr=self.training_opt['sampler'].get('lr', 0.01))

        self.logger.log.info("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            # model_args = list(val['params'].values())
            # model_args.append(self.test_mode)
            model_args = val['params']
            model_args.update({'test': self.test_mode})
            model_args.update({'dataset': self.dataset_name, 'log_dir': self.output_path, 'logger': self.logger.log})

            self.networks[key] = source_import(f'{self.base_dir}/nn/Models/{def_file}.py').create_model(**model_args)
            if 'KNNClassifier' in type(self.networks[key]).__name__:
                # Put the KNN classifier on one single GPU
                self.networks[key] = self.networks[key].cuda()
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                self.logger.log.info('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False
                    # self.logger.log.info('  | ', param_name, param.requires_grad)

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
            
    def __init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}
        
        ## todo: check freq path
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
                self.criterion_optimizer_scheduler = self.__init_optimizer(optim_params)
            else:
                self.criterion_optimizer = None
    
    def __init_optimizer_and_scheduler(self, optim_params):

        # Initialize model optimizer and scheduler
        self.logger.log.info('Initializing model optimizer.')
        self.model_optimizer, self.model_optimizer_scheduler = self.__init_optimizer(self.model_optim_params_list)

    def __init_optimizer(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.config['coslr']:
            self.logger.log.info("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        elif self.config['coslrwarmup']:
            self.logger.log.info("===> Using coslrwarmup eta_min={}, warmup_epochs={}".format(
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
    

    def train(self):

        if self.__training_data is None:

            self.load_data(test_mode = False)

        self.data = self.__training_data

        # Compute epochs from iterations
        if self.training_opt.get('num_iterations', False):
            self.training_opt['num_epochs'] = math.ceil(self.training_opt['num_iterations'] / len(self.data['train']))
        if self.config.get('warmup_iterations', False):
            self.config['warmup_epochs'] = math.ceil(self.config['warmup_iterations'] / len(self.data['train']))

        # If using steps for training, we need to calculate training steps 
        # for each epoch based on actual number of training data instead of 
        # oversampled data number 
        self.logger.log.info('Using steps for training.')
        self.training_data_num = len(self.data['train'].dataset)
        self.epoch_steps = int(self.training_data_num  \
                                / self.training_opt['batch_size'])

        # Initialize optimizer and scheduler
        self.__init_optimizer_and_scheduler(self.model_optim_params_list)
        self.__init_criterions()
        if self.memory['init_centroids']:
            self.criterions['FeatureLoss'].centroids.data = self.__centroids_cal(self.data['train_plain'])

        # When training the network
        self.logger.log.info('-----------------------------------Phase: train-----------------------------------')
        self.logger.log.info(f'Do shuffle??? --- {self.do_shuffle}')

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
                    inputs, labels = super().shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                    if self.meta_sample:
                        # do inner loop
                        self.__meta_forward(inputs, labels, verbose=step % self.training_opt['display_step'] == 0)
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    super().batch_forward(inputs, labels, 
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    super().batch_loss(labels)
                    super().batch_backward()

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

                        self.logger.log.info('Epoch: [%d/%d]' % (epoch, self.training_opt['num_epochs']))
                        self.logger.log.info('Step: %5d' % (step))
                        self.logger.log.info('Minibatch_loss_feature: %.3f' % (minibatch_loss_feat) if minibatch_loss_feat else '')
                        self.logger.log.info('Minibatch_loss_performance: %.3f' % (minibatch_loss_perf) if minibatch_loss_perf else '',)
                        self.logger.log.info('Minibatch_accuracy_micro: %.3f' % (minibatch_acc))

                        loss_info = {
                                'Epoch': epoch,
                                'Step': step,
                                'Total': minibatch_loss_total,
                                'CE': minibatch_loss_perf,
                                'feat': minibatch_loss_feat
                            }

                        self.logger.log_loss(loss_info)

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

            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = super().eval_with_preds(total_preds, total_labels)
            rsls_eval = super().eval(phase='val')
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
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            
            self.logger.log.info('===> Saving checkpoint')
            super().save_latest(epoch)

        self.logger.log.info('Training Complete.')

        self.logger.log.info('est validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
        # Save the best model and best centroids if calculated
        super().save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        # Test on the test set
        super().reset_model(best_model_weights)
        super().eval('test' if 'test' in self.data else 'val')
        self.logger.log.info('Done')

    

    def load_data(
            self,
            force: bool = False,
            test_mode: bool = None
            ):

        super().load_data()

        self.logger.log.info('Loading data for Training')
        self.logger.log.info(self.config)

        dataset = self.dataset_name.lower()

        if test_mode is None:

            test_mode = self.test_mode

        if not test_mode:

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
            self.__training_data = {x: BALMSDataLoader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}',
                                            dataset=dataset, 
                                            phase=x, 
                                            batch_size=self.training_opt['batch_size'],
                                            logger=self.logger,
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
                self.__training_data['meta'] = BALMSDataLoader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}',
                                            dataset=dataset, 
                                            phase='train' if 'CIFAR' in dataset else 'val',
                                            batch_size=sampler_defs.get('meta_batch_size', self.training_opt['batch_size'], ),
                                            logger=self.logger,
                                            sampler_dic=cbs_sampler_dic,
                                            num_workers=self.training_opt['num_workers'],
                                            cifar_imb_ratio=self.training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in self.training_opt else None,
                                            meta=True)
                self.meta_sample = True
                # init meta learner and meta set
                assert self.learner is not None
                self.meta_data = iter(self.__training_data['meta'])

        else:

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
            splits.append('train_plain')

            data = {x: BALMSDataLoader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}',
                                            dataset=dataset, 
                                            phase=x,
                                            batch_size=self.training_opt['batch_size'],
                                            logger=self.logger,
                                            sampler_dic=None, 
                                            test_open=False,
                                            num_workers=self.training_opt['num_workers'],
                                            shuffle=False,
                                            cifar_imb_ratio=self.training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in self.training_opt else None)
                    for x in splits}
            
            self.__testing_data = data
            self.data = data

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
            self.logger.log.info(s)
    


    def load_pretrained_model(self):

        if self.networks is None:
            self.__init_model()

        model_dir = f'{self.output_path}/final_model_checkpoint.pth'
        
        self.logger.log.info('Validation on the best model.')
        self.logger.log.info('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():
            # if not self.test_mode and key == 'classifier':
            if not self.test_mode and \
                'DotProductClassifier' in self.config['networks'][key]['def_file']:
                # Skip classifier initialization 
                self.logger.log.info('Skiping classifier initialization')
                continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    


    


        

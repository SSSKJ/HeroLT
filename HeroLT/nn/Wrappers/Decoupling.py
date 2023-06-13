from HeroLT.utils import source_import
from HeroLT.nn.Wrappers import CVModel
from HeroLT.nn.Dataloaders import DecouplingLoader
from HeroLT.nn.Samplers import ClassAwareSampler
from HeroLT.utils import torch2numpy, mic_acc_cal, get_priority

import torch
import torch.optim as optim

import copy

## todo: check source_import function
## todo: feature uniform

class Decoupling(CVModel):

    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            test_mode: bool = False,
            ) -> None:
        
        super().__init__(
            model_name = 'Decoupling',
            dataset = dataset,
            base_dir = base_dir)
        
        self.__load_config()
        # Initialize model
        self.__init_model()

    def __init_optimizer_and_scheduler(self):
        # Initialize model optimizer and scheduler
        self.logger.log('Initializing model optimizer.')
        self.scheduler_params = self.training_opt['scheduler_params']
        self.model_optimizer, self.model_optimizer_scheduler = self.__init_optimizers(self.model_optim_params_list)

    def load_data(
            self,
            force: bool = False):
        
        super().load_data()

        ## todo: need to check if the files have been placed in the right place

        training_opt = self.config['training_opt']
        dataset = self.dataset_name.lower()

        if not self.test_mode:

            if self.__training_data is not None and not force:

                return self.__training_data

            self.logger.log('Loading data for Training')
            sampler_defs = training_opt['sampler']
            if sampler_defs and sampler_defs['type'] == 'ClassAwareSampler':
                    sampler_dic = {
                        'sampler': ClassAwareSampler.get_sampler(),
                        'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
                    }
            else:
                sampler_dic = None

            splits = ['train', 'train_plain', 'val']
            if dataset not in ['inatural2018', 'imagenet_lt']:
                splits.append('test')
            self.__training_data = {x: DecouplingLoader.load_data(data_root = f'{self.base_dir}/datasets/{dataset}',
                                            dataset = dataset, 
                                            phase = x, 
                                            batch_size = training_opt['batch_size'],
                                            logger = self.logger,
                                            sampler_dic = sampler_dic,
                                            num_workers = training_opt['num_workers'])
                    for x in splits}

        else:

            
            if (self.__testing_data is not None) and (not force):

                return self.__testing_data

            self.logger.log('Loading data for Testing')
            self.logger.log('Under testing phase, we load training data simply to calculate \
                training data number for each class.')

            if dataset == 'inatural2018':
                splits = ['train', 'val']
                self.test_phase = 'val'
            else:
                splits = ['train', 'val', 'test']
                self.test_phase = 'test'
            if dataset == 'imagenet_lt':
                splits = ['train', 'val']
                self.test_phase = 'val'

            splits.append('train_plain')

            data = {x: DecouplingLoader.load_data(data_root = f'{self.base_dir}/datasets/{dataset}',
                                            dataset = dataset, 
                                            phase = x,
                                            batch_size = training_opt['batch_size'],
                                            logger = self.logger,
                                            sampler_dic = None, 
                                            num_workers = training_opt['num_workers'],
                                            shuffle = False)
                    for x in splits}
            
            self.__testing_data = data
            self.data = data

    
    def __init_model(self):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        self.logger.log(f"Using {torch.cuda.device_count()} GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = val['params']
            model_args.update({'test': self.test_mode})
            model_args.update({'dataset': self.dataset_name, 'log_dir': self.output_path})

            self.networks[key] = source_import(f'{self.base_dir}/nn/Models/{def_file}.py').create_model(**model_args)
            self.networks[key] = torch.nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                self.logger.log('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

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

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())

            self.criterions[key] = source_import(f'{self.base_dir}/nn/Loss/{def_file}.py').create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                self.logger.log('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, self.criterion_optimizer_scheduler = self.__init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def __init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.config['coslr']:
            self.logger.log("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler
    

    ## todo: Parallel Training
    def train(self, device = -1):
        
        if self.__training_data is None:

            self.load_data(train = True)

        self.data = self.__training_data

        self.logger.log('Using steps for training.')
        self.training_data_num = len(self.data['train'].dataset)
        self.epoch_steps = int(self.training_data_num / self.training_opt['batch_size'])

        # Initialize optimizer and scheduler
        self.__init_optimizer_and_scheduler()
        self.__init_criterions()
        if self.memory['init_centroids']:
            self.criterions['FeatureLoss'].centroids.data = self.__centroids_cal(self.data['train_plain'])

        # When training the network
        self.logger.log('-----------------------------------Phase: train-----------------------------------')
        self.logger.log(f'Do shuffle??? --- {self.do_shuffle}')
        
        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

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

                        self.logger.log('Epoch: [%d/%d]' % (epoch, self.training_opt['num_epochs']))
                        self.logger.log('Step: %5d' % (step))
                        self.logger.log('Minibatch_loss_feature: %.3f' % (minibatch_loss_feat) if minibatch_loss_feat else '')
                        self.logger.log('Minibatch_loss_performance: %.3f' % (minibatch_loss_perf) if minibatch_loss_perf else '',)
                        self.logger.log('Minibatch_accuracy_micro: %.3f' % (minibatch_acc))

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
                    self.data['train'].sampler.update_weights(*inlist)

            ## todo: print
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

            
            # todo: Log results
            self.logger.log_acc(rsls)

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

        self.logger.log('Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
        # Save the best model and best centroids if calculated
        self.__save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        # Test on the test set
        self.__reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        self.logger.log('Done')
    
    def load_pretrained_model(self):

        if self.networks is None:
            self.__init_model()

        # Load pre-trained model parameters
        model_dir = f'{self.output_path}/final_model_checkpoint.pth'
        
        self.logger.log('Validation on the best model.')
        self.logger.log('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)










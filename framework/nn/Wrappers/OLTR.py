from BaseModel import BaseModel
from utils import source_import
from tools.loaders import OLTR_loader
from utils import *

import torch
from torch import nn
import torch.optim as optim

import copy
from tqdm import tqdm

class OLTR(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            test_mode: bool = False,
            ) -> None:
        
        super().__init__(
            model_name = 'OLTR',
            dataset = dataset,
            base_dir = base_dir)
        
        self.__load_config()
        self.test_mode = test_mode

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            

    def __init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def __init_optimizer_and_scheduler(self):
        # Initialize model optimizer and scheduler
        print('Initializing model optimizer.')
        self.scheduler_params = self.training_opt['scheduler_params']
        self.model_optimizer, \
        self.model_optimizer_scheduler = self.__init_optimizers(self.model_optim_params_list)

    def __init_optimizers(self, optim_params):

        optimizer = optim.SGD(optim_params)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.scheduler_params['step_size'],
                                              gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler
    
    def __load_config(self):

        config_path = f'{self.base_dir}/configs/{self.model_name}/'

        self.s1_config = source_import(f'{config_path}/s1_config.py').config
        self.s2_config = source_import(f'{config_path}/s2_config.py').config

    def train(self, phase):

        if self.__training_data is None:

            self.load_data(train = True)

        self.data = self.__training_data

        self.training_opt = eval(f'self.{phase}_config')['training_opt']
        self.memory = eval(f'self.{phase}_config')['memory']

        # Initialize model
        self.__init_model()
        print('Using steps for training.')
        self.training_data_num = len(self.data['train'].dataset)
        self.epoch_steps = int(self.training_data_num  \
                                / self.training_opt['batch_size'])
        
        self.__init_optimizer_and_scheduler()
        self.__init_criterions()
        
        if self.memory['init_centroids']:
            self.criterions['FeatureLoss'].centroids.data = \
                self.__centroids_cal(self.data['train_plain'])
        
        # When training the network
        print('Phase: train')

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
            
            # Iterate over dataset
            for step, (inputs, labels, _) in enumerate(self.data['train']):

                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.__batch_forward(inputs, labels, 
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.__batch_loss(labels)
                    self.__batch_backward()

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item()
                        _, preds = torch.max(self.logits, 1)
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print('Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_feature: %.3f' 
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f' 
                                     % (minibatch_loss_perf),
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (minibatch_acc))

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # After every epoch, validation
            self.eval(phase='val')

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = copy.deepcopy(epoch)
                best_acc = copy.deepcopy(self.eval_acc_mic_top1)
                best_centroids = copy.deepcopy(self.centroids)
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        print()
        print('Training Complete.')

        print('Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
        # Save the best model and best centroids if calculated
        self.__save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)
                
        print('Done')

    def load_data(
            self,
            phase: str,
            force: bool = False,
            ):
        
        super().load_data()

        ## todo: need to check if the files have been placed in the right place

        training_opt = eval(f'self.{phase}_config')['training_opt']
        dataset = self.dataset_name.lower()
        relatin_opt = eval(f'self.{phase}_config')['memory']

        if not self.test_mode:

            if self.__training_data is not None and not force:

                return self.__training_data

            print('Loading data for Training')
            sampler_defs = training_opt['sampler']
            if sampler_defs:
                sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(), 
                            'num_samples_cls': sampler_defs['num_samples_cls']}
            else:
                sampler_dic = None

            self.__training_data = {x: OLTR_loader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}', 
                                        dataset=self.dataset_name, 
                                        phase=x, 
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=sampler_dic,
                                        num_workers=training_opt['num_workers'])
                    for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centroids'] else ['train', 'val'])}

        else:

            
            if (self.__testing_data is not None) and (not force):

                return self.__testing_data

            print('Loading data for Testing')
            print('Under testing phase, we load training data simply to calculate \
                training data number for each class.')

            data = {x: OLTR_loader.load_data(data_root=f'{self.base_dir}/datasets/{dataset}', 
                                        dataset=dataset, 
                                        phase=x,
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=None, 
                                        num_workers=training_opt['num_workers'],
                                        shuffle=False)
                    for x in ['train', 'test']}
            
            self.__testing_data = data
            self.data = data


    def __init_model(self, optimizer=True):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)

            self.networks[key] = source_import(def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)
            
            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for modulated attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'modulatedatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})
            
    def __batch_forward(self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
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
                else:
                    self.centroids = None

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)

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

        # First, apply performance loss
        self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                    * self.criterion_weights['PerformanceLoss']

        # Add performance loss to total loss
        self.loss = self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def __centroids_cal(self, data):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        for model in self.networks.values():
            model.eval()

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            
            for inputs, labels, _ in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def load_pretrained_model(self):
            
        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')
        
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)
        
    def __save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

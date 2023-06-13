from HeroLT.nn.Wrappers import BaseModel
from HeroLT.utils import mic_acc_cal, shot_acc, weighted_mic_acc_cal, weighted_shot_acc, class_count, F_measure
from HeroLT.utils.logger import Logger

import torch
import torch.nn.functional as F

import os
import copy
import pickle
import numpy as np
from tqdm import tqdm



class CVModel(BaseModel):

    def __init__(
            self, 
            model_name: str, 
            dataset_name: str, 
            base_dir: str = '../../',
            test_mode: bool = False,
            ) -> None:
        
        super().__init__(model_name, dataset_name, base_dir)

        self.test_mode = test_mode
        self.networks = None

        self.logger = Logger(self.base_dir, self.model_name, self.dataset_name)
        self.logger.log(f'Log will be saved to {self.base_dir}/logs/{self.model_name}_{self.dataset_name}.log')

        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = self.config['shuffle'] if 'shuffle' in self.config else False

    
    def __load_config(self):
        super().__load_config()

    def load_data(self):
        super().load_data()

    def load_pretrained_model(self):
        super().load_pretrained_model()

    def __save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = f'{self.output_path}/latest_model_checkpoint.pth'
        torch.save(model_states, model_dir)
        
    def __save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = f'{self.output_path}/final_model_checkpoint.pth'

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(self.output_path, 
                                'logits_%s'%('open' if openset else 'close'))
        self.logger.log("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)

    def __reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

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
        self.logger.log('Training acc Top1: %.3f ' % (rsl['train_all']))
        self.logger.log('Many_top1: %.3f' % (rsl['train_many']))
        self.logger.log('Median_top1: %.3f' % (rsl['train_median']))
        self.logger.log('Low_top1: %.3f' % (rsl['train_low']))

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False):

        self.logger.log('-----------------------------------Phase: %s-----------------------------------' % (phase))
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
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

            fname = f'{self.output_path}/{name}'
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
        
        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}
        
        self.logger.log('Phase: %s, Evaluation_accuracy_micro_top1: %.3f, Averaged F-measure: %.3f, Many_shot_accuracy_top1: %.3f, Median_shot_accuracy_top1: %.3f, Low_shot_accuracy_top1: %.3f' 
            % (phase, self.eval_acc_mic_top1, self.eval_f_measure, self.many_acc_top1, self.median_acc_top1, self.low_acc_top1),)
        if phase != 'val':
            acc_str = "{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)
            self.logger.log(acc_str)
        
        if phase == 'test':
            with open(f'{self.output_path}/cls_accs.pkl', 'wb') as f:
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
            fname = f'{self.output_path}/feats_all.pkl'
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids
    
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
        
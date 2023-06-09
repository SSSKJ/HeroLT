from .BaseModel import BaseModel
from ..Models.tailGNN import tailGNN
from ..Layers.Discriminator import Discriminator
from ...utils import link_dropout, normalize_output, performance_measure
from ...utils.logger import get_logger
from ..Dataloaders.GraphDataLoader import GraphDataLoader

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
from copy import deepcopy

class TailGNN(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'TailGNN',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()
        self.model = None
        self.disc = None
        self.best_model = None
        self.best_disc = None
        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')


    def __init_model(self):

        self.model = tailGNN(params = self.config, ver = 1).to(self.config['device'])
        self.disc = Discriminator(self.config['nclass']).to(self.config['device'])

    def __init_optimizer_and_scheduler(self):

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config['lr'], weight_decay = self.config['wd'])
        self.optimizer_D = optim.Adam(self.disc.parameters(), lr = self.config['lr'], weight_decay = self.config['wd'])

    def load_pretrained_model(self):

        if self.model is None or self.disc is None:

            self.__init_model()
        
        ###### Load Pre-trained Model #######
        self.logger.info('Load Pre-trained Model') 

        model_path = f'{self.output_path}/{self.model_name}_best_model_on_{self.dataset_name}.model'
        if os.path.exists(model_path):
            pretrained_model = torch.load(model_path)
            self.model.load_state_dict(pretrained_model.state_dict())
            self.best_model = deepcopy(self.model)
        else:
            self.logger.info(f'Can\'t find pretrain model file under {model_path} for {self.model_name} on {self.dataset_name}, fail to load model')

        disc_path = f'{self.output_path}/{self.model_name}_best_disc_on_{self.dataset_name}.model'
        if os.path.exists(disc_path):
            pretrained_disc = torch.load(disc_path)
            self.disc.load_state_dict(pretrained_disc.state_dict())
            self.best_disc = deepcopy(self.disc)
        else:
            self.logger.info(f'Can\'t find pretrain discriminator file under {model_path} for {self.model_name} on {self.dataset_name}, fail to load discriminator')


    def save_model(self, model, name):

        ###### Save Model #######
        self.logger.info('Save Model')
        model_path = f'{self.output_path}/{name}.model'
        os.makedirs(f'{self.output_path}/', exist_ok=True)
        torch.save(model, model_path)
    
    def load_data(self):

        super().load_data()
    
        self.config, (self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test) = GraphDataLoader.load_data(self.config, self.dataset_name, self.model_name, f'{self.base_dir}/data/GraphData/', self.logger)
        
    def train(self):

        self.criterion = nn.BCELoss()
        self.load_data()

        seed = self.config['rnd'] - 1
        seed_result = {}
        seed_result['acc'] = []
        seed_result['bacc'] = []
        seed_result['precision'] = []
        seed_result['recall'] = []
        seed_result['mAP'] = []

        for r in range(self.config['num_seed']):

            seed += 1
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            self.logger.info(f'seed: {seed}')

            self.tail_adj = link_dropout(self.adj.cpu().numpy(), self.idx_train, k = self.config['k'])
            self.tail_adj = torch.FloatTensor(self.tail_adj)

            # Model and optimizer
            self.__init_model()
            self.__init_optimizer_and_scheduler()
            
            self.model.to(self.config['device'])
            self.disc.to(self.config['device'])
            self.features = self.features.to(self.config['device'])
            self.labels = self.labels.to(self.config['device'])
            self.adj = self.adj.to(self.config['device'])
            self.tail_adj = self.tail_adj.to(self.config['device'])

            self.h_labels = torch.full((len(self.idx_train), 1), 1.0, device = self.config['device'])
            self.t_labels = torch.full((len(self.idx_train), 1), 0.0, device = self.config['device'])

            es = 0
            val_bacc = []
            best_val_bacc = 0
            # Train model
            for epoch in range(self.config['ep']):
                L_d = self.__train_disc(epoch, self.idx_train)
                L_d = self.__train_disc(epoch, self.idx_train)

                Loss, acc_train, loss_val, acc_val, bacc_val, precision_val, recall_val, map_val = self.__train_embed(epoch, self.idx_train)
                acc_test, bacc_test, precision_test, recall_test, map_test = self.__test()
                val_bacc.append(bacc_val)
                max_idx = val_bacc.index(max(val_bacc))

                if bacc_val > best_val_bacc:
                    best_val_bacc = bacc_val
                    best_test_result = [acc_test, bacc_test, precision_test, recall_test, map_test]

                    self.best_model = deepcopy(self.model)
                    self.save_model(self.best_model, f'{self.model_name}_best_model_on_{self.dataset_name}')

                    self.best_disc = deepcopy(self.disc)
                    self.save_model(self.best_disc, f'{self.model_name}_best_disc_on_{self.dataset_name}')

                    es = 0
                else:
                    es += 1
                    if es >= self.config['ep_early']:
                        self.logger.info("Early stopping!")
                        break

                st = "[seed {}][{}][Epoch {}]".format(seed, 'TailGNN', epoch)
                # st += "[Train] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}|| ".format(acc_train, macro_F_train,
                #                                                                                      gmeans_train, bacc_train)
                st += "[Val] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}|| ".format(acc_val, bacc_val, precision_val, recall_val, map_val)
                st += "[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(acc_test, bacc_test, precision_test, recall_test, map_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f},  bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}".format(
                    max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4])

                if epoch % 100 == 0:
                    self.logger.info(st)

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['bacc'].append(float(best_test_result[1]))
            seed_result['precision'].append(float(best_test_result[2]))
            seed_result['recall'].append(float(best_test_result[3]))
            seed_result['mAP'].append(float(best_test_result[4]))

        acc = seed_result['acc']
        bacc = seed_result['bacc']
        precision = seed_result['precision']
        recall = seed_result['recall']
        mAP = seed_result['mAP']

        self.logger.info(
            '[Averaged result] ACC: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}, Precision: {:.1f}+{:.1f}, Recall: {:.1f}+{:.1f}, mAP: {:.1f}+{:.1f}'.format(
                np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc), np.mean(precision), np.std(precision), np.mean(recall), np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger.info('ACC bACC Precision Recall mAP')
        self.logger.info('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc), np.mean(precision),
                                                                                            np.std(precision), np.mean(recall), np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger.info(self.config)

    def __train_disc(self, epoch, batch):
        self.disc.train()
        self.optimizer_D.zero_grad()

        embed_h, _, _ = self.model(self.features, self.adj, True)
        embed_t, _, _ = self.model(self.features, self.tail_adj, False)

        prob_h = self.disc(embed_h)
        prob_t = self.disc(embed_t)

        # loss
        errorD = self.criterion(prob_h[batch], self.h_labels)
        errorG = self.criterion(prob_t[batch], self.t_labels)
        L_d = (errorD + errorG) / 2

        L_d.backward()
        self.optimizer_D.step()
        return L_d


    def __train_embed(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()

        embed_h, output_h, support_h = self.model(self.features, self.adj, True)
        embed_t, output_t, support_t = self.model(self.features, self.tail_adj, False)

        # loss
        L_cls_h = F.nll_loss(output_h[batch], self.labels[batch])
        L_cls_t = F.nll_loss(output_t[batch], self.labels[batch])
        L_cls = (L_cls_h + L_cls_t) / 2

        # weight regularizer
        m_h = normalize_output(support_h, batch)
        m_t = normalize_output(support_t, batch)

        prob_h = self.disc(embed_h)
        prob_t = self.disc(embed_t)

        errorG = self.criterion(prob_t[batch], self.t_labels)
        L_d = errorG
        L_all = L_cls - (self.config['eta'] * L_d) + self.config['mu'] * m_h

        L_all.backward()
        self.optimizer.step()
        acc_train, _, _, _, _ = performance_measure(embed_h[batch], self.labels[batch], pre='valid')

        # validate:
        self.model.eval()
        _, embed_val, _ = self.model(self.features, self.adj, False)
        loss_val = F.nll_loss(embed_val[self.idx_val], self.labels[self.idx_val])
        acc_val, bacc_val, precision_val, recall_val, map_val = performance_measure(embed_val[self.idx_val], self.labels[self.idx_val], pre='valid')

        return (L_all, L_cls, L_d), acc_train, loss_val, acc_val, bacc_val, precision_val, recall_val, map_val


    def __test(self):
        self.model.eval()
        _, embed_test, _ = self.model(self.features, self.adj, False)
        # loss_test = F.nll_loss(embed_test[idx_test], labels[idx_test])

        acc_test, bacc_test, precision_test, recall_test, map_test = performance_measure(embed_test[self.idx_test], self.labels[self.idx_test], pre='test')
        return acc_test, bacc_test, precision_test, recall_test, map_test
    
    def eval(self):

        self.best_model.eval()
        self.features.to(self.config['device'])
        self.adj.to(self.config['device'])

        _, embed_test, _ = self.best_model(self.features, self.adj, False)
        
        acc_test, bacc_test, precision_test, recall_test, map_test = performance_measure(embed_test[self.idx_test], self.labels[self.idx_test], pre='test')
        self.logger.log("[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(acc_test, bacc_test, precision_test, recall_test, map_test))
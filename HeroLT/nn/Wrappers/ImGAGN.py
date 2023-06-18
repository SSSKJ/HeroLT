from . import BaseModel
from ..Models import GCN
from ..Models.ImGAGN_Generator import Generator
from ..Dataloaders import GraphDataLoader
from ...utils.logger import get_logger
from ...utils import performance_measure, normalize, sparse_mx_to_torch_sparse_tensor

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import scipy.sparse as sp

import os
import time
from copy import deepcopy

class ImGAGN(BaseModel):

    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'ImGAGN',
            dataset_name = dataset,
            base_dir = base_dir)
        
        super().load_config()
        self.model = None
        self.generator = None
        self.best_model = None
        self.best_generator = None
        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')
    
    def load_data(self):

        super().load_data()

        self.config, (self.adj, self.adj_real, self.features, self.labels, self.idx_temp, self.idx_test, self.generate_node, self.minority, self.majority, self.minority_all) = GraphDataLoader.load_data(self.config, self.dataset_name, self.model_name, f'{self.base_dir}/data/GraphData/', self.logger)

    def __init_model(self):

        self.model = GCN(nfeat = self.features.shape[1],
            nhid = self.config['nhid'],
            nclass = self.labels.max().item() + 1,
            dropout = self.config['dropout'],
            generate_node = self.generate_node,
            min_node = self.minority).to(self.config['device'])
        
        self.generator = Generator(self.minority_all.shape[0]).to(self.config['device'])

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

        generator_path = f'{self.output_path}/{self.model_name}_best_generator_on_{self.dataset_name}.model'
        if os.path.exists(generator_path):
            pretrained_generator = torch.load(generator_path)
            self.generator.load_state_dict(pretrained_generator.state_dict())
            self.best_generator = deepcopy(self.generator)
        else:
            self.logger.info(f'Can\'t find pretrain generator file under {model_path} for {self.model_name} on {self.dataset_name}, fail to load generator')


    def save_model(self, model, name):

        ###### Save Model #######
        self.logger.info('Save Model')
        model_path = f'{self.output_path}/{name}.model'
        os.makedirs(f'{self.output_path}/', exist_ok=True)
        torch.save(model, model_path)
        
    def __init_optimizer_and_scheduler(self):

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config['lr'], weight_decay = self.config['wd'])
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = self.config['lr'], weight_decay = self.config['wd'])


    ## todo: Parallel Training
    def train(self, device = -1):
        
        num = 10
        self.load_data()
        device = self.config['device']

        t_total = time.time()
        repeatition = self.config['num_seed']
        seed = self.config['rnd'] - 1
        seed_result = {}
        seed_result['acc'] = []
        seed_result['bacc'] = []
        seed_result['precision'] = []
        seed_result['recall'] = []
        seed_result['mAP'] = []

        for r in range(repeatition):
            ## Fix seed ##
            # torch.cuda.empty_cache()
            seed += 1
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            self.logger.info(f'seed: {seed}')

            # Model and optimizer
            self.__init_model()
            self.__init_optimizer_and_scheduler()

            num_false = self.labels.shape[0] - self.features.shape[0]

            es = 0
            val_bacc = []
            best_val_bacc = 0

            self.model.to(device)
            features = self.features.to(device)
            adj = self.adj.to(device)
            labels = self.labels.to(device)
            idx_temp = self.idx_temp.to(device)
            idx_test = self.idx_test.to(device)
            self.generator.to(device)

            for epoch_gen in range(self.config['epochs_gen']):
                part = epoch_gen % num
                range_val_maj = range(int(part*len(self.majority)/num), int((part+1)*len(self.majority)/num))
                range_val_min = range(int(part * len(self.minority) / num), int((part + 1) * len(self.minority) / num))

                range_train_maj = list(range(0,int(part*len(self.majority)/num)))+ list(range(int((part+1)*len(self.majority)/num),len(self.majority)))
                range_train_min = list(range(0,int(part*len(self.minority)/num)))+ list(range(int((part+1)*len(self.minority)/num),len(self.minority)))

                idx_val = torch.cat((self.majority[range_val_maj], self.minority[range_val_min]))
                idx_train = torch.cat((self.majority[range_train_maj], self.minority[range_train_min]))
                idx_train = torch.cat((idx_train, self.generate_node))
                num_real = features.shape[0] - len(idx_test) -len(idx_val)

                # Train model
                self.generator.train()
                self.optimizer_G.zero_grad()
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.generate_node.shape[0], 100))))
                z = z.to(device)

                adj_min = self.generator(z)
                gen_imgs1 = torch.mm(F.softmax(adj_min[:,0:self.minority.shape[0]], dim=1), features[self.minority])
                gen_imgs1_all = torch.mm(F.softmax(adj_min, dim=1), features[self.minority_all])

                matr = F.softmax(adj_min[:,0:self.minority.shape[0]], dim =1).data.cpu().numpy()
                pos=np.where(matr>1/matr.shape[1])
                adj_temp = sp.coo_matrix((np.ones(pos[0].shape[0]),(self.generate_node[pos[0]].numpy(), self.minority_all[pos[1]].numpy())),
                                        shape=(labels.shape[0], labels.shape[0]),
                                        dtype=np.float32)

                adj_new = self.__add_edges(self.adj_real, adj_temp)
                adj_new = adj_new.to(device)

                # model.eval()
                output, output_gen, output_AUC = self.model(torch.cat((features, gen_imgs1.data),0), adj)

                labels_true = torch.LongTensor(num_false).fill_(0)
                labels_min = torch.LongTensor(num_false).fill_(1)
                labels_true = labels_true.to(device)
                labels_min = labels_min.to(device)

                g_loss = F.nll_loss(output_gen[self.generate_node], labels_true) \
                        + F.nll_loss(output[self.generate_node], labels_min) \
                        + self.__euclidean_dist(features[self.minority], gen_imgs1).mean()
                g_loss.backward()
                self.optimizer_G.step()

                best_test_result, max_idx, best_val_bacc, acc_test, bacc_test, precision_test, recall_test, map_test = [], 0, 0, 0, 0, 0, 0, 0
                acc_val, bacc_val, precision_val, recall_val, map_val = 0, 0, 0, 0, 0

                for epoch in range(self.config['ep']):
                    
                    features_new = torch.cat((features, gen_imgs1.data.detach()),0)

                    self.model.train()
                    self.optimizer.zero_grad()
                    output, output_gen, output_AUC = self.model(features_new, adj_new)
                    labels_true = torch.cat((torch.LongTensor(num_real).fill_(0), torch.LongTensor(num_false).fill_(1)))

                    labels_true = labels_true.to(device)

                    loss_dis = - self.__euclidean_dist(features_new[self.minority], features_new[self.majority]).mean()
                    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) \
                                + F.nll_loss(output_gen[idx_train], labels_true) \
                                +loss_dis

                    loss_train.backward()
                    self.optimizer.step()

                    if not self.config['fastmode']:
                        self.model.eval()
                        output, output_gen, output_AUC = self.model(features_new, adj_new)

                    acc_val, bacc_val, precision_val, recall_val, map_val = performance_measure(output[idx_val], labels[idx_val], pre='valid')

                    val_bacc.append(bacc_val)
                    max_idx = val_bacc.index(max(val_bacc))
                    if bacc_val > best_val_bacc:
                        best_val_bacc = bacc_val
                        output, output_gen, output_AUC = self.model(features_new, adj)
                        acc_tmp, bacc_tmp, precision_tmp, recall_tmp, map_tmp = performance_measure(output[idx_test], labels[idx_test], pre='valid')
                        acc_test = acc_tmp
                        bacc_test = bacc_tmp
                        precision_test = precision_tmp
                        recall_test = recall_tmp
                        map_test = map_tmp
                        best_test_result = [acc_test, bacc_test, precision_test, recall_test, map_test]

                        self.best_model = deepcopy(self.model)
                        self.save_model(self.best_model, f'{self.model_name}_best_model_on_{self.dataset_name}')
                        
                        self.best_generator = deepcopy(self.generator)
                        self.save_model(self.best_generator, f'{self.model_name}_best_generator_on_{self.dataset_name}')

                st = "[seed {}][{}][Epoch {}]".format(seed, self.model_name, epoch_gen)
                st += "[Val] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}|| ".format(acc_val, bacc_val, precision_val, recall_val, map_val)
                st += "[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(acc_test, bacc_test, precision_test, recall_test, map_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f},  bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}".format(
                    max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4])
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
                np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc), np.mean(precision), np.std(precision),
                np.mean(recall), np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger.info('ACC bACC Precision Recall mAP')
        self.logger.info('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc), np.mean(precision),
                                                                                            np.std(precision), np.mean(recall), np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger.info(self.config)
        self.logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    def __add_edges(self, adj_real, adj_new):

        adj = adj_real + adj_new
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def __euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
    def eval(self):

        self.best_generator.eval()
        self.best_model.eval()
        device = self.config['device']
        features = self.features.to(device)
        labels = self.labels.to(device)
        idx_test = self.idx_test.to(device)

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.generate_node.shape[0], 100))))
        z = z.to(device)
        adj_min = self.best_generator(z)
        gen_imgs1 = torch.mm(F.softmax(adj_min[:,0:self.minority.shape[0]], dim=1), features[self.minority])
        matr = F.softmax(adj_min[:,0:self.minority.shape[0]], dim =1).data.cpu().numpy()
        
        pos=np.where(matr>1/matr.shape[1])
        adj_temp = sp.coo_matrix((np.ones(pos[0].shape[0]),(self.generate_node[pos[0]].numpy(), self.minority_all[pos[1]].numpy())),
                                                shape=(labels.shape[0], labels.shape[0]),
                                                dtype=np.float32)
        adj_new = self.__add_edges(self.adj_real, adj_temp)
        adj_new = adj_new.to(device)          
        features_new = torch.cat((features, gen_imgs1.data.detach()),0)
        output, output_gen, output_AUC = self.model(features_new, adj_new)

        acc_tmp, bacc_tmp, precision_tmp, recall_tmp, map_tmp = performance_measure(output[idx_test], labels[idx_test], pre='valid')
        self.logger.log("[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(acc_tmp, bacc_tmp, precision_tmp, recall_tmp, map_tmp))
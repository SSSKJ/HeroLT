from HeroLT.nn.Wrappers import BaseModel
from HeroLT.nn.Dataloaders import GraphDataLoader
from HeroLT.nn.Models import graphSMOTE
from HeroLT.utils.logger import get_logger
from HeroLT.utils import seed_everything, performance_measure, classification, confusion

import torch.optim as optim

import numpy as np

from copy import deepcopy

class GraphSMOTE(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'GraphSMOTE',
            dataset = dataset,
            base_dir = base_dir)

        self.__load_config()
        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')


    def load_data(self):

        super().load_data()

        self.config, (self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test) = GraphDataLoader.load_data(self.config, self.dataset_name, self.model_name, f'{self.base_dir}/data/{self.dataset_name}/', self.logger)

    def __init_model(self):
        
        self.model = graphSMOTE(self.config, self.adj).to(self.config['device'])

    def load_pretrained_model(self):
        ## todo
        pass

    def __init_optimizer_and_scheduler(self):
        self.optimizer_fe = optim.Adam(self.model.encoder.parameters(), lr=self.config['lr'], weight_decay=self.config['wd'])  # feature extractor
        self.optimizer_ep = optim.Adam(self.model.decoder.parameters(), lr=self.config['lr'], weight_decay=self.config['wd'])  # edge predictor
        self.optimizer_cls = optim.Adam(self.model.classifier.parameters(), lr=self.config['lr'], weight_decay=self.config['wd'])  # node classifier

    def train(self):

        seed_result = {}
        seed_result['acc'] = []
        seed_result['bacc'] = []
        seed_result['precision'] = []
        seed_result['recall'] = []
        seed_result['mAP'] = []

        self.load_data()
        
        for seed in range(self.config['rnd'], self.config['rnd'] + self.config['num_seed']):
            self.logger(f'============== seed:{seed} ==============')
            seed_everything(seed)
            self.logger(f'seed: {seed}')

            self.__init_model()
            self.__init_optimizer_and_scheduler()
            
            # pretrain
            pretrain_losses = []
            self.model.train()
            for epoch in range(self.config['ep_pre']):
                self.optimizer_fe.zero_grad()
                self.optimizer_ep.zero_grad()

                loss = self.model(self.features, self.adj, self.labels, self.idx_train, pretrain=True)
                loss.backward()

                self.optimizer_fe.step()
                self.optimizer_ep.step()

                if epoch % 100 == 0:
                    self.logger("[Pretrain][Epoch {}] Recon Loss: {}".format(epoch, loss.item()))

                pretrain_losses.append(loss.item())
                min_idx = pretrain_losses.index(min(pretrain_losses))
                if epoch - min_idx > 500:
                    self.logger("Pretrain converged")
                    break
           

            # Main training
            val_bacc = []
            test_results = []

            best_metric = 0

            for epoch in range(self.config['ep']):
                self.model.train()
                self.optimizer_fe.zero_grad()
                self.optimizer_cls.zero_grad()
                self.optimizer_ep.zero_grad()

                loss_reconstruction, loss_nodeclassification = self.model(self.features, self.adj, self.labels, self.idx_train)

                loss = loss_nodeclassification + self.config['rw'] * loss_reconstruction
                loss.backward()

                self.optimizer_fe.step()
                self.optimizer_ep.step()
                self.optimizer_cls.step()

                # Evaluation
                self.model.eval()
                embed = self.model.encoder(self.features)
                output = self.model.classifier(embed)

                acc_val, bacc_val, precision_val, recall_val, map_val = performance_measure(output[self.idx_val], self.labels[self.idx_val], pre='valid')

                val_bacc.append(bacc_val)
                max_idx = val_bacc.index(max(val_bacc))

                if best_metric <= bacc_val:
                    best_metric = bacc_val
                    best_model = deepcopy(self.model)

                # Test
                acc_test, bacc_test, precision_test, recall_test, map_test= performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')

                test_results.append([acc_test, bacc_test, precision_test, recall_test, map_test])
                best_test_result = test_results[max_idx]

                st = "[seed {}][{}][Epoch {}]".format(seed, 'GraphSMOTE', epoch)
                st += "[Val] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}|| ".format(
                    acc_val, bacc_val, precision_val, recall_val, map_val)
                st += "[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(
                    acc_test, bacc_test, precision_test, recall_test, map_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f},  bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}".format(
                    max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4])
                   
                if epoch % 100 == 0:
                    self.logger(st)

                if (epoch - max_idx > self.config['ep_early']) or (epoch+1 == self.config['ep']):
                    if epoch - max_idx > self.config['ep_early']:
                        self.logger("Early stop")
                    embed = best_model.encoder(self.features)
                    output = best_model.classifier(embed)
                    best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4] = performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')
                    # acc_list, macro_F_list, gmean_list, bacc_list = utils.performance_per_class(output[self.idx_test],
                    #                                                                             self.labels[
                    #                                                                                 self.idx_test],
                    #                                                                             pre='test')
                    self.logger("[Best Test Result] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}".format(best_test_result[0], best_test_result[1], best_test_result[2],
                                                                                                                                best_test_result[3], best_test_result[4]))
                    self.logger(classification(output[self.idx_test], self.labels[self.idx_test].detach().cpu()))
                    self.logger(confusion(output[self.idx_test], self.labels[self.idx_test].detach().cpu()))
                    break

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['bacc'].append(float(best_test_result[1]))
            seed_result['precision'].append(float(best_test_result[2]))
            seed_result['recall'].append(float(best_test_result[3]))
            seed_result['mAP'].append(float(best_test_result[4]))
            # self.logger("[Best Test Result per class] ACC: {}, Macro-F1: {}, G-Means: {}, bACC: {}".format(
            #     acc_list, macro_F_list, gmean_list, bacc_list), file=text)
            # self.logger("[Best Test Result per class] ACC: {}, Macro-F1: {}, G-Means: {}, bACC: {}".format(acc_list, macro_F_list, gmean_list, bacc_list))

        acc = seed_result['acc']
        bacc = seed_result['bacc']
        precision = seed_result['precision']
        recall = seed_result['recall']
        mAP = seed_result['mAP']

        self.logger(
            '[Averaged result] ACC: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}, Precision: {:.1f}+{:.1f}, Recall: {:.1f}+{:.1f}, mAP: {:.1f}+{:.1f}'.format(
                np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc), np.mean(precision), np.std(precision),
                np.mean(recall), np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger('ACC bACC Precision Recall mAP')
        self.logger('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc),
                                                                                             np.mean(precision), np.std(precision), np.mean(recall),
                                                                                             np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger(self.config)
        



    
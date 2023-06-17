from . import BaseModel
from ..Dataloaders import GraphDataLoader
from ...utils.logger import get_logger
from ..Models import lTE4G
from ...utils import seed_everything, performance_measure, scheduler

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from copy import deepcopy
import numpy as np

class LTE4G(BaseModel):

    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'LTE4G',
            dataset_name = dataset,
            base_dir = base_dir)
        

        super().load_config()
        self.logger = get_logger(self.base_dir, f'{self.model_name}_{self.dataset_name}.log')
        self.loaded = False


    def load_data(self):

        super().load_data()

        self.config, (self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test, self.class_num_mat, self.idx_train_set, self.idx_val_set, self.idx_test_set, self.degrees) = GraphDataLoader.load_data(self.config, self.dataset_name, self.model_name, f'{self.base_dir}/data/GraphData/', self.logger)

    def __init_model(self):
        
        self.model = lTE4G(self.config, self.adj).to(self.config['device'])

    def load_pretrained_model(self):

        self.loaded = True
        ## todo: load pretrained model
        ####### Load Pre-trained Original Imbalance graph #######
        # self.logger.info('Load Pre-trained Encoder')       
        # rec_with_ep_pre = 'True_ep_pre_' + str(self.config['ep_pre']) + '_rw_' + str(self.config['rw']) if self.config['rec'] else 'False'
        # encoder_info = f'cls_{self.config['cls_og']}_cw_{self.config['class_weight']}_gamma_{self.config['gamma']}_alpha_{self.config['alpha}_lr_{self.config['lr}_dropout_{self.config['dropout}_rec_{rec_with_ep_pre}_seed_{seed}.pkl'
        # if self.config['im_ratio != 1: # manual
        #     pretrained_encoder = torch.load(f'./pretrained/manual/{self.config['dataset}/{self.config['im_class_num}/{self.config['im_ratio}/'+encoder_info)
        # else: # natural
        #     pretrained_encoder = torch.load(f'./pretrained/natural/{self.config['dataset}/'+encoder_info)

        # model.load_state_dict(pretrained_encoder.state_dict())
        pass

    ## optimizer for pretrained part
    def __init_optimizer_and_scheduler(self):
        self.optimizer_fe = optim.Adam(self.model.encoder.parameters(), lr=self.config['lr'], weight_decay=self.config['wd']) # feature extractor
        self.optimizer_cls_og = optim.Adam(self.model.classifier_og.parameters(), lr=self.config['lr'], weight_decay=self.config['wd'])   
        if self.config['rec']:
            self.optimizer_ep = optim.Adam(self.model.decoder.parameters(), lr=self.config['lr'] , weight_decay=self.config['wd']) # edge prediction

    
    def train(self):

        seed_result = {}

        seed_result['acc'] = []
        seed_result['bacc'] = []
        seed_result['precision'] = []
        seed_result['recall'] = []
        seed_result['mAP'] = []

        self.load_data()
        
        for seed in range(self.config['rnd'], self.config['rnd'] + self.config['num_seed']):
            self.logger.info(f'============== seed:{seed} ==============')
            seed_everything(seed)
            self.logger.info(f'seed: {seed}')
            self.logger.info(f'Samples per label: {list(self.class_num_mat[:,0])}')

            self.logger.info('-----Number of training samples in each Expert-----')
            self.logger.info('HH: %s' % len(self.idx_train_set['HH']))
            self.logger.info('HT: %s' % len(self.idx_train_set['HT']))
            self.logger.info('TH: %s' %  len(self.idx_train_set['TH']))
            self.logger.info('TT: %s' % len(self.idx_train_set['TT']))

            self.__init_model()

            self.degrees = torch.tensor(self.degrees).to(self.config['device'])
            avg_degree = []
            for i, label in enumerate(self.labels.unique()):
                avg_degree.append((sum(self.degrees[self.idx_train][self.labels[self.idx_train] == label]) / sum(self.labels[self.idx_train] == label)).item())
            
            avg_degree = torch.tensor(avg_degree).to(self.config['device'])
            class_weight = 1 / torch.tensor(self.class_num_mat[:,0]).to(self.config['device'])

            # ground truch of head / tail separation
            data_set = [self.idx_train, self.idx_val, self.idx_test]
            ht_gt = {}
            for data in data_set:
                ht_gt[data] = (self.labels[data] >= self.config['sep_point']).type(torch.long) # head to '0', tail to '1'
        
            # ======================================= Encoder Training ======================================= #
            if not self.loaded:
                ####### Pre-train Original Imbalance graph #######
                self.logger.info('Start Pre-training Encoder')       
                best_encoder = None
                self.__init_optimizer_and_scheduler()

                # pretrain encoder & decoder (adopted by GraphSMOTE)
                if self.config['rec']:
                    self.model.train()
                    for epoch in range(self.config['ep_pre']):
                        self.optimizer_fe.zero_grad()
                        self.optimizer_ep.zero_grad()
                        
                        loss = self.model(self.features, self.adj, pretrain=True)
                        loss.backward(retain_graph=True)

                        self.optimizer_fe.step()
                        self.optimizer_ep.step()

                        if epoch % 100 == 0:
                            self.logger.info("[Pretrain][Epoch {}] Recon Loss: {}".format(epoch, loss.item()))

                val_bacc_og = []
                test_results = []

                best_metric = 0.0

                for epoch in trange(self.config['ep']):
                    self.model.train()
                    self.optimizer_fe.zero_grad()
                    self.optimizer_cls_og.zero_grad()

                    if self.config['rec']:
                        self.optimizer_ep.zero_grad()
                        loss_nodeclassification, loss_reconstruction = self.model(self.features, self.adj, labels=self.labels, idx_train=self.idx_train, weight=class_weight, is_og=True)
                        loss = loss_nodeclassification + self.config['rw'] * loss_reconstruction
                        loss.backward(retain_graph=True)
                        
                        self.optimizer_fe.step()
                        self.optimizer_ep.step()
                        self.optimizer_cls_og.step()

                    else:
                        loss_nodeclassification = self.model(self.features, labels=self.labels, idx_train=self.idx_train, weight=class_weight, is_og=True)
                        loss = loss_nodeclassification
                        loss.backward(retain_graph=True)
                        
                        self.optimizer_fe.step()
                        self.optimizer_cls_og.step()

                    # Evaluation
                    self.model.eval()
                    embed = self.model.encoder(self.features)
                    output_original = self.model.classifier_og(embed)

                    _, bacc_val, _, _, _ = performance_measure(output_original[self.idx_val], self.labels[self.idx_val], pre='valid')
                    
                    val_bacc_og.append(bacc_val)
                    max_idx = val_bacc_og.index(max(val_bacc_og))

                    if best_metric <= bacc_val:
                        best_metric = bacc_val
                        best_encoder = deepcopy(self.model)
                    
                    if (epoch - max_idx > self.config['ep_early']) or (epoch+1 == self.config['ep']):
                        if epoch - max_idx > self.config['ep_early']:
                            self.logger.info("Early stop")
                        break

                # todo: Save pre-trained encoder
                # if self.config['save_encoder']:
                #     self.logger.info('Saved pre-trained Encoder')
                #     rec_with_ep_pre = 'True_ep_pre_' + str(self.config['ep_pre']) + '_rw_' + str(self.config['rw']) if self.config['rec'] else 'False'
                #     encoder_info = f'cls_{self.config['cls_og}_cw_{self.config['class_weight}_gamma_{self.config['gamma}_alpha_{self.config['alpha}_lr_{self.config['lr}_dropout_{self.config['dropout}_rec_{rec_with_ep_pre}_seed_{seed}.pkl'
                #     if self.config['im_ratio'] != 1: # manual
                #         os.makedirs(f'./pretrained/manual/{self.config['dataset}/{self.config['im_class_num}/{self.config['im_ratio}', exist_ok=True)
                #         pretrained_encoder = torch.save(best_encoder, f'./pretrained/manual/{self.config['dataset}/{self.config['im_class_num}/{self.config['im_ratio}/'+encoder_info)
                #     else: # natural
                #         os.makedirs(f'./pretrained/natural/{self.config['dataset}', exist_ok=True)
                #         pretrained_encoder = torch.save(best_encoder, f'./pretrained/natural/{self.config['dataset}/'+encoder_info)

                self.model = best_encoder

            # ======================================= Head/Tail Separation & Class Prtotypes ======================================= #
            self.model.eval()
            embed = self.model.encoder(self.features)
            prediction = self.model.classifier_og(embed)
            prediction = torch.softmax(prediction, 1)

            centroids = torch.empty((self.config['nclass'], embed.shape[1])).to(embed.device)

            for i, label in enumerate(self.labels.unique()):
                resources = []
                centers = list(map(int, self.idx_train[self.labels[self.idx_train] == label]))
                if self.idx_train[self.labels[self.idx_train] == label].shape[0] > 0:
                    resources.extend(centers)
                    adj_dense = self.adj.to_dense()[centers]
                    adj_dense[adj_dense>0] = 1

                    similar_matrix = (F.normalize(self.features) @ F.normalize(self.features).T)[centers]
                    similar_matrix -= adj_dense

                    if self.config['criterion'] == 'mean':
                        avg_num_candidates = int(sum(self.class_num_mat[:,0]) / len(self.class_num_mat[:,0]))
                    elif self.config['criterion'] == 'median':
                        avg_num_candidates = int(np.median(self.class_num_mat[:,0]))
                    elif self.config['criterion'] == 'max':
                        avg_num_candidates = max(self.class_num_mat[:,0])

                    if self.class_num_mat[i,0] < avg_num_candidates:
                        num_candidates_to_fill = avg_num_candidates - self.class_num_mat[i,0]
                        neighbors = np.array(list(set(map(int,self.adj.to_dense()[centers].nonzero()[:,1])) - set(centers)))
                        similar_nodes = np.array(list(set(map(int,similar_matrix.topk(10+1)[1][:,1:].reshape(-1)))))

                        # Candidate Selection
                        candidates_by_neighbors = prediction.cpu()[neighbors, i].sort(descending=True)[1][:num_candidates_to_fill]
                        resource = neighbors[candidates_by_neighbors]
                        if len(candidates_by_neighbors) != 0:
                            resource = [resource] if len(candidates_by_neighbors) == 1 else resource
                            resources.extend(resource)
                        if len(resources) < num_candidates_to_fill:
                                num_candidates_to_fill = num_candidates_to_fill - len(resources)
                                candidates_by_similar_nodes = prediction.cpu()[similar_nodes, i].sort(descending=True)[1][:num_candidates_to_fill]
                                resource = similar_nodes[candidates_by_similar_nodes]
                                if len(candidates_by_similar_nodes) != 0:
                                    resource = [resource] if len(candidates_by_similar_nodes) == 1 else resource
                                    resources.extend(resource)

                    resource = torch.tensor(resources)

                    centroids[i, :] = embed[resource].mean(0)
            
            similarity = (F.normalize(embed) @ F.normalize(centroids).t())

            # Top-1 Similarity
            sim_top1_val = torch.argmax(similarity[self.idx_val], 1).long() # top 1 similarity
            sim_top1_test = torch.argmax(similarity[self.idx_test], 1).long() # top 1 similarity

            idx_val_ht_pred = (sim_top1_val >= self.config['sep_point']).long()
            idx_test_ht_pred = (sim_top1_test >= self.config['sep_point']).long()
            
            idx_class = {}
            for index in [self.idx_val, self.idx_test]:
                idx_class[index] = {}

            idx_class[self.idx_val]['H'] = self.idx_val[(idx_val_ht_pred == 0)].detach().cpu()
            idx_class[self.idx_val]['T'] = self.idx_val[(idx_val_ht_pred == 1)].detach().cpu()

            idx_class[self.idx_test]['H'] = self.idx_test[(idx_test_ht_pred == 0)].detach().cpu()
            idx_class[self.idx_test]['T'] = self.idx_test[(idx_test_ht_pred == 1)].detach().cpu()
            

            # ======================================= Expert Training ======================================= #
            classifier_dict = {}
            for sep in ['HH', 'HT', 'TH', 'TT']:
                idx_train = self.idx_train_set[sep]
                idx_val = self.idx_val_set[sep]
                idx_test = self.idx_test_set[sep]

                best_metric_expert = -1
                max_idx = 0
                val_bacc_teacher = []
                test_results = []
                
                if sep[1] == 'T':
                    # if degree belongs to tail, finetune head degree classifier
                    classifier = deepcopy(classifier_dict[sep[0] + 'H'])
                else:
                    classifier = self.model.expert_dict[sep].to(self.config['device'])
                optimizer = optim.Adam(classifier.parameters(), lr=self.config['lr_expert'], weight_decay=self.config['wd'])
                
                for epoch in range(self.config['expert_ep']):
                    classifier.train()
                    optimizer.zero_grad()

                    loss = self.model(self.features, labels=self.labels, idx_train=idx_train, classifier=classifier, sep=sep, is_expert=True)
                    
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Evaluation
                    classifier.eval()
                    output = classifier(embed)

                    acc_val, bacc_val, precision_val, recall_val, map_val = (0,0,0,0,0) if len(idx_val) ==0 else performance_measure(output[idx_val], self.labels[idx_val], sep_point=self.config['sep_point'], sep=sep, pre='valid')
                    
                    val_bacc_teacher.append(bacc_val)
                    max_idx = val_bacc_teacher.index(max(val_bacc_teacher))

                    if best_metric_expert <= bacc_val:
                        best_metric_expert = bacc_val
                        classifier_dict[sep] = deepcopy(classifier) # save best model

                    # Test
                    acc_test, bacc_test, precision_test, recall_test, map_test= (0,0,0,0,0) if len(idx_test) == 0 else performance_measure(output[idx_test], self.labels[idx_test], sep_point=self.config['sep_point'], sep=sep, pre='test')

                    test_results.append([acc_test, bacc_test, precision_test, recall_test, map_test])
                    best_test_result = test_results[max_idx]

                    st = "[seed {}][{}][Expert-{}][Epoch {}]".format(seed, self.model_name, sep, epoch)
                    st += "[Val] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}|| ".format(
                        acc_val, bacc_val, precision_val, recall_val, map_val)
                    st += "[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(
                        acc_test, bacc_test, precision_test, recall_test, map_test)
                    st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f},  bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}".format(
                        max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4])
                    
                    if epoch % 100 == 0:
                        self.logger.info(st)

                    if (epoch - max_idx >= 300) or (epoch + 1 == self.config['ep']):
                        if epoch - max_idx >= 300:
                            self.logger.info('Early Stop!')
                        break
                    

            # ======================================= Student Training ======================================= #
            for sep in ['H', 'T']:
                classifier = self.model.expert_dict[sep].to(self.config['device'])
                optimizer = optim.Adam(classifier.parameters(), lr=self.config['lr_expert'], weight_decay=self.config['wd'])

                # set idx_train
                idx_train = torch.cat((self.idx_train_set[sep + 'H'], self.idx_train_set[sep + 'T']), 0)
                idx_val = torch.cat((self.idx_val_set[sep + 'H'], self.idx_val_set[sep + 'T']), 0)
                idx_test = torch.cat((self.idx_test_set[sep + 'H'], self.idx_test_set[sep + 'T']), 0)

                best_metric_student = -1
                max_idx = 0
                val_bacc_student = []
                test_results = []
                
                for epoch in range(self.config['curriculum_ep']):
                    classifier.train()
                    optimizer.zero_grad()

                    kd_head, kd_tail, ce_loss = self.model(self.features, labels=self.labels, idx_train=self.idx_train_set, embed=embed, classifier=classifier, sep=sep, teacher=classifier_dict, is_student=True)
                    alpha = scheduler(epoch, self.config['curriculum_ep'])

                    # Head-to-Tail Curriculum Learning
                    loss = ce_loss + (alpha * kd_head + (1-alpha) * kd_tail)

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Evaluation
                    classifier.eval()
                    output = classifier(embed)

                    acc_val, bacc_val, precision_val, recall_val, map_val = (0,0,0,0,0) if len(idx_val) ==0 else performance_measure(output[idx_val], self.labels[idx_val], sep_point=self.config['sep_point'], sep=sep, pre='valid')
                    
                    val_bacc_student.append(bacc_val)
                    max_idx = val_bacc_student.index(max(val_bacc_student))

                    if best_metric_student <= bacc_val:
                        best_metric_student = bacc_val
                        classifier_dict[sep] = deepcopy(classifier) # save best model

                    # Test
                    acc_test, bacc_test, precision_test, recall_test, map_test= (0,0,0,0,0) if len(idx_test) == 0 else performance_measure(output[idx_test], self.labels[idx_test], sep_point=self.config['sep_point'], sep=sep, pre='test')

                    test_results.append([acc_test, bacc_test, precision_test, recall_test, map_test])
                    best_test_result = test_results[max_idx]

                    st = "[seed {}][{}][Student-{}][Epoch {}]".format(seed, self.model_name, sep, epoch)
                    st += "[Val] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}|| ".format(
                        acc_val, bacc_val, precision_val, recall_val, map_val)
                    st += "[Test] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}\n".format(
                        acc_test, bacc_test, precision_test, recall_test, map_test)
                    st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f},  bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}".format(
                        max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3], best_test_result[4])
                    
                    if epoch % 100 == 0:
                        self.logger.info(st)

                    if epoch + 1 == self.config['curriculum_ep']:
                        break

            # ======================================= Inference Phase =======================================
            final_pred = torch.add(torch.zeros((self.idx_test.shape[0], self.config['nclass'])), -999999)
            
            test_to_idx = {}
            for i in range(len(self.idx_test)):
                test = int(self.idx_test[i])
                test_to_idx[test] = i
            
            for sep in ['H', 'T']:
                idx_test = list(map(int,idx_class[self.idx_test][sep]))
                student = classifier_dict[sep]
                student.eval()
                pred = student(embed)

                idx_mapped = list(map(lambda x: test_to_idx[x], idx_test))
                if sep == 'H':
                    final_pred[idx_mapped, 0:self.config['sep_point']] = pred[idx_test].cpu()
                elif sep == 'T':
                    final_pred[idx_mapped, self.config['sep_point']:self.config['nclass']] = pred[idx_test].cpu()

            acc, bacc, precision, recall, mAP = performance_measure(final_pred, self.labels[self.idx_test], pre='test')

            self.logger.info('=======================================================')
            self.logger.info('[LTE4G] ACC: {:.1f}, bACC: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, mAP: {:.1f}'.format(acc, bacc, precision, recall, mAP))

            seed_result['acc'].append(float(acc))
            seed_result['bacc'].append(float(bacc))
            seed_result['precision'].append(float(precision))
            seed_result['recall'].append(float(recall))
            seed_result['mAP'].append(float(mAP))
    
        acc = seed_result['acc']
        bacc = seed_result['bacc']
        precision = seed_result['precision']
        recall = seed_result['recall']
        mAP = seed_result['mAP']

        self.logger.info('ACC bACC Precision Recall mAP')
        self.logger.info('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(bacc), np.std(bacc),
                                                                                             np.mean(precision), np.std(precision), np.mean(recall),
                                                                                             np.std(recall), np.mean(mAP), np.std(mAP)))
        self.logger.info(self.config)
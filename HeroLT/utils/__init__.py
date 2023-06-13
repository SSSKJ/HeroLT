import importlib
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sklearn
from sklearn.metrics import f1_score

import numpy as np
import scipy.sparse as sp

import os.path as osp


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_count(data: DataLoader):
    
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num

def special_mkdir(root, name):

    if name not in os.listdir(root):
        os.mkdir(f'{root}/{name}')

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x
    
def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1

def weighted_mic_acc_cal(preds, labels, ws):
    acc_mic_top1 = ws[preds == labels].sum() / ws.sum()
    return acc_mic_top1

def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score

def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce

def get_priority(ptype, logits, labels):
    if ptype == 'score':
        ws = 1 - logits2score(logits, labels)
    elif ptype == 'entropy':
        ws = logits2entropy(logits)
    elif ptype == 'CE':
        ws = logits2CE(logits, labels)
    
    return ws

def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
    
def weighted_shot_acc(preds, labels, ws, train_data, many_shot_thr=100, low_shot_thr=20):
    
    training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(ws[labels==l].sum())
        class_correct.append(((preds[labels==l] == labels[labels==l]) * ws[labels==l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))          
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def F_measure(preds, labels, theta=None):
    
    return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)   
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model


import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score, \
    precision_score, recall_score, average_precision_score
from imblearn.metrics import geometric_mean_score
import os.path as osp
import os
import logging
import sys
import scipy.sparse as sp

def split_natural(labels, idx_map):
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        c_num = len(c_idx)

        if c_num < 10:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i,0] = 0
            c_num_mat[i,1] = 0
            c_num_mat[i,2] = c_num
            # print('[{}-th class] Total: {} | Train: {} | Val: {} | Test: {}'.format(i,len(c_idx), c_num_mat[i,0], c_num_mat[i,1], c_num_mat[i,2]))
            # train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
            # val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
            test_idx = test_idx + c_idx[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]
        else:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i, 0] = int(c_num * 0.1)  # 10% for train
            c_num_mat[i, 1] = int(c_num * 0.1)  # 10% for validation
            c_num_mat[i, 2] = int(c_num * 0.8)  # 80% for test
            train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
            val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
            test_idx = test_idx + c_idx[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    random.shuffle(train_idx)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def separate_class_degree(adj, idx_train_set_class, above_head=None, below_tail=None, below=None, rand=False, is_eval=False):
    idx_train_set = {}
    idx_train_set['HH'] = []
    idx_train_set['HT'] = []
    idx_train_set['TH'] = []
    idx_train_set['TT'] = []

    adj_dense = adj.to_dense()
    adj_dense[adj_dense != 0] = 1
    degrees = np.array(list(map(int, torch.sum(adj_dense, dim=0))))

    if rand:
        for sep in ['H', 'T']:
            idxs = np.array(idx_train_set_class[sep])
            np.random.shuffle(idxs)
        
            idx_train_set[sep+'H'] = idxs[:int(len(idxs)/2)]
            idx_train_set[sep+'T'] = idxs[int(len(idxs)/2):]
            
        for idx in ['HH', 'HT', 'TH', 'TT']:
            random.shuffle(idx_train_set[idx])
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

            degree_dict = {}
            above_head = 0
            below_tail = 0
        
        return idx_train_set, degree_dict, degrees, above_head, below_tail


    if not is_eval:
        above_head = {}
        below_tail = {}
        degree_dict = {}

        for sep in ['H', 'T']:
            if len(idx_train_set_class[sep]) == 0:
                continue

            elif len(idx_train_set_class[sep]) == 1:
                idx = idx_train_set_class[sep]
                if sep == 'H':
                    rand = random.choice(['HH', 'HT'])
                    idx_train_set[rand].append(int(idx))
                elif sep == 'T':
                    rand = random.choice(['TH', 'TT'])
                    idx_train_set[rand].append(int(idx))

            else:
                degrees_idx_train = degrees[idx_train_set_class[sep]]

                above_head = below + 1
                below_tail = below
                gap_head = abs(degrees_idx_train - (below+1))
                gap_tail = abs(degrees_idx_train - below)

                if sep == 'H':
                    idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                    idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                    if sum(gap_head == gap_tail) > 0:
                        for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                            rand = random.choice(['HH', 'HT'])
                            idx_train_set[rand].append(int(idx))

                elif sep == 'T':
                    idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                    idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                    if sum(gap_head == gap_tail) > 0:
                        for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                            rand = random.choice(['TH', 'TT'])
                            idx_train_set[rand].append(int(idx))

        for idx in ['HH', 'HT', 'TH', 'TT']:
            random.shuffle(idx_train_set[idx])
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

        return idx_train_set, degree_dict, degrees, above_head, below_tail
    
    elif is_eval:
        for sep in ['H', 'T']:
            if len(idx_train_set_class[sep]) == 0:
                continue

            else:
                degrees_idx_train = degrees[idx_train_set_class[sep]]

                gap_head = abs(degrees_idx_train - above_head)
                gap_tail = abs(degrees_idx_train - below_tail)

                if sep == 'H':
                    if len(idx_train_set_class[sep]) == 1:
                        if gap_head < gap_tail:
                            idx_train_set['HH'].append(int(idx_train_set_class[sep]))
                        elif gap_tail < gap_head:
                            idx_train_set['HT'].append((idx_train_set_class[sep]))
                        else:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['HH', 'HT'])
                                idx_train_set[rand].append(int(idx))
                    else:
                        idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                        idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                        if sum(gap_head == gap_tail) > 0:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['HH', 'HT'])
                                idx_train_set[rand].append(int(idx))

                elif sep == 'T':
                    if len(idx_train_set_class[sep]) == 1:
                        if gap_head < gap_tail:
                            idx_train_set['TH'].append(int(idx_train_set_class[sep]))
                        elif gap_tail < gap_head:
                            idx_train_set['TT'].append(int(idx_train_set_class[sep]))
                        else:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['TH', 'TT'])
                                idx_train_set[rand].append(int(idx))
                    else:
                        idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                        idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                        if sum(gap_head == gap_tail) > 0:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['TH', 'TT'])
                                idx_train_set[rand].append(int(idx))
            
        for idx in ['HH', 'HT', 'TH', 'TT']:
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])
                
        return idx_train_set

def separate_eval(idx_eval, labels, ht_dict_class, degrees, above_head, below_tail):
    idx_eval_set = {}
    idx_eval_set['HH'] = []
    idx_eval_set['HT'] = []
    idx_eval_set['TH'] = []
    idx_eval_set['TT'] = []
    
    for idx in idx_eval:
        label = int(labels[idx])
        degree = int(degrees[idx])
        if (label in ht_dict_class['H']) and (degree >= above_head):
            idx_eval_set['HH'].append(int(idx))

        elif (label in ht_dict_class['H']) and (degree <= below_tail):
            idx_eval_set['HT'].append(int(idx))

        elif (label in ht_dict_class['T']) and (degree >= above_head):
            idx_eval_set['TH'].append(int(idx))

        elif (label in ht_dict_class['T']) and (degree <= below_tail):
            idx_eval_set['TT'].append(int(idx))
        
    
    for idx in ['HH', 'HT', 'TH', 'TT']:
        random.shuffle(idx_eval_set[idx])
        idx_eval_set[idx] = torch.LongTensor(idx_eval_set[idx])
            
    return idx_eval_set

def separate_ht(samples_per_label, labels, idx_train, method='pareto_28', rand=False, manual=False):
    class_dict = {}
    idx_train_set = {}

    if rand:
        ht_dict = {}
        arr = np.array(idx_train)
        np.random.shuffle(arr)
        sample_num = int(idx_train.shape[0]/2)
        sample_label_num = int(len(labels.unique())/2)
        label_list = np.array(labels.unique())
        np.random.shuffle(label_list)
        ht_dict['H'] = label_list[0:sample_label_num]
        ht_dict['T'] = label_list[sample_label_num:]

        idx_train_set['H'] = arr[0:sample_num]
        idx_train_set['T'] = arr[sample_num:]

    elif manual:
        ht_dict = {}
        samples = samples_per_label
        point = np.arange(len(samples_per_label)-1)[list(map(lambda x: samples[x] != samples[x+1], range(len(samples)-1)))][0]
        label_list = np.array(labels.unique())
        ht_dict['H'] = label_list[0:point+1]
        ht_dict['T'] = label_list[point+1:]

        print('Samples per label:', samples_per_label)
        print('Separation:', ht_dict.items())

        idx_train_set['H'] = []
        idx_train_set['T'] = []
        for label in label_list:
            idx = 'H' if label <= point else 'T'
            idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))
            
    else:
        ht_dict = separator_ht(samples_per_label, method) # H/T

        print('Samples per label:', samples_per_label)
        print('Separation:', ht_dict.items())

        for idx, value in ht_dict.items():
            class_dict[idx] = []
            idx_train_set[idx] = []
            idx = idx
            label_list = value

            for label in label_list:
                class_dict[idx].append(label)
                idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))
            
    for idx in list(ht_dict.keys()):
        random.shuffle(idx_train_set[idx])
        idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

    return idx_train_set, ht_dict


def separator_ht(dist, method='pareto_28', degree=False): # Head / Tail separator
    head = int(method[-2]) # 2 in pareto_28
    tail = int(method[-1]) # 8 in pareto_28
    head_idx = int(len(dist) * (head/10))
    ht_dict = {}

    if head_idx == 0:
        ht_dict['H'] = list(range(0, 1))
        ht_dict['T'] = list(range(1, len(dist)))
        return ht_dict

    else:
        crierion = dist[head_idx].item()

        case1_h = sum(np.array(dist) >= crierion)
        case1_t = sum(np.array(dist) < crierion)

        case2_h = sum(np.array(dist) > crierion)
        case2_t = sum(np.array(dist) <= crierion)

        gap_case1 = abs(case1_h/case1_t - head/tail)
        gap_case2 = abs(case2_h/case2_t - head/tail)

        if gap_case1 < gap_case2:
            idx = sum(np.array(dist) >= crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        elif gap_case1 > gap_case2:
            idx = sum(np.array(dist) > crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        else:
            rand = random.choice([1, 2])
            if rand == 1:
                idx = sum(np.array(dist) >= crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))
            else:
                idx = sum(np.array(dist) > crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))

        return ht_dict

def Graph_accuracy(output, labels, sep_point=None, sep=None, pre=None):
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    if correct > len(labels):
        print("wrong")
    return correct / len(labels)

def mean_average_precision(output, labels, sep_point=None, sep=None):
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    return average_precision_score(F.one_hot(labels, output.shape[1]), output, average='macro')

def classification(output, labels, sep_point=None, sep=None):
    target_names = []
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        pred = output.max(1)[1].type_as(labels)
        for i in labels.unique():
            target_names.append(f'class_{int(i)}')

        return classification_report(labels, pred)

def confusion(output, labels, sep_point=None, sep=None):
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        
        pred = output.max(1)[1].type_as(labels)
    
        return confusion_matrix(labels, pred)

def performance_measure(output, labels, sep_point=None, sep=None, pre=None):
    acc = Graph_accuracy(output, labels, sep_point=sep_point, sep=sep, pre=pre)*100
    mAP = mean_average_precision(output.cpu().detach(), labels.cpu().detach(), sep_point=sep_point, sep=sep) * 100

    if len(labels) == 0:
        return np.nan
    
    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)
    
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    # macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    # gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    precision = precision_score(labels.cpu().detach(), output.cpu().detach(), average='macro') * 100
    recall = recall_score(labels.cpu().detach(), output.cpu().detach(), average='macro') * 100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach())*100

    return acc, bacc, precision, recall, mAP

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    
    adj_tgt[adj_tgt != 0] = 1

    edge_num = adj_tgt.nonzero().shape[0] #number of non-zero
    total_num = adj_tgt.shape[0]**2 #possible edge

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2) # element-wise

    return loss

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def refine_label_order(labels):
    print('Refine label order, Many to Few')
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]:i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.numpy())

    return labels.new(new_labels), idx_map

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    return sum_m 

def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()
    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    
    return adj

def normalize_sym(adj):
    """Symmetric-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()

    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    adj = torch.spmm(adj, deg_inv_sqrt.to_dense()).to_sparse()

    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse, shape):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()
    sp_matrix = sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    return sp_matrix

def scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1/2) * np.cos((epoch*np.pi) / curriculum_ep) + 1/2

def setupt_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")
    return logger

def set_filename(args):
    rec_with_ep_pre = 'True_ep_pre_' + str(args.ep_pre) + '_rw_' + str(args.rw) if args.rec else 'False'

    if args.im_ratio == 1: # Natural Setting
        results_path = f'./results/natural/{args.dataset}'
        logs_path = f'./logs/natural/{args.dataset}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/natural/{args.dataset}/({args.layer}){textname}', 'w')
        file = f'./logs/natural/{args.dataset}/({args.layer})lte4g.txt'
        
    else: # Manual Imbalance Setting (0.2, 0.1, 0.05)
        results_path = f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        logs_path = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer}){textname}', 'w')
        file = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer})lte4g.txt'
        
    return text, file

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_sp_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def performance_per_class(output, labels, sep_point=None, sep=None, pre=None):
    acc_list, macro_F_list, gmean_list, bacc_list = [], [], [], []
    if len(labels) == 0:
        return np.nan
    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point  # [4,5,6] -> [0,1,2]

    num_classes = len(set(labels.tolist()))
    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        acc = Graph_accuracy(output[c_idx], labels[c_idx], sep_point=sep_point, sep=sep, pre=pre) * 100
        macro_F = f1_score(labels[c_idx].cpu().detach(), output[c_idx].cpu().detach(), average='macro') * 100
        gmean = geometric_mean_score(labels[c_idx].cpu().detach(), output[c_idx].cpu().detach(), average='macro') * 100
        bacc = balanced_accuracy_score(labels[c_idx].cpu().detach(), output[c_idx].cpu().detach()) * 100
        acc_list.append('%.1f' % acc)
        macro_F_list.append('%.1f' % macro_F)
        gmean_list.append('%.1f' % gmean)
        bacc_list.append('%.1f' % bacc)

    return acc_list, macro_F_list, gmean_list, bacc_list


def tailGNN_normalize_adj(adj, norm_type=1, iden=False):
    # 1: mean norm, 2: spectral norm
    # add the diag into adj, namely, the self-connection. then normalization
    if iden:
        adj = adj + np.eye(adj.shape[0])  # self-loop
    if norm_type == 1:
        D = np.sum(adj, axis=1)
        adjNor = adj / D
        adjNor[np.isinf(adjNor)] = 0.
    else:
        adj[adj > 0.0] = 1.0
        D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5))
        adjNor = np.dot(np.dot(D_, adj), D_)

    return adjNor, adj


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def tailGNN_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = np.where(rowsum == 0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = np.where(rowsum == 0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    if sp.issparse(features):
        return features.todense()
    else:
        return features


def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_to_torch_tensor(features, adj, tail_adj, labels, idx_train, idx_val, idx_test):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = convert_sparse_tensor(adj)  # + sp.eye(adj.shape[0]))
    tail_adj = convert_sparse_tensor(tail_adj)  # + sp.eye(tail_adj.shape[0])
    iden = sp.eye(adj.shape[0])
    iden = convert_sparse_tensor(iden)

    return features, adj, tail_adj, iden, labels, idx_train, idx_val, idx_test


def link_dropout(adj, idx, k=5):
    tail_adj = adj.copy()
    num_links = np.random.randint(k, size=idx.shape[0])
    num_links += 1

    for i in range(idx.shape[0]):
        index = tail_adj[idx[i]].nonzero()[0]
        new_idx = np.random.choice(index, min(num_links[i], len(index)), replace=False)
        tail_adj[idx[i]] = 0.0
        for j in new_idx:
            tail_adj[idx[i], j] = 1.0
    return tail_adj


# split head vs tail nodes
def split_nodes(idx, adj, k=5):
    num_idx_links = np.sum(adj[idx], axis=1)
    idx_train = np.where(num_idx_links > k)[0]

    num_links = np.sum(adj, axis=1)
    idx_valtest = np.where(num_links <= k)[0]
    np.random.shuffle(idx_valtest)

    p = int(idx_valtest.shape[0] / 3)
    idx_val = idx_valtest[:p]
    idx_test = idx_valtest[p:]

    return idx_train, idx_val, idx_test


# Tang Kaihua New Add
def print_grad_norm(named_parameters, verbose=False):
    if not verbose:
        return None

    total_norm = 0.0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters.items():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)

    print('----------Total norm {:.5f}-----------------'.format(total_norm))
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
        print("{:<50s}: {:.5f}, ({})".format(name, norm, param_to_shape[name]))
    print('-------------------------------')

    return total_norm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

def calibration(true_labels, pred_labels, confidences, num_bins=15):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
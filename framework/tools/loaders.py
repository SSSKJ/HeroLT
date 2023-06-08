from torchvision import transforms
from torch.utils.data import DataLoader

from .datasets import *
from utils import *

import numpy as np

# Data Loader definiation from Decoupling
class Decoupling_loader:

    # Image statistics
    RGB_statistics = {
        'inatural2018': {
            'mean': [0.466, 0.471, 0.380],
            'std': [0.195, 0.194, 0.192]
        },
        'default': {
            'mean': [0.485, 0.456, 0.406],
            'std':[0.229, 0.224, 0.225]
        }
    }

    # Data transformation with augmentation
    @classmethod
    def get_data_transform(cls, split, rgb_mean, rbg_std, key='default'):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]) if key == 'inatural2018' else transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ])
        }
        return data_transforms[split]

    # Load datasets
    @classmethod
    def load_data(cls, data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, shuffle=True):

        if phase == 'train_plain':
            txt_split = 'train'
        elif phase == 'train_val':
            txt_split = 'val'
            phase = 'train'
        else:
            txt_split = phase
        
        ## get file name
        txt = f'{data_root}/{dataset}_{txt_split}.txt'

        print('Loading data from %s' % (txt))

        if dataset == 'inatural2018':
            print('===> Loading iNatural2018 statistics')
            key = 'inatural2018'
        else:
            key = 'default'

        rgb_mean, rgb_std = cls.RGB_statistics[key]['mean'], cls.RGB_statistics[key]['std']

        if phase not in ['train', 'val']:
            transform = cls.get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = cls.get_data_transform(phase, rgb_mean, rgb_std, key)

        print('Use data transformation:', transform)

        set_ = LT_Dataset(data_root, txt, transform)
        print(len(set_))

        if sampler_dic and phase == 'train':
            print('Using sampler: ', sampler_dic['sampler'])
            print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            print('Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            print('No sampler.')
            print('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        


class ImGAGN_loader:

    @staticmethod
    def load_data(ratio_generated, data):

        dataset = Graph_Dataset(root="data", dataset = args.dataset, is_normalize = args.is_normalize, add_self_loop = args.add_sl)
        
        features, edges, labels, idx_test = data.features.cpu(), data.edge_index.cpu(), data.labels.cpu(), data.idx_test.cpu()
        idx_train = torch.LongTensor([x for x in range(len(labels)) if x not in idx_test])

        for i in range(max(labels)):
            chosen = idx_train[(labels == (max(labels) - i))[idx_train]]
            if chosen.shape[0] > 0:
                break
        majority = np.array([x for x in idx_train if labels[x] != max(labels) - i])
        minority = np.array([x for x in idx_train if labels[x] == max(labels) - i])

        num_minority = minority.shape[0]
        num_majority = majority.shape[0]
        print("Number of majority: ", num_majority)
        print("Number of minority: ", num_minority)

        generate_node = []
        generate_label=[]
        for i in range(labels.shape[0], labels.shape[0]+int(ratio_generated*num_majority)-num_minority):
            generate_node.append(i)
            generate_label.append(1)
        idx_train= np.hstack((idx_train, np.array(generate_node)))
        print(idx_train.shape)

        # minority_test = np.array([x for x in idx_test if labels[x] in smallest_class])
        minority_test = np.array([x for x in idx_test if labels[x] == max(labels) - i])
        minority_all = np.hstack((minority, minority_test))

        labels= np.hstack((labels, np.array(generate_label)))

        adj_real = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        # adj_real = utils.torch_sparse_tensor_to_sparse_mx(adj.to_sparse(), (labels.shape[0], labels.shape[0]))
        adj = adj_real + adj_real.T.multiply(adj_real.T > adj_real) - adj_real.multiply(adj_real.T > adj_real)

        features = normalize(features)
        features = torch.FloatTensor(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)
        generate_node = torch.LongTensor(np.array(generate_node))
        minority = torch.LongTensor(minority)
        majority = torch.LongTensor(majority)
        minority_all = torch.LongTensor(minority_all)

        return adj, adj_real, features, labels, idx_train, idx_test, generate_node, minority, majority, minority_all



        



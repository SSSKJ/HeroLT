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
    def load_data(cls, data_root, dataset, phase, batch_size, logger, sampler_dic=None, num_workers=4, shuffle=True):

        if phase == 'train_plain':
            txt_split = 'train'
        elif phase == 'train_val':
            txt_split = 'val'
            phase = 'train'
        else:
            txt_split = phase
        
        ## get file name
        txt = f'{data_root}/{dataset}_{txt_split}.txt'

        logger.log('Loading data from %s' % (txt))

        if dataset == 'inatural2018':
            logger.log('===> Loading iNatural2018 statistics')
            key = 'inatural2018'
        else:
            key = 'default'

        rgb_mean, rgb_std = cls.RGB_statistics[key]['mean'], cls.RGB_statistics[key]['std']

        if phase not in ['train', 'val']:
            transform = cls.get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = cls.get_data_transform(phase, rgb_mean, rgb_std, key)

        logger.log('Use data transformation:', transform)

        set_ = Decoupling_LT_Dataset(data_root, txt, transform)
        logger.log(len(set_))

        if sampler_dic and phase == 'train':
            logger.log('Using sampler: ', sampler_dic['sampler'])
            logger.log('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            logger.log('Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            logger.log('No sampler.')
            logger.log('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        

class BLAMS_loader:

    @classmethod
    def load_data(data_root, dataset, phase, batch_size, logger, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, cifar_imb_ratio=None, meta=False):
        if phase == 'train_plain':
            txt_split = 'train'
        elif phase == 'train_val':
            txt_split = 'val'
            phase = 'train'
        else:
            txt_split = phase
        
        ## get file name
        txt = f'{data_root}/{dataset}_{txt_split}.txt'

        logger.log('Loading data from %s' % (txt))


        if dataset == 'iNaturalist18':
            logger.log('===> Loading iNaturalist18 statistics')
            key = 'iNaturalist18'
        else:
            key = 'default'

        if dataset == 'CIFAR10_LT':
            logger.log('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
            set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
        elif dataset == 'CIFAR100_LT':
            logger.log('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
            set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
        else:
            rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
            if phase not in ['train', 'val']:
                transform = get_data_transform('test', rgb_mean, rgb_std, key)
            else:
                transform = get_data_transform(phase, rgb_mean, rgb_std, key)

            logger.log('Use data transformation:', transform)

            set_ = LT_Dataset(data_root, txt, dataset, transform, meta)


        logger.log(len(set_))

        if sampler_dic and phase == 'train' and sampler_dic.get('batch_sampler', False):
            logger.log('Using sampler: ', sampler_dic['sampler'])
            return DataLoader(dataset=set_,
                            batch_sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)

        elif sampler_dic and (phase == 'train' or meta):
            logger.log('Using sampler: ', sampler_dic['sampler'])
            logger.log('Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            logger.log('No sampler.')
            logger.log('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

        

class TDE_loader:

    @classmethod
    def load_data(cls, data_root, dataset, phase, batch_size, top_k_class=None, sampler_dic=None, num_workers=4, shuffle=True, cifar_imb_ratio=None):

        txt_split = phase
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
        template = './data/%s/%s'%(dataset, dataset)

        print('Loading data from %s' % (txt))

        if dataset == 'iNaturalist18':
            print('===> Loading iNaturalist18 statistics')
            key = 'iNaturalist18'
        else:
            key = 'default'

        if dataset == 'CIFAR10_LT':
            print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
            set_ = TDE_IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
        elif dataset == 'CIFAR100_LT':
            print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
            set_ = TDE_IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
        else:
            rgb_mean, rgb_std = Decoupling_loader.RGB_statistics[key]['mean'], Decoupling_loader.RGB_statistics[key]['std']
            if phase not in ['train', 'val']:
                transform = Decoupling_loader.get_data_transform('test', rgb_mean, rgb_std, key)
            else:
                transform = Decoupling_loader.get_data_transform(phase, rgb_mean, rgb_std, key)
            print('Use data transformation:', transform)

            set_ = TDE_LT_Dataset(data_root, txt, transform, template=template, top_k=top_k_class)
        

        print(len(set_))

        if sampler_dic and phase == 'train':
            print('=====> Using sampler: ', sampler_dic['sampler'])
            # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            print('=====> Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            print('=====> No sampler.')
            print('=====> Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)


class Graph_loader:

    @classmethod
    def load_data(self, config, dataset, model_name, root, device = -1):
        
        self.config = config
        self.config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load data - Cora, CiteSeer, cora_full
        self._dataset = Graph_Dataset(root = root, dataset = dataset, is_normalize = self.config['is_normalize'], add_self_loop = self.config['add_sl'])
        self.edge_index = self._dataset.edge_index

        adj = self._dataset.adj
        features = self._dataset.features
        labels = self._dataset.labels
        class_sample_num = 20
        im_class_num = self.config['im_class_num']

        # Natural Setting
        if self.config['im_ratio'] == 1:
            self.config['criterion'] = 'mean'
            labels, og_to_new = refine_label_order(labels)
            idx_train, idx_val, idx_test, class_num_mat = split_natural(labels, og_to_new)
            samples_per_label = torch.tensor(class_num_mat[:,0])

        # Set embeder
        if model_name == 'lte4g':
            idx_train_set_class, ht_dict_class = separate_ht(samples_per_label, labels, idx_train, method = self.config['sep_class'], manual = False)
            idx_train_set, degree_dict, degrees, above_head, below_tail  = separate_class_degree(adj, idx_train_set_class, below = self.config['sep_degree'])
            
            idx_val_set = separate_eval(idx_val, labels, ht_dict_class, degrees, above_head, below_tail)
            idx_test_set = separate_eval(idx_test, labels, ht_dict_class, degrees, above_head, below_tail)

            self.config['sep_point'] = len(ht_dict_class['H'])

            self.idx_train_set_class = idx_train_set_class
            self.degrees = degrees
            self.above_head = above_head
            self.below_tail = below_tail

            print('Above Head Degree:', above_head)
            print('Below Tail Degree:', below_tail)
            
            self.idx_train_set = {}
            self.idx_val_set = {}
            self.idx_test_set = {}
            for sep in ['HH', 'HT', 'TH', 'TT']:
                self.idx_train_set[sep] = idx_train_set[sep].to(self.config['device'])
                self.idx_val_set[sep] = idx_val_set[sep].to(self.config['device'])
                self.idx_test_set[sep] = idx_test_set[sep].to(self.config['device'])

        if model_name != 'tailgnn':
            adj = normalize_adj(adj) if self.config['adj_norm_1'] else normalize_sym(adj)

        self.adj = adj.to(self.config['device'])
        self.features = features.to(self.config['device'])
        self.labels = labels.to(self.config['device'])
        self.class_sample_num = class_sample_num
        self.im_class_num = im_class_num

        self.idx_train = idx_train.to(self.config['device'])
        self.idx_val = idx_val.to(self.config['device'])
        self.idx_test = idx_test.to(self.config['device'])

        self.samples_per_label = samples_per_label
        self.class_num_mat = class_num_mat
        print(class_num_mat)

        self.config['nfeat'] = features.shape[1]
        self.config['nclass'] = labels.max().item() + 1
        self.config['im_class_num'] = im_class_num

        return self.config, eval(f'self.__{model_name}_data_preprocessor')

    def __ImGAGN_data_preprocessor(self):
        
        ratio_generated = self.config['ratio_generated']

        features, edges, labels, idx_test = self.features.cpu(), self.edge_index.cpu(), self.labels.cpu(), self.idx_test.cpu()
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

        minority_test = np.array([x for x in idx_test if labels[x] == max(labels) - i])
        minority_all = np.hstack((minority, minority_test))

        labels= np.hstack((labels, np.array(generate_label)))

        adj_real = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

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
    
    def __TailGNN_data_preprocessor(self):

        return self.features, self.adj.to_dense(), self.labels, self.idx_train, self.idx_val, self.idx_test
    
    def __GraphSMOTE_data_preprocessor(self):

        return self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test
    
class CIFAR100_LT(object):
    def __init__(self, distributed, root='./data/cifar100', imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=40):

        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        
        train_dataset = MiSLAS_IMBALANCECIFAR100(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform)
        eval_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)
        
        self.cls_num_list = train_dataset.get_cls_num_list()

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        

class CIFAR10_LT(object):

    def __init__(self, distributed, root='./data/cifar10', imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=40):

        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        
        train_dataset = MiSLAS_IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform)
        eval_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)
        
        self.cls_num_list = train_dataset.get_cls_num_list()

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        

class Places_LT(object):
    def __init__(self, distributed, root="", batch_size=60, num_works=40):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        train_txt = "./datasets/data_txt/Places_LT_train.txt"
        eval_txt = "./datasets/data_txt/Places_LT_test.txt"

        
        train_dataset = MiSLAS_LT_Dataset(root, train_txt, transform=transform_train)
        eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        
        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)

class iNatural2018(object):
    def __init__(self, distributed, root="", batch_size=60, num_works=40):
        
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        train_txt = "./datasets/data_txt/iNaturalist18_train.txt"
        eval_txt = "./datasets/data_txt/iNaturalist18_val.txt"
        
        train_dataset = MiSLAS_LT_Dataset(root, train_txt, transform=transform_train)
        eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        
        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        
class ImageNet_LT(object):
    def __init__(self, distributed, root="", batch_size=60, num_works=40):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        
        train_txt = "./datasets/data_txt/ImageNet_LT_train.txt"
        eval_txt = "./datasets/data_txt/ImageNet_LT_test.txt"
        
        train_dataset = MiSLAS_LT_Dataset(root, train_txt, transform=transform_train)
        eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        
        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        
class OLTR_loader:

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    @classmethod
    def load_data(cls, data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):

        txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

        print('Loading data from %s' % (txt))

        if phase not in ['train', 'val']:
            transform = cls.data_transforms['test']
        else:
            transform = cls.data_transforms[phase]

        print('Use data transformation:', transform)

        set_ = Decoupling_LT_Dataset(data_root, txt, transform)

        if phase == 'test' and test_open:
            open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
            print('Testing with opensets from %s'%(open_txt))
            open_set_ = Decoupling_LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
            set_ = ConcatDataset([set_, open_set_])

        if sampler_dic and phase == 'train':
            print('Using sampler.')
            print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                            num_workers=num_workers)
        else:
            print('No sampler.')
            print('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        


        


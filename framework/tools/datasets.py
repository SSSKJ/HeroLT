from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import to_undirected, is_undirected, from_scipy_sparse_matrix


import os
import os.path as osp
from PIL import Image

from scipy.io import loadmat
from sklearn import preprocessing

from utils import *
from data import *
from .ClassAwareSampler import *

# Dataset definiation from Decoupling
class Decoupling_LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index
    
class TDE_LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        # select top k class
        if top_k:
            # only select top k in training, in case train/val/test not matching.
            if 'train' in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key = lambda x:x[1], reverse=True)
                # saving
                torch.save(dist, template + '_top_{}_mapping'.format(top_k))
            else:
                # loading
                dist = torch.load(template + '_top_{}_mapping'.format(top_k))
            selected_labels = {item[0]:i for i, item in enumerate(dist[:top_k])}
            # replace original path and labels
            self.new_img_path = []
            self.new_labels = []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    self.new_img_path.append(path)
                    self.new_labels.append(selected_labels[label])
            self.img_path = self.new_img_path
            self.labels = self.new_labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index
    
# Dataset definiation from ImGAGN
class Graph_Dataset(InMemoryDataset):
    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", dataset='cora', transform=None, pre_transform=None, is_normalize=True, add_self_loop=False):
        self.root, self.dataset, self.data_dir = download_data(root=root, dataset=dataset)
        create_dirs(self.dirs)
        super().__init__(root=self.data_dir, transform=transform, pre_transform=pre_transform)
        path = osp.join(self.data_dir, "processed", self.processed_file_names[0])
        self.data, self.slices = torch.load(path)

        if add_self_loop:
            print("Add self loop")
            self.data.edge_index = remove_self_loops(self.data.edge_index)[0]
            self.data.edge_index = add_self_loops(self.data.edge_index)[0]

        adj = sp.coo_matrix((np.ones(self.data.edge_index.shape[1]), (self.data.edge_index[0,:], self.data.edge_index[1,:])), shape=(self.data.y.shape[0], self.data.y.shape[0]), dtype=np.float32)
        self.adj = sparse_mx_to_torch_sparse_tensor(adj)

        if is_normalize: ## Normalize features
            features = sp.csr_matrix(self.data.x, dtype=np.float32)
            features = normalize(features)
            self.features = torch.FloatTensor(np.array(features.todense()))
        else: ## Not normalize features
            self.features = self.data.x

        self.labels = torch.LongTensor(self.data.y)
        self.edge_index = self.data.edge_index
        if 'planetoid' in self.root:
            self.train_mask = self.data.train_mask
            self.val_mask = self.data.val_mask
            self.test_mask = self.data.test_mask

        del self.data
        del self.slices

    def process_full_batch_data(self, data):
        """
        Augmented view data generation using the full-batch data.

        :param view1data:
        :return:
        """
        print("Processing full batch data")
        if 'planetoid' in self.root: # for LT dataset
            data = Data(edge_index=data.edge_index, edge_attr=data.edge_attr,
                    x=data.x, y=data.y, num_nodes=data.num_nodes, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        else:
            data = Data(edge_index=data.edge_index, edge_attr=data.edge_attr, x=data.x, y=data.y, num_nodes=data.num_nodes)
        return [data]

    def process(self):
        """
        Process a full batch data.
        :return:
        """
        processed_path = osp.join(self.processed_dir, self.processed_file_names[0])
        if not osp.exists(processed_path):
            path = osp.join(self.raw_dir, self.raw_file_names[0])
            data, _ = torch.load(path)
            edge_attr = data.edge_attr
            edge_attr = torch.ones(data.edge_index.shape[1]) if edge_attr is None else edge_attr
            data.edge_attr = edge_attr
            if not is_undirected(data.edge_index):
                data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_attr)
            data_list = self.process_full_batch_data(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return [f'byg.data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")

    @property
    def model_dir(self):
        return osp.join(self.data_dir, "model")

    @property
    def result_dir(self):
        return osp.join(self.data_dir, "result")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]

    def download(self):
        pass

class Email(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['edges.txt', 'graph.embeddings', 'labels.txt']
    @property
    def processed_file_names(self):
        return ['data.pt']
    def process(self):
        load_features = np.genfromtxt("{}/graph.embeddings".format(self.raw_dir), skip_header=1, dtype=np.float32)
        idx = load_features[load_features[:, 0].argsort()]  # sort regards to ascending index
        features = torch.from_numpy(idx[:, 1:])

        load_labels = np.genfromtxt("{}/labels.txt".format(self.raw_dir), dtype=np.int32)
        labels = torch.from_numpy(load_labels[:, 1]).to(torch.long)

        edges = np.genfromtxt("{}/edges.txt".format(self.raw_dir), dtype=np.int32)
        edge_index = torch.from_numpy(edges).T.to(torch.long)

        data = Data(x=features, edge_index=edge_index, y=labels)

        torch.save(self.collate([data]), self.processed_paths[0])

class BlogCatalog(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['blogcatalog.embeddings_64', 'blogcatalog.mat']
    @property
    def processed_file_names(self):
        return ['data.pt']
    def process(self):
        embed = np.loadtxt("{}/blogcatalog.embeddings_64".format(self.raw_dir))
        feature = np.zeros((embed.shape[0], embed.shape[1] - 1))
        feature[embed[:, 0].astype(int), :] = embed[:, 1:]
        features = normalize(feature)
        features = torch.FloatTensor(features)

        mat = loadmat("{}/blogcatalog.mat".format(self.raw_dir))
        adj = mat['network']
        edge_index, _ = from_scipy_sparse_matrix(adj)

        label = mat['group']
        labels = np.array(label.todense().argmax(axis=1)).squeeze()
        labels[labels > 16] = labels[labels > 16] - 1
        labels = torch.LongTensor(labels)

        data = Data(x=features, edge_index=edge_index, y=labels)

        torch.save(self.collate([data]), self.processed_paths[0])

class DataGPN(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self):
        return [osp.join(self.name, '_network'), osp.join(self.name, '_test.mat'), osp.join(self.name, '_train.mat')]
    @property
    def processed_file_names(self):
        return ['data.pt']
    def process(self):
        n1s = []
        n2s = []
        for line in open("{}/{}/raw/{}_network".format(self.root, self.name, self.name)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        data_train = loadmat("{}/{}/raw/{}_train.mat".format(self.root, self.name, self.name))
        data_test = loadmat("{}/{}/raw/{}_test.mat".format(self.root, self.name, self.name))

        labels = np.zeros((num_nodes, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        adj = normalize_sp_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        # adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        edge_index, _ = from_scipy_sparse_matrix(adj)

        data = Data(x=features, edge_index=edge_index, y=labels)
        torch.save(self.collate([data]), self.processed_paths[0])


"""
Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

class TDE_IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, phase, imbalance_ratio, root = '/gruntdata5/kaihua/datasets', imb_type='exp'):
        train = True if phase == "train" else False
        super(TDE_IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                             #transforms.Resize(224),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index
    
    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class TDE_IMBALANCECIFAR100(TDE_IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class MiSLAS_LT_Dataset(Dataset):
    num_classes = 8142

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 

class LT_Dataset_Eval(Dataset):
    num_classes = 365

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
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

class iNa2018(object):
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
        
class MiSLAS_IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(MiSLAS_IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
class MiSLAS_IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(MiSLAS_IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
        
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
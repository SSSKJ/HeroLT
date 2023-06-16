
from HeroLT.utils import create_dirs, sparse_mx_to_torch_sparse_tensor, normalize, normalize_sp_adj

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

import os.path as osp
from sklearn import preprocessing

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, is_undirected, from_scipy_sparse_matrix
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.datasets import CitationFull, AttributedGraphDataset

import os.path as osp

## Download PYG datasets
def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora_full':
        dataset = "cora"
        root = osp.join(root, "cora-full")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": CitationFull, "src": "pyg"}
    elif dataset == "email":
        dataset = "Email"
        root = osp.join(root, "email")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": Email, "src": "pyg"}
    elif dataset == "amz_cloth":
        dataset = "Amazon_clothing"
        root = osp.join(root, "amazon-clothing")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": DataGPN, "src": "pyg"}
    elif dataset == "amz_eletronics":
        dataset = "Amazon_eletronics"
        root = osp.join(root, "amazon-eletronics")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": DataGPN, "src": "pyg"}
    elif dataset == "wiki":
        dataset = "Wiki"
        root = osp.join(root, "wiki")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": "wiki", "class": AttributedGraphDataset, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics', 'actor'")
    return params


def download_pyg_data(config):
    """
    Downloads a dataset from the PyTorch Geometric library

    :param config: A dict containing info on the dataset to be downloaded
    :return: A tuple containing (root directory, dataset name, data directory)
    """
    leaf_dir = config["kwargs"]["root"].split("/")[-1].strip()
    data_dir = osp.join(config["kwargs"]["root"], "" if config["name"] == leaf_dir else config["name"])
    dst_path = osp.join(data_dir, "raw", "data.pt")
    if not osp.exists(dst_path):
        DatasetClass = config["class"]
        dataset = DatasetClass(**config["kwargs"])
        torch.save((dataset.data, dataset.slices), dst_path)
    return config["kwargs"]["root"], config["name"], data_dir

def download_data(root, dataset):
    """
    Download data from different repositories. Currently only PyTorch Geometric is supported

    :param root: The root directory of the dataset
    :param name: The name of the dataset
    :return:
    """
    config = decide_config(root=root, dataset=dataset)
    if config["src"] == "pyg":
        return download_pyg_data(config)


class Dataset(InMemoryDataset):
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

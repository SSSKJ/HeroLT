## dataset names and links
dataset_list = {'cora-full': '',
            'email': '',
            'wiki': '',
            'amazon-clothing': '',
            'amazon-electronics': '',
            'imageNet-lt': '',
            'places-lt': '',
            'inatural2018': '',
            'cifar10-lt': '',
            'cifar100-lt': '',
            'lvisv0.5': '',
            'eurlex-4k': '',
            'amazoncat-13k': '',
            'wiki10-31k': ''
            }


## todo: Download online data

from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, CitationFull, Actor, Reddit, AttributedGraphDataset
from tools.datasets import *

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
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": CitationFull, "src": "pyg"}
    elif dataset == "email":
        dataset = "Email"
        root = osp.join(root, "pyg", 'Email')
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": Email, "src": "pyg"}
    elif dataset == "amz_cloth":
        dataset = "Amazon_clothing"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": DataGPN, "src": "pyg"}
    elif dataset == "amz_eletronics":
        dataset = "Amazon_eletronics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": DataGPN, "src": "pyg"}
    elif dataset == "wiki":
        dataset = "Wiki"
        root = osp.join(root, "pyg")
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
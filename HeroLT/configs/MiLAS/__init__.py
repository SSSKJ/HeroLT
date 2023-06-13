from pathlib import Path
from yacs.config import CfgNode as CN

_C = CN()
_C.name = ''
_C.print_freq = 40
_C.workers = 16
_C.log_dir = 'logs'
_C.model_dir = 'ckps'


_C.dataset = 'cifar10'
_C.data_path = './data/cifar10'
_C.num_classes = 100
_C.imb_factor = 0.01
_C.backbone = 'resnet32_fe'
_C.resume = ''
_C.head_class_idx = [0, 1]
_C.med_class_idx = [0, 1]
_C.tail_class_idx = [0, 1]

_C.deterministic = True
_C.gpu = 0
_C.world_size = -1
_C.rank = -1
_C.dist_url = 'tcp://224.66.41.62:23456'
_C.dist_backend = 'nccl'
_C.multiprocessing_distributed = False
_C.distributed = False

_C.mode = None
_C.smooth_tail = None
_C.smooth_head = None
_C.shift_bn = False
_C.lr_factor = None
_C.lr = 0.1
_C.batch_size = 128
_C.weight_decay = 0.002
_C.num_epochs = 200
_C.momentum = 0.9
_C.cos = False
_C.mixup = True
_C.alpha = 1.0

def update_config(cfg, path):
    cfg.defrost()
    
    cfg.merge_from_file(path)

    return cfg

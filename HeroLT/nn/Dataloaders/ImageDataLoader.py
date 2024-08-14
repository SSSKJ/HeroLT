import torch
from torch.utils.data import DataLoader

from .public import get_data_transform, RGB_statistics

from ..Datasets.DecouplingDataset import LT_Dataset as Decoupling_LT_Dataset

from ..Datasets.BALMSDataset import LT_Dataset as BALMS_LT_Dataset
from ..Datasets.BALMSDataset import IMBALANCECIFAR10 as BALMS_IMBALANCECIFAR10
from ..Datasets.BALMSDataset import IMBALANCECIFAR100 as BALMS_IMBALANCECIFAR100

from ..Datasets.TDEDataset import LT_Dataset as TDE_LT_Dataset
from ..Datasets.TDEDataset import IMBALANCECIFAR10 as TDE_IMBALANCECIFAR10
from ..Datasets.TDEDataset import IMBALANCECIFAR100 as TDE_IMBALANCECIFAR100

from ..Datasets.OLTRDataset import ConcatDataset
from ..Datasets.OLTRDataset import LT_Dataset as OLTR_LT_Dataset

class ImageDataLoader:

    @classmethod
    def load_data(cls, data_root, dataset_name, model_name, phase, batch_size, logger, sampler_dic=None, num_workers=4, shuffle=True, **kwargs):

        dataset = dataset_name.lower()
        txt_split = phase

        if model_name != 'TDE':

            txt_split = 'train' if phase == 'train_plain' else phase

            if model_name in ['Decoupling', 'BAMLS'] and phase == 'train_val':
                    
                    txt_split = 'val'
                    phase = 'train'

        else:
            
            kwargs['template'] = f'{data_root}/%s/%s'%(dataset, dataset)
            
        
        ## get file name
        txt = f'{data_root}/{dataset}_{txt_split}.txt'

        logger.log.info('Loading data from %s' % (txt))

        if dataset == 'inatural2018':
            logger.log.info('===> Loading iNatural2018 statistics')
            key = 'inatural2018'
        else:
            key = 'default'

        set_ = cls.get_set(cls, key, phase, model_name, data_root, dataset, txt, logger, **kwargs)
        logger.log.info(len(set_))

        meta = kwargs.get('meta', False)
        batch_sampler = sampler_dic.get('batch_sampler', False)

        if sampler_dic:

            if phase == 'train':
                
                if batch_sampler:
                    logger.log.info('Using sampler: ', sampler_dic['sampler'])
                    return DataLoader(dataset=set_,
                                    batch_sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                                    num_workers=num_workers)
                
                else:
                    logger.log.info('Using sampler: ', sampler_dic['sampler'])
                    logger.log.info('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
                    logger.log.info('Sampler parameters: ', sampler_dic['params'])
                    return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                                    sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                                    num_workers=num_workers)
                
            elif meta:
                logger.log.info('Using sampler: ', sampler_dic['sampler'])
                logger.log.info('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
                logger.log.info('Sampler parameters: ', sampler_dic['params'])
                return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                                sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                                num_workers=num_workers)
        
        logger.log.info('No sampler.')
        logger.log.info('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)
    
    def get_set(self, key, phase, model_name, data_root, dataset, txt, logger, **kwargs):

        if model_name != 'Decoupling':

            if dataset == 'CIFAR10_LT'.lower():
                logger.log.info('====> CIFAR10 Imbalance Ratio: ', kwargs['cifar_imb_ratio'])
                return eval(f'{model_name}_IMBALANCECIFAR10')(phase, imbalance_ratio=kwargs['cifar_imb_ratio'], root=data_root)
            elif dataset == 'CIFAR100_LT'.lower():
                logger.log.info('====> CIFAR100 Imbalance Ratio: ', kwargs['cifar_imb_ratio'])
                return eval(f'{model_name}_IMBALANCECIFAR10')(phase, imbalance_ratio=kwargs['cifar_imb_ratio'], root=data_root)
            

        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

        if phase not in ['train', 'val']:
            transform = get_data_transform('test', model_name, rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, model_name, rgb_mean, rgb_std, key)

        logger.log.info('Use data transformation:', transform)

        set_ = eval(f'{model_name}_LT_Dataset')(root = data_root, txt = txt, transform = transform, **kwargs)
        set_.load()
        
        test_open = kwargs.get('test_open', False)

        if phase == 'test' and test_open:
            open_txt = f'{data_root}/%s/%s_open.txt'%(dataset, dataset)
            logger.info('Testing with opensets from %s'%(open_txt))
            open_set_ = (f'{model_name}_LT_Dataset')(f'{data_root}/%s/%s_open'%(dataset, dataset), open_txt, transform)
            set_ = ConcatDataset([set_, open_set_])


        return set_
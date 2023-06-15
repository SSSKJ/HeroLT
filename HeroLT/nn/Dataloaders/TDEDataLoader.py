from .public import get_data_transform, RGB_statistics
from ..Datasets.TDEDataset import LT_Dataset, IMBALANCECIFAR10, IMBALANCECIFAR100

from torch.utils.data import DataLoader

class TDEDataLoader:

    @classmethod
    def load_data(cls, data_root, dataset, phase, batch_size, logger, top_k_class=None, sampler_dic=None, num_workers=4, shuffle=True, cifar_imb_ratio=None):

        txt_split = phase
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
        template = './data/%s/%s'%(dataset, dataset)

        logger.log.info('Loading data from %s' % (txt))

        if dataset == 'iNaturalist18':
            logger.log.info('===> Loading iNaturalist18 statistics')
            key = 'iNaturalist18'
        else:
            key = 'default'

        if dataset == 'CIFAR10_LT':
            logger.log.info('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
            set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
        elif dataset == 'CIFAR100_LT':
            logger.log.info('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
            set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
        else:
            rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
            if phase not in ['train', 'val']:
                transform = get_data_transform('test', rgb_mean, rgb_std, key)
            else:
                transform = get_data_transform(phase, rgb_mean, rgb_std, key)
            logger.log.info('Use data transformation:', transform)

            set_ = LT_Dataset(data_root, txt, transform, template=template, top_k=top_k_class)
        

        logger.log.info(len(set_))

        if sampler_dic and phase == 'train':
            logger.log.info('=====> Using sampler: ', sampler_dic['sampler'])
            # logger.log.info('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            logger.log.info('=====> Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            logger.log.info('=====> No sampler.')
            logger.log.info('=====> Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

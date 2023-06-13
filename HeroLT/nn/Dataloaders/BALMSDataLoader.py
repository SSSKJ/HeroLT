from torch.utils.data import DataLoader

from HeroLT.nn.Datasets.BALMSDataset import LT_Dataset, IMBALANCECIFAR10, IMBALANCECIFAR100
from HeroLT.nn.Dataloaders import get_data_transform, RGB_statistics


class BALMSDataLoader:

    def load_data(data_root, dataset, phase, batch_size, logger, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, cifar_imb_ratio=None, meta=False):

        if phase == 'train_plain':
            txt_split = 'train'
        elif phase == 'train_val':
            txt_split = 'val'
            phase = 'train'
        else:
            txt_split = phase

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
            # logger.log('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            logger.log('Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            logger.log('No sampler.')
            logger.log('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

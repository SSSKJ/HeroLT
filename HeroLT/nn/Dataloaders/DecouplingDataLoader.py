from torch.utils.data import DataLoader
from ..Datasets.DecouplingDataset import LT_Dataset

from .public import get_data_transform, RGB_statistics

# Data Loader definiation from Decoupling
class DecouplingDataLoader:

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

        logger.log.info('Loading data from %s' % (txt))

        if dataset == 'inatural2018':
            logger.log.info('===> Loading iNatural2018 statistics')
            key = 'inatural2018'
        else:
            key = 'default'

        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key)

        logger.log.info('Use data transformation:', transform)

        set_ = LT_Dataset(data_root, txt, transform)
        logger.log.info(len(set_))

        if sampler_dic and phase == 'train':
            logger.log.info('Using sampler: ', sampler_dic['sampler'])
            logger.log.info('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            logger.log.info('Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            logger.log.info('No sampler.')
            logger.log.info('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
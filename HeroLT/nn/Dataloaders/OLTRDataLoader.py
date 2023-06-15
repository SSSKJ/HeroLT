from torchvision import transforms
from torch.utils.data import DataLoader

from ..Datasets.DecouplingDataset import LT_Dataset
from ..Datasets.OLTRDataset import ConcatDataset

import numpy as np

class OLTRDataLoader:

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
    def load_data(cls, data_root, dataset, phase, batch_size, logger, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):

        txt_split = phase if phase != 'train_plain' else 'train'
        txt = f'{data_root}/{dataset}_{txt_split}.txt'

        logger.info('Loading data from %s' % (txt))

        if phase not in ['train', 'val']:
            transform = cls.data_transforms['test']
        else:
            transform = cls.data_transforms[phase]

        logger.info('Use data transformation:', transform)

        set_ = LT_Dataset(data_root, txt, transform)

        if phase == 'test' and test_open:
            open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
            logger.info('Testing with opensets from %s'%(open_txt))
            open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
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
        

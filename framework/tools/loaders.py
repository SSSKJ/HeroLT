from torchvision import transforms
from torch.utils.data import DataLoader

from datasets import LT_Dataset

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
    def load_data(cls, data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, shuffle=True):

        if phase == 'train_plain':
            txt_split = 'train'
        elif phase == 'train_val':
            txt_split = 'val'
            phase = 'train'
        else:
            txt_split = phase
        
        ## get file name
        txt = f'{data_root}/{dataset}_{txt_split}.txt'

        print('Loading data from %s' % (txt))

        if dataset == 'inatural2018':
            print('===> Loading iNatural2018 statistics')
            key = 'inatural2018'
        else:
            key = 'default'

        rgb_mean, rgb_std = cls.RGB_statistics[key]['mean'], cls.RGB_statistics[key]['std']

        if phase not in ['train', 'val']:
            transform = cls.get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = cls.get_data_transform(phase, rgb_mean, rgb_std, key)

        print('Use data transformation:', transform)

        set_ = LT_Dataset(data_root, txt, transform)
        print(len(set_))

        if sampler_dic and phase == 'train':
            print('Using sampler: ', sampler_dic['sampler'])
            print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
            print('Sampler parameters: ', sampler_dic['params'])
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                            num_workers=num_workers)
        else:
            print('No sampler.')
            print('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        

        
        



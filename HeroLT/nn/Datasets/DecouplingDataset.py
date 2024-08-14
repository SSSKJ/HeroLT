from torch.utils.data import Dataset

import os
from PIL import Image


# Dataset definiation from Decoupling
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, **kwargs):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.root = root
        self.txt = txt


    def load(self):
        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(self.root, line.split()[0]))
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
import yaml
import os

import datasets
from utils import special_mkdir

class BaseModel:

    def __init__(
            self, 
            model_name: str, 
            dataset_name: str, 
            device: str,
            base_dir: str = '../../'
            ) -> None:
        
        self.model_name = model_name
        if dataset_name.lower() not in datasets.datasets:
            raise RuntimeError(f'Dataset {dataset_name} not match')
        self.dataset_name = dataset_name
        self.base_dir = base_dir

        ## todo: Parallel Training
        self.device = device
        self.output_path = f'{self.base_dir}/outputs/{self.dataset_name}/'
        self.__training_data = None
        self.__testing_data = None
        self.__model = None

        special_mkdir(f'{base_dir}/datasaets/', dataset_name)
        special_mkdir(f'{base_dir}/outputs/', dataset_name)

    def __load_config(self):

        config_path = f'{self.base_dir}/config/{self.model_name}/config.yaml'
        with open(config_path) as f:
            self.config = yaml.load(f)

    def load_data(self):
        
        files = os.listdir(f'{self.base_dir}/datasets')

        if self.dataset_name.lower() not in files:
            ## download and unzip, move files into the right place
            print(f'Download dataset {self.dataset_name}')
        

    def load_pretrained_model(self):

        files = os.listdir(f'{self.base_dir}/pretrained_models')

        if self.dataset_name.lower() not in files:
            ## download and unzip, move files into the right place
            print(f'Download pretrained model for {self.model_name} on {self.dataset_name}')

    def save_model(self):
        pass

    def __init_model(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass
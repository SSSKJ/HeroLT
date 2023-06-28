import yaml
import os

from ...data import dataset_list

class BaseModel:

    def __init__(
            self, 
            model_name: str, 
            dataset_name: str, 
            base_dir: str = '../../'
            ) -> None:
        
        self.model_name = model_name
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in dataset_list:
            raise RuntimeError(f'Dataset {self.dataset_name} not match')
        
        self.base_dir = base_dir

        self.output_path = f'{self.base_dir}/outputs/{self.model_name}/{self.dataset_name}/'
        os.makedirs(self.output_path, exist_ok = True)

    def load_config(self):

        config_path = ''
        if os.path.exists(f'{self.base_dir}/configs/{self.model_name}/config.yaml'):
            config_path = f'{self.base_dir}/configs/{self.model_name}/config.yaml'
        elif os.path.exists(f'{self.base_dir}/configs/{self.model_name}/{self.dataset_name}/config.yaml'):
            config_path = f'{self.base_dir}/configs/{self.model_name}/{self.dataset_name}/config.yaml'
        else:
            raise RuntimeError(f'Can not find any configuration files')
        
        with open(config_path) as f:
            self.config = yaml.load(f, Loader = yaml.FullLoader)

    def load_data(self):
        
        files = list(os.listdir(f'{self.base_dir}/data'))
        files += list(os.listdir(f'{self.base_dir}/data/GraphData/'))
        if os.path.exists(f'{self.base_dir}/data/{self.model_name}/'):
            files += list(os.listdir(f'{self.base_dir}/data/{self.model_name}/'))

        if self.dataset_name not in files:

            ## download and unzip, move files into the right place
            print(f'Download dataset {self.dataset_name}')
        

    def load_pretrained_model(self):

        files = os.listdir(f'{self.base_dir}/pretrained_models')

        if self.dataset_name not in files:
            ## download and unzip, move files into the right place
            print(f'Download pretrained model for {self.model_name} on {self.dataset_name}')

    def save_model(self):
        pass

    def __init_model(self):
        pass

    def __init_optimizer_and_scheduler(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass
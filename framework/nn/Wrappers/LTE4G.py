from BaseModel import BaseModel

class LTE4G(BaseModel):


    def __init__(
            self,
            dataset: str,
            device: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'LTE4G',
            dataset = dataset,
            device = device,
            base_dir = base_dir)
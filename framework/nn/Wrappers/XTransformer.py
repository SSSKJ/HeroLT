from BaseModel import BaseModel

class XTransformer(BaseModel):


    def __init__(
            self,
            dataset: str,
            device: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'XTransformer',
            dataset = dataset,
            device = device,
            base_dir = base_dir)
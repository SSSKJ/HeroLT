from BaseModel import BaseModel

class XRTransformer(BaseModel):


    def __init__(
            self,
            dataset: str,
            device: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'XRTransformer',
            dataset = dataset,
            device = device,
            base_dir = base_dir)
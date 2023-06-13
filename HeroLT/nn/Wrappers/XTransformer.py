from BaseModel import BaseModel

class XTransformer(BaseModel):


    def __init__(
            self,
            dataset: str,
            base_dir: str = '../../',
            ) -> None:
        
        super().__init__(
            model_name = 'XTransformer',
            dataset = dataset,
            base_dir = base_dir)
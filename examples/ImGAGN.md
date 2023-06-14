1. 运行以下语句进行复现

   ```python
   from framework.nn.Wrappers import ImGAGN
   model = ImGAGN('imagenet_lt', './framework/')
   model.train()
   ```
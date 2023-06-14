1. 配置对应的配置文件到HeroLT/configs/ImGAGN/imagenet_lt/config.yaml下

1. 运行以下语句进行复现

   ```python
   from framework.nn.Wrappers import ImGAGN
   model = ImGAGN('imagenet_lt', './framework/')
   model.train()
   ```
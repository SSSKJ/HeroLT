1. 配置对应的配置文件到HeroLT/configs/TailGNN/config.yaml下

2. 运行以下语句进行复现, 其中数据集名称包含cora-full, email, wiki, amazon-clothing, amazon-electronics

   ```python
   from framework.nn.Wrappers import TailGNN
   model = ImGAGN('imagenet_lt', './framework/')
   model.train()
   ```
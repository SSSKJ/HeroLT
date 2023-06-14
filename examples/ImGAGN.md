1. 配置对应的配置文件到HeroLT/configs/ImGAGN/config.yaml下

2. 运行以下语句进行复现, 其中数据集名称包含cora-full, email, wiki, amazon-clothing, amazon-electronics

   ```python
   from HeroLT.nn.Wrappers import ImGAGN
   model = ImGAGN('email', './HeroLT/')
   model.train()
   ```
   
3. 等待实现读取以及保存预训练模型
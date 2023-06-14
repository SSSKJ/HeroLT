1. 配置对应的配置文件到HeroLT/configs/LTE4G/config.yaml下

2. 运行以下语句进行复现, 其中数据集名称包含cora-full, email, wiki, amazon-clothing, amazon-electronics

   ```python
   from HeroLT.nn.Wrappers import LTE4G
   model = LTE4G('email', './HeroLT/')
   model.train()
   ```

3. 等待实现读取以及保存预训练模型
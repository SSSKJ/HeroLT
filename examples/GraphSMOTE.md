1. 配置对应的配置文件到HeroLT/configs/GraphSMOTE/config.yaml下

1. 将数据集放置到HeroLT/data/下，如HeroLT/data/email

3. 运行以下语句进行复现, 其中数据集名称包含cora-full, email, wiki, amazon-clothing, amazon-electronics

   ```python
   import sys
   sys.path.append('../')  
   from HeroLT.nn.Wrappers import GraphSMOTE
   model = GraphSMOTE('email', '../HeroLT/')
   model.train()
   ```

3. 等待实现读取以及保存预训练模型
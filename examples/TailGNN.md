1. 配置对应的配置文件到HeroLT/configs/TailGNN/config.yaml下

2. 将数据集放置到HeroLT/data/GraphData/下，如HeroLT/data/GraphData/email

3. 运行以下语句进行复现, 其中数据集名称包含cora-full, email, wiki, amazon-clothing, amazon-electronics

   ```python
   import sys
   sys.path.append('../')  
   from HeroLT.nn.Wrappers import TailGNN
   model = TailGNN('email', '../HeroLT/')
   model.train()
   ```
   
4. 等待实现读取以及保存预训练模型
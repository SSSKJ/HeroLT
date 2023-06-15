1. 怎么获取数据

2. 将数据文件解压，然后将所有文件（如文件在文件夹下请全部取出），放置到 /HeroLT/data/TDE/imagenet_lt/下，其中imagenet_lt为数据集名称重命名，命名列表包含imagenet-lt, places-lt, inatural2018, cifar10-lt, cifar100-lt以及lvisv1.0

3. 怎么获取模型

4. 将数据文件解压，然后将所有文件（如文件在文件夹下请全部取出），放置到 /HeroLT/outputs/TDE/imagenet_lt/下，其中imagenet_lt为数据集名称重命名，命名列表包含imagenet-lt, places-lt, inatural2018, cifar10-lt, cifar100-lt以及lvisv1.0

4. 配置对应的配置文件到HeroLT/configs/TDE/imagenet_lt/config.yaml下

5. 运行以下语句进行复现

   ```python
   from HeroLT.nn.Wrappers import TDE
   model = TDE('imagenet_lt', './HeroLT/', True)
   model.load_pretrained_model()
   model.train() ## train if you need to
   model.load_data()
   model.eval(phase=model.test_phase)
   model.output_logits()
   ```
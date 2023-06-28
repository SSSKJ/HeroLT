1. Configure the corresponding configuration under `HeroLT/configs/XRLinear/config.yaml`

2. Download dataset with commands, put it under `HeroLT/data/NLPData/` and change the name to lowercase. e.g., `HeroLT/data/NLPData/xmc/eurlex-4k`

   ```shell
   # eurlex-4k, wiki10-31k, amazoncat-13k
   DATASET="eurlex-4k"
   wget https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz
   tar -zxvf ./${DATASET}.tar.gz
   ```

   We now support the following dataset: eurlex-4k, wiki10-31k, and amazoncat-13k.

3. Run the following code to reproduce.

   ```python
   import sys
   sys.path.append('../')  
   from HeroLT.nn.Wrappers import XRLinear
   model = XRLinear('eurlex-4k', '../HeroLT/')
   model.train()
   ```

4. The models, output results and log file will be saved to `HeroLT/outputs/XRLinear/eurlex-4k/` if the dataset you are running is eurlex-4k.


1. Configure the corresponding configuration under `HeroLT/configs/XRTransformer/config.yaml`

2. Download dataset with commands and put it under `HeroLT/data/NLPData/`. e.g., `HeroLT/data/NLPData/xmc/eurlex-4k`

   ```shell
   # eurlex-4k, wiki10-31k, amazoncat-13k
   DATASET="eurlex-4k"
   wget https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz
   tar -zxvf ./${DATASET}.tar.gz
   ```

   We now support the following dataset: eurlex-4k, wiki10-31k, and amazoncat-13k.

4. Run the following code to reproduce.

   ```python
   import sys
   sys.path.append('../')  
   from HeroLT.nn.Wrappers import XRTransformer
   model = XRTransformer('eurlex-4k', '../HeroLT/')
   model.train()
   ```

5. The models, output results and log file will be saved to `HeroLT/outputs/XRTransformer/eurlex-4k/` if the dataset you are running is eurlex-4k.


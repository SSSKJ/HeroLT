1. Configure the corresponding configuration under `HeroLT/configs/XTransformer/config.yaml`

2. Download dataset with the script named `download-data.sh` under `HeroLT/data/NLPData/`  with commands, put it under `HeroLT/data/NLPData` and change the name to lowercase . e.g., `HeroLT/data/NLPData/eurlex-4k`

   ```shell
   # eurlex-4k, wiki10-31k, amazoncat-13k
   bash download-data.sh eurlex-4k
   ```

   We now support the following dataset: eurlex-4k, wiki10-31k, and amazoncat-13k.

3. Download pretrained models with the script named `download-models.sh` under `HeroLT/output/XTransformer/`  with commands, put it under `HeroLT/output/XTransformer/` and change the name to lowercase. e.g., `HeroLT/output/XTransformer/eurlex-4k`

   ```shell
   # eurlex-4k, wiki10-31k, amazoncat-13k
   bash download-models.sh eurlex-4k
   ```

4. Run the following code to reproduce.

   ```python
   import sys
   sys.path.append('../')  
   from HeroLT.nn.Wrappers import XTransformer
   model = XTransformer('eurlex-4k', '../HeroLT/')
   model.train()
   ```

5. The models, output results and log file will be saved to `HeroLT/outputs/XTransformer/eurlex-4k/` if the dataset you are running is eurlex-4k.


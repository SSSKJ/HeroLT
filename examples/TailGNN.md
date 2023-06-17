1. Configure the corresponding configuration under `HeroLT/configs/TailGNN/config.yaml`
2. Download dataset and put it under `HeroLT/data/`GraphData/, e.g., `HeroLT/data/GraphData/wiki`.
We now support the following dataset: email, wiki, [cora-full](https://github.com/Leo-Q-316/ImGAGN/tree/main/dataset/cora), [amazon-clothing](https://github.com/kaize0409/GPN_Graph-Few-shot/tree/master/few_shot_data), [amazon-eletronics](https://github.com/kaize0409/GPN_Graph-Few-shot/tree/master/few_shot_data).
3. Run the following code to reproduce.
   ```python
   import sys
   sys.path.append('../')  
   from HeroLT.nn.Wrappers import TailGNN
   model = TailGNN('email', '../HeroLT/')
   model.train()
   ```
4. Print the results and save the pre-trained model if applicable.
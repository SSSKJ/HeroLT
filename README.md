<!-- <h1 align="center">
    HeroLT
</h1> -->

**HeroLT** is a comprehensive long-tailed learning benchmark that examine long-tailed learning concerning three pivotal angles:
 1. **(A1) the characterization of data long-tailedness:** long-tailed data exhibits a highly skewed data distribution and an extensive number of categories; 
 2. **(A2) the data complexity of various domains:** a wide range of complex domains may naturally encounter long-tailed distribution, e.g., tabular data, sequential data, grid data, and relational data; and
 3. **(A3) the heterogeneity of emerging tasks:** it highlights the need to consider the applicability and limitations of existing methods on heterogeneous tasks.

<div  align="center">
 <img src="https://github.com/SSSKJ/HeroLT/blob/main/figs/angle.png" width = "700" height = "366" />
</div>


We provide a fair and accessible performance evaluation of 13 state-of-the-art methods on multiple benchmark datasets across multiple tasks using accuracy-based and ranking-based evaluation metrics.

| [Code Structure](#code-structure) | [Quick Start](#quick-start) | [Algorithms](#algorithms) | [Datasets](#datasets) | [License](#license) | [Publications](#publications) | 

----

## Code Structure
```
HeroLT
├── framework
│   ├── data                     # Datasets in HeroLT 
│   |   ├── ... ..
│   ├── nn           
│   |   ├── layers              # Implementation for layers (i.e., graph convolution)
│   |   ├── Models              # 
│   |   ├── Modules             # 
│   |   ├── Wrappers            # 
│   |   ├── loss                # Loss functions in long-tailed learning 
│   |   ├── ... ..
│   ├── tools                   #         
│   |   ├── ... ..
│   ├── utils                   # utility functions and classes      
│   |   ├── ... ..         
│   ├── ... ...          
├── examples                    #
└── README.md
```

## Quick Start

We provide the following example for users to quickly implementing HerolT.

### Step 1. Dependency

First of all, users need to clone the source code and install the required packages:

- scikit-learn==0.20.3
- pyod==1.0.0
- Keras==2.8.0
- tensorflow==2.8.0
- torch==1.9.0
- rtdl==0.0.13
- delu
- lightgbm
- xgboost
- catboost 
- copulas

<!-- ```bash
git clone https://github.com/SSSKJ/HeroLT/
cd HeroLT
``` -->

### Step 2. Prepare datasets

To run an LT task, users should prepare a dataset. The DataZoo provided in HeroLT can help to automatically download and preprocess widely-used public datasets for various LT applications, including CV, NLP, graph learning, etc. Users can directly specify `data = DATASET_NAME`in the configuration. For example, 

```bash
data = 'Cora_Full'
```

### Step 3. Prepare models

Then, users should specify the model architecture that will be trained. HeroLT provides a ModelZoo that contains the implementation of widely adopted model architectures for various LT tasks. Users can set up `model = MODEL_NAME` to apply a specific model architecture in LT tasks. For example,

```bash
model = 'GraphSMOTE'
```

### Step 4. Start running

Here we demonstrate how to run a standard LT task with HeroLT, with setting `data = 'Cora_Full'`and `model = 'GraphSMOTE'` to run GraphSMOTE for an node classification task on Cora_Full dataset. Users can customize training configurations, such as `lr`, in the configuration, and run a standard LT task as: 

```bash
# Run with custom configurations
python main.py ---data 'Cora_Full' --model 'GraphSMOTE' --lr ${lr}
```

Then you can observe some monitored metrics during the training process as:

```
============== seed:123 ==============
[seed 123][graphsmote][Epoch 0][Val] ACC: 2.3, bACC: 1.9, Precision: 1.7, Recall: 1.9, mAP: 2.5|| [Test] ACC: 2.5, bACC: 2.0, Precision: 0.9, Recall: 2.0, mAP: 2.1
  [*Best Test Result*][Epoch 0] ACC: 2.5,  bACC: 2.0, Precision: 0.9, Recall: 2.0, mAP: 2.1
[seed 123][graphsmote][Epoch 100][Val] ACC: 50.4, bACC: 30.1, Precision: 34.0, Recall: 30.1, mAP: 26.5|| [Test] ACC: 50.9, bACC: 30.7, Precision: 40.1, Recall: 30.7, mAP: 24.4
  [*Best Test Result*][Epoch 100] ACC: 50.9,  bACC: 30.7, Precision: 40.1, Recall: 30.7, mAP: 24.4
[seed 123][graphsmote][Epoch 200][Val] ACC: 60.2, bACC: 45.9, Precision: 54.3, Recall: 45.9, mAP: 38.3|| [Test] ACC: 60.1, bACC: 45.4, Precision: 55.3, Recall: 45.4, mAP: 36.6
  [*Best Test Result*][Epoch 200] ACC: 60.1,  bACC: 45.4, Precision: 55.3, Recall: 45.4, mAP: 36.6
[seed 123][graphsmote][Epoch 300][Val] ACC: 61.0, bACC: 48.6, Precision: 53.5, Recall: 48.6, mAP: 42.3|| [Test] ACC: 60.5, bACC: 48.5, Precision: 56.3, Recall: 48.5, mAP: 40.4
  [*Best Test Result*][Epoch 282] ACC: 60.6,  bACC: 48.3, Precision: 56.1, Recall: 48.3, mAP: 39.9
... ...
```

## Algorithms

HeroLT includes [13 algorithms](https://github.com/SSSKJ/HeroLT/framework/nn/Wrappers), as shown in the following Table.

| Algorithm      | Venue     | Long-tailedness                          | Task                                        |
|----------------|-----------|------------------------------------------|---------------------------------------------|
| X-Transformer  | 20KDD     | Data imbalance, extreme # of categories  | Multi-label text classification             |
| XR-Transformer | 21NeurIPS | Data imbalance, extreme \# of categories | Multi-label text classification             |
| XR-Linear      | 22KDD     | Data imbalance, extreme \# of categories | Multi-label text classification             |
| BBN            | 20CVPR    | Data imbalance                           | Image classification                        |
| BALMS          | 20NeurIPS | Data imbalance                           | Image classification, Instance segmentation |
| OLTR           | 19CVPR    | Data imbalance, extreme \# of categories | Image classification                        |
| TDE            | 20NeurIPS | Data imbalance, extreme \# of categories | Image classification                        |
| MiSLAS         | 21CVPR    | Data imbalance                           | Image classification                        |
| Decoupling     | 20ICLR    | Data imbalance                           | Image classification                        |
| GraphSMOTE     | 21WSDM    | Data imbalance                           | Node classification                         |
| ImGAGN         | 21KDD     | Data imbalance                           | Node classification                         |
| TailGNN        | 21KDD     | Data imbalance, extreme \# of categories | Node classification                         |
| LTE4G          | 22CIKM    | Data imbalance, extreme \# of categories | Node classification                         |     


## Datasets

HeroLT includes [14 datasets](https://github.com/SSSKJ/HeroLT/framework/data), as shown in the following Table.
        
|                    | Data Statistics |                 |           |            | Long-Tailedness |       |        |
|--------------------|-----------------|-----------------|-----------|------------|-----------------|:-----:|--------|
| Dataset            | Data            | # of Categories | Size      | # of Edges | IF              | Gini  | Pareto |
| EURLEX-4K          | Sequential      | 3,956           | 15,499    | -          | 1,024           | 0.342 | 3.968  |
| AMAZONCat-13K      | Sequential      | 13,330          | 1,186,239 | -          | 355,211         | 0.327 | 20.000 |
| Wiki10-31K         | Sequential      | 30,938          | 14,146    | -          | 11,411          | 0.312 | 4.115  |
| ImageNet-LT        | Grid            | 1,000           | 115,846   | -          | 256             | 0.517 | 1.339  |
| Places-LT          | Grid            | 365             | 62,500    | -          | 996             | 0.610 | 2.387  |
| iNatural 2018      | Grid            | 8,142           | 437,513   | -          | 500             | 0.647 | 1.658  |
| CIFAR 10-LT (100)  | Grid            | 10              | 12,406    | -          | 100             | 0.617 | 1.751  |
| CIFAR 10-LT (50)   | Grid            | 10              | 13,996    | -          | 50              | 0.593 | 1.751  |
| CIFAR 10-LT (10)   | Grid            | 10              | 20,431    | -          | 10              | 0.520 | 0.833  |
| CIFAR 100-LT (100) | Grid            | 100             | 10,847    | -          | 100             | 0.498 | 1.972  |
| CIFAR 100-LT (50)  | Grid            | 100             | 12,608    | -          | 50              | 0.488 | 1.590  |
| CIFAR 100-LT (10)  | Grid            | 100             | 19,573    | -          | 10              | 0.447 | 0.836  |
| LVIS v0.5          | Grid            | 1,231           | 693,958   | -          | 26,148          | 0.381 | 6.250  |
| Cora-Full          | Relational      | 70              | 19,793    | 146,635    | 62              | 0.321 | 0.919  |
| Wiki               | Relational      | 17              | 2,405     | 25,597     | 45              | 0.414 | 1.000  |
| Email              | Relational      | 42              | 1,005     | 25,934     | 109             | 0.413 | 1.263  |
| Amazon-Clothing    | Relational      | 77              | 24,919    | 208,279    | 10              | 0.343 | 0.814  |
| Amazon-Electronics | Relational      | 167             | 42,318    | 129,430    | 9               | 0.329 | 0.600  |

## License

<!-- HeroLT is released under Apache License 2.0. -->

## Publications
<!-- If you find HeroLT useful for your research or development, please cite the following <a href="https://arxiv.org/" target="_blank">paper</a>:
```
@article{heroLT,
  title = {HeroLT: Benchmarking Heterogeneous Long-Tailed Learning},
  author = {Wang, Haohui and Guan, Weijie and Chen, Jianpeng and Wang, Zi and Zhou, Dawei},
}
``` -->
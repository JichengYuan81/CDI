# Counterfactual Distribution Intervention for Few-Shot Class-Incremental Learning

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) focuses on enabling models to learn new classes from limited samples while preventing catastrophic forgetting of previously acquired knowledge. Existing methods often fail to differentiate between intrinsic class features and irrelevant background correlations, making models sensitive to shifts in data distribution during incremental learning, which limits their ability to generalize. To tackle this issue, we propose the Counterfactual Distribution Intervention (CDI) framework, which focuses on learning robust feature representations by mitigating non-discriminative interference through causal intervention. The framework consists of three key modules: In the base-class phase, Causal Feature Decoupling (CFD) explicitly separates features into causal and stylistic components. Building on this, Distribution-Shifted Counterfactual Intervention (DSCI) applies structured intervention to stylistic features, forcing the model to learn distribution-invariant causal representations. During the incremental stage, Random Sampling Counterfactual Intervention (RSCI) repurposes base-class style features to generate diverse counterfactual samples, efficiently utilizing limited data while maintaining feature stability. Extensive experiments on three benchmark datasets—CUB-200, CIFAR-100, and mini-ImageNet—demonstrate that the CDI framework significantly outperforms existing state-of-the-art methods, confirming its effectiveness in improving model generalization and preventing forgetting.


## Pipeline
<p align="center">
  <img width=900 src=".github/pipeline.png">
</p>


## Environment
The system I used and tested in

- Ubuntu 20.04.5 LTS
- NVIDIA GeForce RTX 3090
- Pytorch 1.12.1

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Datasets
For miniImagenet and CUB200, Please refer to [CEC](https://github.com/icoz69/CEC-CVPR2021)  to prepare the dataset.
For CIFAR100, the dataset will be download automatically.

## Training scripts

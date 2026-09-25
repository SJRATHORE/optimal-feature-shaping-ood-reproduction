# Optimal Feature Shaping for OOD Detection — Reproduction


Reproduction and adaptation of:

*Towards Optimal Feature-Shaping Methods for Out-of-Distribution Detection*  
Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould  
ICLR 2024

# Project Overview

This project reproduces the optimal feature-shaping method proposed by Zhao et al. for out-of-distribution (OOD) detection.

Our reproduction uses:

- *In-distribution dataset:* CIFAR-10
- *Out-of-distribution dataset:* SVHN
- *Backbone:* ResNet-18
- *Baseline:* Maximum Softmax Probability (MSP)
- *Proposed method:* Optimal feature shaping using ID training features
- *Evaluation:* AUROC and FPR95

The original paper evaluates several architectures. 
In this project, the method is adapted to a ResNet-18 backbone to provide a computationally feasible reproduction while preserving the core feature-shaping methodology.

# Authors

- Sudheer Rathore: 31683
- Anser Abbas: 31675

## Status

Milestone 2 — In progress

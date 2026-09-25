\# Provenance



This document records the origin of the major components used in this reproduction project.



\## Paper


**Towards Optimal Feature-Shaping Methods for Out-of-Distribution Detection**


Qinyu Zhao et al., ICLR 2024



Official repository:



https://github.com/Qinyu-Allen-Zhao/OptFSOOD



The mathematical formulation of optimal feature shaping, Interval-Specific Feature Impact (ISFI), percentile-based feature intervals, and the closed-form ID-only solution are derived from the paper.



\---



\## Component Provenance



| Component | Status | Source / Notes |

|---|---|---|

| CIFAR-10 and SVHN data pipeline | Written by us | Implemented using `torchvision.datasets` and torchvision transforms |

| CIFAR-10 preprocessing | Written by us | Standard CIFAR normalization with random crop and horizontal flip for training |

| ResNet-18 backbone | Adapted | Based on `torchvision.models.resnet18`; modified for 32x32 CIFAR images |

| First convolution modification | Adapted by us | Changed to 3x3 convolution, stride 1, padding 1 |

| Initial max pooling removal | Adapted by us | Replaced standard ResNet max-pool with identity operation |

| Training pipeline | Written by us | SGD training, cosine annealing scheduler, checkpointing, evaluation and CSV logging |

| Per-epoch experiment logging | Written by us | Training/test loss, training/test accuracy and learning rate stored in `results/training\_log.csv` |

| Penultimate feature extraction | Written by us | Extracts the 512-dimensional ResNet-18 representation from all CIFAR-10 training samples |

| Feature-value percentile limits | Adapted from Zhao et al. | Uses the paper's 0.1 and 99.9 percentile limits |

| Number of feature intervals | Adapted from Zhao et al. | Uses K = 100 as specified in the paper |

| Interval-Specific Feature Impact (ISFI) | Adapted from Zhao et al. | Implementation follows the feature-impact formulation described in the paper |

| ID-only optimal theta calculation | Adapted from Zhao et al. | Implements the closed-form solution from Equation 14 |

| Feature reshaping at inference | Adapted from Zhao et al. | Features are rescaled according to their learned interval-specific theta value |

| MSP scoring | Written by us | Computed from the maximum softmax probability |

| MLS scoring | Written by us | Computed from the maximum model logit |

| Energy scoring | Written by us | Computed using log-sum-exp over logits |

| AUROC evaluation | Written by us | Calculated using scikit-learn with ID samples treated as the positive class |

| FPR95 evaluation | Written by us | Threshold chosen to accept 95% of ID samples, then OOD false-positive rate is measured |

| Result plotting | Written by us | Matplotlib plots generated from the saved CSV experiment logs |

| ResNet-18 / CIFAR-10 / SVHN experiment | Our adaptation | Compute-feasible reproduction setup; not the exact architecture or full OOD benchmark used in the paper |



\---



\## Use of the Official Repository



The official OptFSOOD repository was identified as a reference for the original work.



No official repository source files were copied directly into this project.



The feature-shaping implementation in this repository was written specifically for this reproduction based on the mathematical method described in the paper.



\---



\## Differences From the Original Experimental Setup



The reproduction intentionally differs from the complete experimental setup in the paper.



Our experiment uses:



\- ResNet-18 trained from scratch

\- CIFAR-10 as the in-distribution dataset

\- SVHN as the out-of-distribution dataset

\- A single OOD dataset for the primary reproduction experiment



The paper evaluates different CIFAR architectures and reports results across multiple OOD datasets.



Therefore, this repository should be considered an adapted reproduction of the core method rather than an exact replication of every result reported in the paper.



\---



\## Generated Artifacts



The following files are generated locally and are intentionally excluded from Git:



```text

data/

checkpoints/

artifacts/
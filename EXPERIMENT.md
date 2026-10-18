\# Additional Experiment — Sensitivity to Number of Feature Intervals



Branch: `experiment-k-sweep`



\## Question



How sensitive is Optimal Feature Shaping to the number of feature-value intervals K?



\## Experimental Setup



The trained ResNet-18 classifier, CIFAR-10 training features, CIFAR-10 test set,

SVHN test set, percentile limits, and OOD evaluation metrics were kept fixed.



Only the number of feature intervals K was changed.



Tested values:



\- K = 25

\- K = 50

\- K = 100

\- K = 200



No classifier retraining was performed.



\## Results



| K | Shaped MLS AUROC | Shaped MLS FPR95 | Shaped Energy AUROC | Shaped Energy FPR95 |

|---:|---:|---:|---:|---:|

| 25 | 88.46% | 58.58% | 88.32% | 60.15% |

| 50 | 88.43% | 58.86% | 88.29% | 59.95% |

| 100 | 88.42% | 59.08% | 88.28% | 59.99% |

| 200 | 88.42% | 59.20% | 88.28% | 59.71% |



\## Finding



OOD performance was largely insensitive to the tested number of feature intervals.



For shaped MLS, AUROC varied only from 88.42% to 88.46%.

For shaped Energy, AUROC varied only from 88.28% to 88.32%.



Changing K therefore did not recover the performance gap observed between

optimal feature shaping and the unmodified ResNet-18 OOD baselines.



The experiment suggests that the weaker shaping performance in our

CIFAR-10 to SVHN setup cannot be explained simply by the original choice of K=100.



\## Generated Files



\- `experiments/k\_sweep.py`

\- `results/k\_sweep\_results.csv`

\- `results/k\_sweep\_auroc.png`

\- `results/k\_sweep\_fpr95.png`


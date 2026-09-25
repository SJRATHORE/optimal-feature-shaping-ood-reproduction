# Optimal Feature Shaping for Out-of-Distribution Detection

Reproduction and ResNet-18 adaptation of:

**Towards Optimal Feature-Shaping Methods for Out-of-Distribution Detection**
Qinyu Zhao et al., ICLR 2024

This project reproduces the core ID-only optimal feature-shaping method proposed in the paper and evaluates it for out-of-distribution (OOD) detection using CIFAR-10 as the in-distribution dataset and SVHN as the OOD dataset.

## Authors

- Sudheer Rathore — 31683
- Anser Abbas — 31675

---

## 1. Project Goal

The paper studies feature-shaping methods for out-of-distribution detection.

Instead of modifying or retraining the neural network, feature shaping operates on the penultimate feature representation of a trained classifier before the final linear classification layer.

The proposed method divides feature values into intervals and estimates how much features inside each interval contribute to the model's maximum logit.

Using only in-distribution training data, the paper derives a closed-form optimal feature-shaping function.

This repository reproduces that procedure in a reduced CIFAR-scale setting.

---

## 2. Reproduction Setup

### In-Distribution Dataset

**CIFAR-10**

- 50,000 training images
- 10,000 test images
- 10 classes
- Image size: 32 x 32

### Out-of-Distribution Dataset

**SVHN test set**

- 26,032 images
- Used only during OOD evaluation

SVHN is never used to learn the feature-shaping parameters.

### Model

We use a **ResNet-18 trained from scratch on CIFAR-10**.

The standard torchvision ResNet-18 was adapted for 32 x 32 CIFAR images:

- First convolution changed to 3 x 3, stride 1
- Initial max-pooling removed
- Final classifier outputs 10 CIFAR-10 classes
- Penultimate feature dimension: 512

This is an adaptation of the paper rather than an exact architectural reproduction.

The original paper evaluates CIFAR using architectures including DenseNet101, ViT-B-16, and MLP-Mixer-Nano.

---

## 3. Training Configuration

The ResNet-18 classifier was trained for 50 epochs.

Configuration:

- Optimizer: SGD
- Initial learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- Batch size: 128
- Scheduler: Cosine Annealing
- Random seed: 42
- Loss: Cross Entropy

Training augmentation:

- Random crop with padding 4
- Random horizontal flip
- CIFAR-10 normalization

Best CIFAR-10 test accuracy:

**94.42%**

The best checkpoint occurred at epoch 49.

---

## 4. Training Logs and Evidence

Training metrics were recorded for every epoch.

The complete training log is stored in:

```text
results/training_log.csv
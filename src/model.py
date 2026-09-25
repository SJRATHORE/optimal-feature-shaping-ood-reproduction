import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10.

    Exposes the 512-dimensional penultimate feature vector
    so we can later apply the paper's feature-shaping method.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        model = resnet18(weights=None)

        # CIFAR-10 images are 32x32, so use a smaller first convolution.
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove the aggressive ImageNet max pooling.
        model.maxpool = nn.Identity()

        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

        self.feature_dim = model.fc.in_features  # 512
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = CIFARResNet18()

    x = torch.randn(8, 3, 32, 32)

    features = model.forward_features(x)
    logits = model(x)

    print("Input shape:   ", x.shape)
    print("Feature shape: ", features.shape)
    print("Logit shape:   ", logits.shape)
    print("Feature dim:   ", model.feature_dim)

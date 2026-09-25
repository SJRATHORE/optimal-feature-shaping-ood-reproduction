from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data import get_transforms
from src.model import CIFARResNet18


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Use deterministic preprocessing for feature extraction
    _, eval_transform = get_transforms()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    checkpoint_path = Path(
        "checkpoints/best_resnet18_cifar10.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = CIFARResNet18(num_classes=10).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint test accuracy: "
        f"{checkpoint['test_accuracy']:.2f}%"
    )

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)

            features = model.forward_features(images)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print()
    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels shape:         {labels.shape}")
    print(f"Feature dtype:        {features.dtype}")

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "cifar10_train_features.pt"

    torch.save(
        {
            "features": features,
            "labels": labels,
        },
        output_path,
    )

    print()
    print(f"Saved features to: {output_path}")


if __name__ == "__main__":
    main()
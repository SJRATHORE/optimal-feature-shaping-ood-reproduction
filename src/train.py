import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.model import CIFARResNet18


def main():
    # Use GPU automatically if one exists.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load datasets
    train_loader, _, _ = get_dataloaders()

    # Create model
    model = CIFARResNet18(num_classes=10).to(device)

    # Standard classification loss
    criterion = nn.CrossEntropyLoss()

    # Temporary optimizer for sanity checking
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Get one real CIFAR-10 batch
    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")

    # Forward pass
    logits = model(images)

    print(f"Logits: {logits.shape}")

    # Compute loss
    loss = criterion(logits, labels)

    print(f"Loss before update: {loss.item():.4f}")

    # One training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Backward pass: OK")
    print("Optimizer step: OK")


if __name__ == "__main__":
    main()


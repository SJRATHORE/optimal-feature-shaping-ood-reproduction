from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    """
    Return training and evaluation transforms.

    Training uses basic augmentation.
    Evaluation uses deterministic preprocessing only.
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(
    data_dir="./data",
    batch_size=128,
    num_workers=2,
):
    """
    Create DataLoaders for:

    CIFAR-10 training set  -> model training
    CIFAR-10 test set      -> in-distribution evaluation
    SVHN test set          -> out-of-distribution evaluation
    """

    data_dir = Path(data_dir)

    train_transform, eval_transform = get_transforms()

    cifar10_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    cifar10_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    svhn_test = datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    cifar10_test_loader = DataLoader(
        cifar10_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    svhn_test_loader = DataLoader(
        svhn_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, cifar10_test_loader, svhn_test_loader


if __name__ == "__main__":
    train_loader, cifar_test_loader, svhn_loader = get_dataloaders()

    print(f"CIFAR-10 training images: {len(train_loader.dataset)}")
    print(f"CIFAR-10 test images:     {len(cifar_test_loader.dataset)}")
    print(f"SVHN test images:         {len(svhn_loader.dataset)}")

    images, labels = next(iter(train_loader))

    print()
    print("One CIFAR-10 training batch:")
    print(f"Image tensor shape: {images.shape}")
    print(f"Label tensor shape: {labels.shape}")
    print(f"Image dtype:        {images.dtype}")
    print(f"Label dtype:        {labels.dtype}")

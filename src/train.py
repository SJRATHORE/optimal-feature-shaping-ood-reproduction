import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import get_dataloaders
from src.model import CIFARResNet18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training")

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    accuracy = 100.0 * correct / total

    return loss, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CIFARResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    log_path = results_dir / "training_log.csv"

    best_accuracy = 0.0

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "learning_rate",
        ])

        for epoch in range(1, args.epochs + 1):

            print()
            print(f"Epoch {epoch}/{args.epochs}")

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            test_loss, test_accuracy = evaluate(
                model,
                test_loader,
                criterion,
                device,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}%"
            )

            print(
                f"Test Loss:  {test_loss:.4f} | "
                f"Test Accuracy:  {test_accuracy:.2f}%"
            )

            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                current_lr,
            ])

            csv_file.flush()

            if test_accuracy > best_accuracy:

                best_accuracy = test_accuracy

                checkpoint_path = checkpoint_dir / "best_resnet18_cifar10.pt"

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_accuracy": test_accuracy,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )

                print(
                    f"New best model saved "
                    f"({best_accuracy:.2f}%)"
                )

            scheduler.step()

    print()
    print("Training complete.")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()

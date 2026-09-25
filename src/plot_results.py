from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_accuracy(training_df, output_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(
        training_df["epoch"],
        training_df["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        training_df["epoch"],
        training_df["test_accuracy"],
        label="Test Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ResNet-18 CIFAR-10 Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "training_accuracy.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_training_loss(training_df, output_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(
        training_df["epoch"],
        training_df["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        training_df["epoch"],
        training_df["test_loss"],
        label="Test Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("ResNet-18 CIFAR-10 Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "training_loss.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_auroc(ood_df, output_dir):
    plt.figure(figsize=(10, 5))

    plt.bar(
        ood_df["method"],
        ood_df["auroc"],
    )

    plt.xlabel("Method")
    plt.ylabel("AUROC (%)")
    plt.title("CIFAR-10 vs SVHN OOD Detection — AUROC")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = output_dir / "ood_auroc.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_fpr95(ood_df, output_dir):
    plt.figure(figsize=(10, 5))

    plt.bar(
        ood_df["method"],
        ood_df["fpr95"],
    )

    plt.xlabel("Method")
    plt.ylabel("FPR95 (%)")
    plt.title("CIFAR-10 vs SVHN OOD Detection — FPR95")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = output_dir / "ood_fpr95.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    results_dir = Path("results")

    training_path = results_dir / "training_log.csv"
    ood_path = results_dir / "ood_results.csv"

    print("Loading results...")

    training_df = pd.read_csv(training_path)
    ood_df = pd.read_csv(ood_path)

    print(f"Training epochs: {len(training_df)}")
    print(f"OOD methods:     {len(ood_df)}")
    print()

    plot_training_accuracy(
        training_df,
        results_dir,
    )

    plot_training_loss(
        training_df,
        results_dir,
    )

    plot_auroc(
        ood_df,
        results_dir,
    )

    plot_fpr95(
        ood_df,
        results_dir,
    )

    print()
    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
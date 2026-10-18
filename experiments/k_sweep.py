from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data import get_transforms
from src.model import CIFARResNet18
from src.feature_shaping import OptimalFeatureShaper
from src.evaluate_ood import compute_metrics


K_VALUES = [25, 50, 100, 200]

FEATURE_PATH = Path("artifacts/cifar10_train_features.pt")
CHECKPOINT_PATH = Path("checkpoints/best_resnet18_cifar10.pt")
RESULTS_DIR = Path("results")


def extract_features(model, loader, device):
    model.eval()
    all_features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            features = model.forward_features(images)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def shaped_scores(model, features, shaper, device):
    model.eval()

    scores_mls = []
    scores_energy = []

    batch_size = 1024

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))

            batch = features[start:end].to(device)

            shaped = shaper.transform(batch)
            logits = model.classifier(shaped)

            mls = logits.max(dim=1).values
            energy = torch.logsumexp(logits, dim=1)

            scores_mls.append(mls.cpu())
            scores_energy.append(energy.cpu())

    return (
        torch.cat(scores_mls).numpy(),
        torch.cat(scores_energy).numpy(),
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load training features used to learn the shaper
    # --------------------------------------------------

    print("Loading CIFAR-10 training features...")

    feature_data = torch.load(
        FEATURE_PATH,
        map_location="cpu",
    )

    train_features = feature_data["features"]

    print(f"Training features: {train_features.shape}")

    # --------------------------------------------------
    # Load trained ResNet-18
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = CIFARResNet18(num_classes=10).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    classifier_weight = (
        model.classifier.weight
        .detach()
        .cpu()
    )

    print(
        f"Loaded checkpoint "
        f"(test accuracy: {checkpoint['test_accuracy']:.2f}%)"
    )

    # --------------------------------------------------
    # Load evaluation datasets
    # --------------------------------------------------

    _, eval_transform = get_transforms()

    cifar_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=eval_transform,
    )

    svhn_test = datasets.SVHN(
        root="./data",
        split="test",
        download=False,
        transform=eval_transform,
    )

    cifar_loader = DataLoader(
        cifar_test,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    svhn_loader = DataLoader(
        svhn_test,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    # Extract test features once.
    # We reuse the same features for every K value.

    print()
    print("Extracting CIFAR-10 test features...")
    id_features = extract_features(
        model,
        cifar_loader,
        device,
    )

    print("Extracting SVHN test features...")
    ood_features = extract_features(
        model,
        svhn_loader,
        device,
    )

    print(f"ID features:  {id_features.shape}")
    print(f"OOD features: {ood_features.shape}")

    # --------------------------------------------------
    # K sweep
    # --------------------------------------------------

    results = []

    for k in K_VALUES:
        print()
        print("=" * 60)
        print(f"EXPERIMENT: K = {k}")
        print("=" * 60)

        shaper = OptimalFeatureShaper(
            num_intervals=k,
            lower_percentile=0.1,
            upper_percentile=99.9,
        )

        shaper.fit(
            train_features,
            classifier_weight,
        )

        id_mls, id_energy = shaped_scores(
            model,
            id_features,
            shaper,
            device,
        )

        ood_mls, ood_energy = shaped_scores(
            model,
            ood_features,
            shaper,
            device,
        )

        mls_auroc, mls_fpr95 = compute_metrics(
            id_mls,
            ood_mls,
        )

        energy_auroc, energy_fpr95 = compute_metrics(
            id_energy,
            ood_energy,
        )

        results.append([
            k,
            mls_auroc,
            mls_fpr95,
            energy_auroc,
            energy_fpr95,
        ])

        print(
            f"MLS:    AUROC={mls_auroc:.2f}% "
            f"FPR95={mls_fpr95:.2f}%"
        )

        print(
            f"Energy: AUROC={energy_auroc:.2f}% "
            f"FPR95={energy_fpr95:.2f}%"
        )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    csv_path = RESULTS_DIR / "k_sweep_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "K",
            "shaped_mls_auroc",
            "shaped_mls_fpr95",
            "shaped_energy_auroc",
            "shaped_energy_fpr95",
        ])

        writer.writerows(results)

    print()
    print(f"Saved results to: {csv_path}")

    # --------------------------------------------------
    # Plot AUROC
    # --------------------------------------------------

    ks = [row[0] for row in results]
    mls_aurocs = [row[1] for row in results]
    energy_aurocs = [row[3] for row in results]

    plt.figure()
    plt.plot(ks, mls_aurocs, marker="o", label="Shaped MLS")
    plt.plot(
        ks,
        energy_aurocs,
        marker="o",
        label="Shaped Energy",
    )

    plt.xlabel("Number of Feature Intervals (K)")
    plt.ylabel("AUROC (%)")
    plt.title("Sensitivity to Number of Feature Intervals")
    plt.legend()
    plt.grid(True)

    auroc_path = RESULTS_DIR / "k_sweep_auroc.png"

    plt.savefig(
        auroc_path,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot FPR95
    # --------------------------------------------------

    mls_fprs = [row[2] for row in results]
    energy_fprs = [row[4] for row in results]

    plt.figure()
    plt.plot(ks, mls_fprs, marker="o", label="Shaped MLS")
    plt.plot(
        ks,
        energy_fprs,
        marker="o",
        label="Shaped Energy",
    )

    plt.xlabel("Number of Feature Intervals (K)")
    plt.ylabel("FPR95 (%)")
    plt.title("Sensitivity to Number of Feature Intervals")
    plt.legend()
    plt.grid(True)

    fpr_path = RESULTS_DIR / "k_sweep_fpr95.png"

    plt.savefig(
        fpr_path,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {auroc_path}")
    print(f"Saved plot: {fpr_path}")


if __name__ == "__main__":
    main()
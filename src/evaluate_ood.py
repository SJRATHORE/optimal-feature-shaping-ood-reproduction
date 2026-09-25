from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import roc_auc_score

from src.data import get_transforms
from src.model import CIFARResNet18
from src.feature_shaping import OptimalFeatureShaper


def compute_metrics(id_scores, ood_scores):
    """
    Higher score = more likely to be ID.

    AUROC:
        ID samples are positive class.

    FPR95:
        Choose a threshold accepting 95% of ID samples,
        then measure how many OOD samples are incorrectly
        accepted as ID.
    """

    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)

    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])

    scores = np.concatenate([
        id_scores,
        ood_scores,
    ])

    auroc = roc_auc_score(
        labels,
        scores,
    ) * 100.0

    threshold = np.percentile(
        id_scores,
        5.0,
    )

    fpr95 = (
        np.mean(ood_scores >= threshold)
        * 100.0
    )

    return auroc, fpr95


def collect_scores(
    model,
    shaper,
    loader,
    device,
):
    model.eval()

    msp_scores = []
    mls_scores = []
    energy_scores = []
    shaped_mls_scores = []
    shaped_energy_scores = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            
            # Raw model
            

            features = model.forward_features(images)

            logits = model.classifier(features)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            # Maximum Softmax Probability
            msp = probabilities.max(
                dim=1
            ).values

            # Maximum Logit Score
            mls = logits.max(
                dim=1
            ).values

            # Energy score
            energy = torch.logsumexp(
                logits,
                dim=1,
            )

            
            # Optimal feature shaping
            

            shaped_features = shaper.transform(
                features
            )

            shaped_logits = model.classifier(
                shaped_features
            )

            shaped_mls = shaped_logits.max(
                dim=1
            ).values

            shaped_energy = torch.logsumexp(
                shaped_logits,
                dim=1,
            )

            # Save scores
            msp_scores.append(msp.cpu())
            mls_scores.append(mls.cpu())
            energy_scores.append(energy.cpu())
            shaped_mls_scores.append(
                shaped_mls.cpu()
            )
            shaped_energy_scores.append(
                shaped_energy.cpu()
            )

    return {
        "MSP": torch.cat(
            msp_scores
        ).numpy(),

        "MLS": torch.cat(
            mls_scores
        ).numpy(),

        "Energy": torch.cat(
            energy_scores
        ).numpy(),

        "Optimal Shaping (MLS)": torch.cat(
            shaped_mls_scores
        ).numpy(),

        "Optimal Shaping (Energy)": torch.cat(
            shaped_energy_scores
        ).numpy(),
    }


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    
    # Datasets
    

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

    
    # Load model
    

    checkpoint = torch.load(
        "checkpoints/best_resnet18_cifar10.pt",
        map_location=device,
    )

    model = CIFARResNet18(
        num_classes=10
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded ResNet-18 checkpoint "
        f"(test accuracy: "
        f"{checkpoint['test_accuracy']:.2f}%)"
    )

    # --------------------------------------------------
    # Load learned shaping function
    
    shaper_state = torch.load(
        "artifacts/optimal_feature_shaper.pt",
        map_location="cpu",
    )

    shaper = OptimalFeatureShaper(
        num_intervals=shaper_state[
            "num_intervals"
        ],
        lower_percentile=shaper_state[
            "lower_percentile"
        ],
        upper_percentile=shaper_state[
            "upper_percentile"
        ],
    )

    shaper.alpha = shaper_state["alpha"]
    shaper.beta = shaper_state["beta"]
    shaper.theta = shaper_state["theta"]

    print("Loaded optimal feature shaper.")

    
    # Obtain OOD scores
    

    print()
    print("Evaluating CIFAR-10 (ID)...")

    id_scores = collect_scores(
        model,
        shaper,
        cifar_loader,
        device,
    )

    print("Evaluating SVHN (OOD)...")

    ood_scores = collect_scores(
        model,
        shaper,
        svhn_loader,
        device,
    )

    
    # Metrics
    

    results = []

    print()
    print("=" * 60)
    print("OOD DETECTION RESULTS")
    print("=" * 60)

    print(
        f"{'Method':<28}"
        f"{'AUROC ↑':>12}"
        f"{'FPR95 ↓':>12}"
    )

    print("-" * 60)

    for method in id_scores.keys():

        auroc, fpr95 = compute_metrics(
            id_scores[method],
            ood_scores[method],
        )

        results.append(
            [method, auroc, fpr95]
        )

        print(
            f"{method:<28}"
            f"{auroc:>11.2f}%"
            f"{fpr95:>11.2f}%"
        )

    
    # Save results
    

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = (
        results_dir
        / "ood_results.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "method",
            "auroc",
            "fpr95",
        ])

        writer.writerows(results)

    print()
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
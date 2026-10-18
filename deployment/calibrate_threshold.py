from pathlib import Path
import json

import numpy as np
import torch

from src.model import CIFARResNet18
from src.feature_shaping import OptimalFeatureShaper


FEATURE_PATH = Path("artifacts/cifar10_train_features.pt")
CHECKPOINT_PATH = Path("checkpoints/best_resnet18_cifar10.pt")
SHAPER_PATH = Path("artifacts/optimal_feature_shaper.pt")
OUTPUT_PATH = Path("deployment/calibration.json")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load ResNet-18
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = CIFARResNet18(num_classes=10).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Load learned shaping function
    shaper_state = torch.load(
        SHAPER_PATH,
        map_location="cpu",
    )

    shaper = OptimalFeatureShaper(
        num_intervals=shaper_state["num_intervals"],
        lower_percentile=shaper_state["lower_percentile"],
        upper_percentile=shaper_state["upper_percentile"],
    )

    shaper.alpha = shaper_state["alpha"]
    shaper.beta = shaper_state["beta"]
    shaper.theta = shaper_state["theta"]

    # Load CIFAR-10 training features
    feature_data = torch.load(
        FEATURE_PATH,
        map_location="cpu",
    )

    features = feature_data["features"]

    print(f"Training features: {features.shape}")

    scores = []

    batch_size = 1024

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(
                start + batch_size,
                len(features),
            )

            batch = features[start:end].to(device)

            shaped = shaper.transform(batch)

            logits = model.classifier(shaped)

            energy = torch.logsumexp(
                logits,
                dim=1,
            )

            scores.append(
                energy.cpu()
            )

    scores = torch.cat(scores).numpy()

    # Higher energy = more ID-like.
    # 5th percentile accepts approximately 95% of ID training samples.
    threshold = float(
        np.percentile(scores, 5.0)
    )

    output = {
        "method": "Optimal Shaping (Energy)",
        "threshold": threshold,
        "calibration_dataset": "CIFAR-10 training set",
        "id_acceptance_target": 0.95,
        "num_calibration_samples": len(scores),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("Calibration complete.")
    print(f"Threshold: {threshold:.6f}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
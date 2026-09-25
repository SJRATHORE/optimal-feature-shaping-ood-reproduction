from pathlib import Path
import math

import torch


class OptimalFeatureShaper:
    """
    ID-only optimal feature shaping from Zhao et al. (ICLR 2024).

    The feature-value range is divided into K equal-width intervals.
    For every interval, we compute the expected interval-specific
    feature impact (ISFI) over the ID training set.

    Eq. 14 of the paper gives the closed-form optimal theta.
    """

    def __init__(
        self,
        num_intervals=100,
        lower_percentile=0.1,
        upper_percentile=99.9,
    ):
        self.num_intervals = num_intervals
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        self.alpha = None
        self.beta = None
        self.theta = None

    def fit(
        self,
        features,
        classifier_weight,
        batch_size=1024,
    ):
        """
        Learn theta* using only ID training features.

        features:
            Tensor [N, M]
            N samples, M-dimensional penultimate features.

        classifier_weight:
            Tensor [C, M]
            Weight matrix of the final linear classifier.
        """

        features = features.float().cpu()
        classifier_weight = classifier_weight.float().cpu()

        n_samples, feature_dim = features.shape

        print(
            f"Fitting shaper on {n_samples} samples "
            f"with {feature_dim}-D features"
        )

        # --------------------------------------------------
        # 1. Determine alpha and beta from all training
        #    feature values.
        #
        # Paper: 0.1th and 99.9th percentiles.
        # --------------------------------------------------

        flattened = features.reshape(-1)

        percentiles = torch.tensor(
            [
                self.lower_percentile / 100.0,
                self.upper_percentile / 100.0,
            ],
            dtype=torch.float32,
        )

        limits = torch.quantile(
            flattened,
            percentiles,
        )

        self.alpha = limits[0].item()
        self.beta = limits[1].item()

        print(f"alpha ({self.lower_percentile}%): {self.alpha:.6f}")
        print(f"beta  ({self.upper_percentile}%): {self.beta:.6f}")

        if self.beta <= self.alpha:
            raise ValueError(
                "Invalid feature interval: beta must be greater than alpha."
            )

        delta = (
            self.beta - self.alpha
        ) / self.num_intervals

        # Sum of ISFI vectors over all samples
        isfi_sum = torch.zeros(
            self.num_intervals,
            dtype=torch.float64,
        )

        # --------------------------------------------------
        # 2. Compute ISFI for every training sample.
        # --------------------------------------------------

        for start in range(0, n_samples, batch_size):

            end = min(
                start + batch_size,
                n_samples,
            )

            z = features[start:end]

            # Eq. 3 / Appendix B:
            # determine the class producing the maximum logit.
            #
            # We use Wz here, matching the paper's derivation
            # where the classifier bias is disregarded.
            logits_no_bias = (
                z @ classifier_weight.T
            )

            max_classes = logits_no_bias.argmax(dim=1)

            # Weight vector corresponding to each sample's
            # maximum-logit class.
            w_max = classifier_weight[max_classes]

            # Contribution w_i * z_i
            contributions = w_max * z

            # Features outside [alpha, beta) are not assigned
            # to an interval. This effectively removes the
            # extreme tails selected by the percentiles.
            valid = (
                (z >= self.alpha)
                & (z < self.beta)
            )

            # Convert every feature value to its interval index.
            interval_indices = torch.floor(
                (z - self.alpha) / delta
            ).long()

            # Clamp only so scatter_add receives safe indices.
            # Invalid values are zeroed using the mask below.
            interval_indices = interval_indices.clamp(
                0,
                self.num_intervals - 1,
            )

            contributions = torch.where(
                valid,
                contributions,
                torch.zeros_like(contributions),
            )

            # I(z): one ISFI value per interval per sample.
            batch_isfi = torch.zeros(
                z.size(0),
                self.num_intervals,
                dtype=torch.float32,
            )

            batch_isfi.scatter_add_(
                dim=1,
                index=interval_indices,
                src=contributions,
            )

            isfi_sum += batch_isfi.sum(
                dim=0,
                dtype=torch.float64,
            )

        # --------------------------------------------------
        # 3. Expected ISFI: E_ID[I(z)]
        # --------------------------------------------------

        mean_isfi = (
            isfi_sum / n_samples
        ).float()

        norm = torch.linalg.vector_norm(mean_isfi)

        if norm.item() == 0:
            raise ValueError(
                "Mean ISFI norm is zero; cannot compute theta."
            )

        # --------------------------------------------------
        # 4. Equation 14
        #
        # theta* = sqrt(K) / ||E[I(z)]|| * E[I(z)]
        # --------------------------------------------------

        self.theta = (
            math.sqrt(self.num_intervals)
            * mean_isfi
            / norm
        )

        print()
        print("Optimal shaping function learned.")
        print(f"Theta shape: {self.theta.shape}")
        print(
            f"Theta L2 norm: "
            f"{torch.linalg.vector_norm(self.theta).item():.6f}"
        )
        print(
            f"Theta minimum: {self.theta.min().item():.6f}"
        )
        print(
            f"Theta maximum: {self.theta.max().item():.6f}"
        )

        return self

    def transform(self, features):
        """
        Apply learned feature shaping.

        Each feature z_i is multiplied by theta_k if it falls
        inside interval k.

        Values outside [alpha, beta) are set to zero.
        """

        if self.theta is None:
            raise RuntimeError(
                "The shaper must be fitted before transform()."
            )

        original_device = features.device

        theta = self.theta.to(original_device)

        alpha = self.alpha
        beta = self.beta

        delta = (
            beta - alpha
        ) / self.num_intervals

        valid = (
            (features >= alpha)
            & (features < beta)
        )

        interval_indices = torch.floor(
            (features - alpha) / delta
        ).long()

        interval_indices = interval_indices.clamp(
            0,
            self.num_intervals - 1,
        )

        scaling = theta[interval_indices]

        shaped = torch.where(
            valid,
            features * scaling,
            torch.zeros_like(features),
        )

        return shaped

    def state_dict(self):
        return {
            "num_intervals": self.num_intervals,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
        }


def main():

    feature_path = Path(
        "artifacts/cifar10_train_features.pt"
    )

    checkpoint_path = Path(
        "checkpoints/best_resnet18_cifar10.pt"
    )

    print("Loading CIFAR-10 training features...")

    feature_data = torch.load(
        feature_path,
        map_location="cpu",
    )

    features = feature_data["features"]

    print(
        f"Feature matrix: {features.shape}"
    )

    print("Loading classifier weights...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    classifier_weight = checkpoint[
        "model_state_dict"
    ]["classifier.weight"]

    print(
        f"Classifier weights: "
        f"{classifier_weight.shape}"
    )

    shaper = OptimalFeatureShaper(
        num_intervals=100,
        lower_percentile=0.1,
        upper_percentile=99.9,
    )

    shaper.fit(
        features,
        classifier_weight,
    )

    output_path = Path(
        "artifacts/optimal_feature_shaper.pt"
    )

    torch.save(
        shaper.state_dict(),
        output_path,
    )

    print()
    print(
        f"Saved optimal shaping parameters to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
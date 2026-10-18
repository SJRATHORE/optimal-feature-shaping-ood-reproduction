from pathlib import Path

from torchvision.datasets import CIFAR10, SVHN


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "deployment" / "examples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CIFAR-10 examples

cifar = CIFAR10(
    root=ROOT / "data",
    train=False,
    download=False,
)

wanted_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "ship",
]

for class_name in wanted_classes:

    target = cifar.class_to_idx[class_name]

    for index in range(len(cifar)):

        image, label = cifar[index]

        if label == target:

            output = (
                OUTPUT_DIR /
                f"cifar_{class_name}.png"
            )

            image.save(output)

            print(
                f"Saved {class_name}: {output}"
            )

            break


# --------------------------------------------------
# SVHN examples

svhn = SVHN(
    root=ROOT / "data",
    split="test",
    download=False,
)

wanted_digits = [0, 1, 2, 3, 4]

for digit in wanted_digits:

    for index in range(len(svhn)):

        image, label = svhn[index]

        if int(label) == digit:

            output = (
                OUTPUT_DIR /
                f"svhn_{digit}.png"
            )

            image.save(output)

            print(
                f"Saved digit {digit}: {output}"
            )

            break


print()
print("Example catalog created.")
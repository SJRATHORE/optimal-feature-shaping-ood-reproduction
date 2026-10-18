from pathlib import Path
from io import BytesIO
import json

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.data import get_transforms
from src.model import CIFARResNet18
from src.feature_shaping import OptimalFeatureShaper

from torchvision import transforms
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ----------------Paths


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet18_cifar10.pt"
SHAPER_PATH = ROOT / "artifacts" / "optimal_feature_shaper.pt"
CALIBRATION_PATH = ROOT / "deployment" / "calibration.json"


# ------------------- CIFAR-10 labels


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ------------------------- Device


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------- Load trained classifier


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model = CIFARResNet18(num_classes=10).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ------------------ Load optimal feature shaper


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


# ----------------------- Load OOD threshold


with open(
    CALIBRATION_PATH,
    "r",
    encoding="utf-8",
) as file:
    calibration = json.load(file)

OOD_THRESHOLD = float(
    calibration["threshold"]
)


# ------------------------ Preprocessing


upload_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


# ------------------- FastAPI application


app = FastAPI(
    title="Optimal Feature Shaping OOD Detection API",
    description=(
        "ResNet-18 CIFAR-10 classifier with ID-only "
        "Optimal Feature Shaping for OOD detection."
    ),
    version="1.0",
)

app.mount(
    "/examples",
    StaticFiles(
        directory=ROOT / "deployment" / "examples"
    ),
    name="examples",
)

@app.get("/")
def root():
    return FileResponse(
        ROOT / "deployment" / "index.html"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "ood_method": "Optimal Shaping (Energy)",
        "ood_threshold": OOD_THRESHOLD,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        )

    image_tensor = (
        upload_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    # ------------- Classification + OOD detection


    with torch.no_grad():

        # Raw ResNet features
        features = model.forward_features(
            image_tensor
        )

        # Standard CIFAR-10 prediction
        raw_logits = model.classifier(
            features
        )

        probabilities = torch.softmax(
            raw_logits,
            dim=1,
        )

        confidence, predicted_index = (
            probabilities.max(dim=1)
        )

        predicted_index = int(
            predicted_index.item()
        )

        confidence = float(
            confidence.item()
        )

        raw_class = CIFAR10_CLASSES[
            predicted_index
        ]

        # Apply optimal feature shaping
        shaped_features = shaper.transform(
            features
        )

        shaped_logits = model.classifier(
            shaped_features
        )

        # Energy score:
        # higher = more ID-like
        ood_score = float(
            torch.logsumexp(
                shaped_logits,
                dim=1,
            ).item()
        )

    # =------------------ OOD decision


    is_ood = ood_score < OOD_THRESHOLD

    predicted_class = (
        None if is_ood else raw_class
    )

    return {
        "filename": file.filename,
        "is_ood": is_ood,
        "predicted_class": predicted_class,
        "classifier_prediction": raw_class,
        "confidence": round(confidence, 6),
        "ood_score": round(ood_score, 6),
        "ood_threshold": round(OOD_THRESHOLD, 6),
        "ood_method": "Optimal Shaping (Energy)",
    }

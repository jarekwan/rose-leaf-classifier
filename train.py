import json
import os
from pathlib import Path

import torch
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from google.cloud import storage


BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
DATA_PREFIX = "rose-leaf/RESELEAFSET"
MODEL_PREFIX = "rose-leaf/models"
LOCAL_DATA_DIR = Path("/tmp/rose-leaf-dataset")
MODEL_PATH = Path("/tmp/rose_leaf_resnet18.pth")
METRICS_PATH = Path("/tmp/rose_leaf_metrics.json")

CLASS_NAMES = [
    "Black Spot",
    "Dry Leaf",
    "Healthy Leaf",
    "Leaf Holes",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

ImageFile.LOAD_TRUNCATED_IMAGES = True


def download_images_from_gcs() -> int:
    """Pobiera zdjęcia z folderów klas w buckecie do lokalnego katalogu."""
    if not BUCKET_NAME:
        raise RuntimeError("Ustaw zmienną środowiskową GCS_BUCKET_NAME.")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    downloaded = 0

    for class_name in CLASS_NAMES:
        class_prefix = f"{DATA_PREFIX}/{class_name}/"
        class_dir = LOCAL_DATA_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        blobs = bucket.list_blobs(prefix=class_prefix)
        class_count = 0

        for blob in blobs:
            relative_name = blob.name[len(class_prefix):]

            if not relative_name:
                continue

            if Path(relative_name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination = class_dir / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))

            downloaded += 1
            class_count += 1

        print(f"{class_name}: pobrano {class_count} zdjęć", flush=True)

        if class_count == 0:
            raise RuntimeError(
                f"Nie znaleziono zdjęć w gs://{BUCKET_NAME}/{class_prefix}"
            )

    return downloaded


def create_model(number_of_classes: int) -> nn.Module:
    """Tworzy ResNet18 z wytrenowanymi wagami i nową warstwą klas."""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, number_of_classes)

    return model


def train() -> None:
    """Pobiera zdjęcia, trenuje model i zapisuje wyniki w GCS."""
    downloaded = download_images_from_gcs()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    base_dataset = datasets.ImageFolder(str(LOCAL_DATA_DIR))

    if base_dataset.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Nieoczekiwane klasy: {base_dataset.classes}. "
            f"Oczekiwano: {CLASS_NAMES}."
        )

    indices = list(range(len(base_dataset)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=base_dataset.targets,
    )

    train_dataset = Subset(
        datasets.ImageFolder(
            str(LOCAL_DATA_DIR),
            transform=train_transform,
        ),
        train_indices,
    )
    validation_dataset = Subset(
        datasets.ImageFolder(
            str(LOCAL_DATA_DIR),
            transform=validation_transform,
        ),
        validation_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(CLASS_NAMES)).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()

        # Zamrożone warstwy BatchNorm nie aktualizują statystyk.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                predicted_classes = predictions.argmax(dim=1)

                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        average_loss = total_loss / len(train_dataset)
        validation_accuracy = correct / total if total else 0.0

        print(
            f"Epoka {epoch + 1}/{EPOCHS} — "
            f"strata treningowa: {average_loss:.4f}, "
            f"trafność walidacyjna: {validation_accuracy:.4f}",
            flush=True,
        )

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": 224,
    }
    torch.save(checkpoint, MODEL_PATH)

    metrics = {
        "downloaded_images": downloaded,
        "classes": CLASS_NAMES,
        "epochs": EPOCHS,
        "validation_accuracy": validation_accuracy,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bucket = storage.Client().bucket(BUCKET_NAME)
    bucket.blob(f"{MODEL_PREFIX}/rose_leaf_resnet18.pth").upload_from_filename(
        str(MODEL_PATH),
        content_type="application/octet-stream",
    )
    bucket.blob(f"{MODEL_PREFIX}/training_metrics.json").upload_from_filename(
        str(METRICS_PATH),
        content_type="application/json",
    )

    print(
        f"Model zapisano w "
        f"gs://{BUCKET_NAME}/{MODEL_PREFIX}/rose_leaf_resnet18.pth",
        flush=True,
    )
    print(
        f"Metryki zapisano w "
        f"gs://{BUCKET_NAME}/{MODEL_PREFIX}/training_metrics.json",
        flush=True,
    )


if __name__ == "__main__":
    train()
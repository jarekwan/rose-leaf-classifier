# import html
# import io
# import os

# import torch
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse
# from google.cloud import storage
# from PIL import Image, UnidentifiedImageError
# from torchvision import models, transforms
# from torch import nn


# BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
# MODEL_OBJECT = "rose-leaf/models/rose_leaf_resnet18.pth"

# app = FastAPI()
# model = None
# class_names = None

# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])


# def load_model():
#     """Pobiera wytrenowany model z GCS i przygotowuje go do predykcji."""
#     global model, class_names

#     if model is not None:
#         return model, class_names

#     if not BUCKET_NAME:
#         raise RuntimeError("Brak zmiennej GCS_BUCKET_NAME.")

#     bucket = storage.Client().bucket(BUCKET_NAME)
#     blob = bucket.blob(MODEL_OBJECT)

#     if not blob.exists():
#         raise FileNotFoundError(
#             f"Nie znaleziono modelu gs://{BUCKET_NAME}/{MODEL_OBJECT}"
#         )

#     checkpoint_bytes = blob.download_as_bytes()
#     checkpoint = torch.load(
#         io.BytesIO(checkpoint_bytes),
#         map_location="cpu",
#         weights_only=True,
#     )

#     class_names = checkpoint["class_names"]
#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, len(class_names))
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     return model, class_names


# @app.get("/", response_class=HTMLResponse)
# def home():
#     return """
#     <!doctype html>
#     <html lang="pl">
#     <head><meta charset="utf-8"><title>Klasyfikator liści róży</title></head>
#     <body>
#         <h1>Klasyfikator liści róży</h1>
#         <p>Wybierz zdjęcie liścia. Aplikacja oceni, do której z czterech
#         klas należy.</p>
#         <form action="/predict" method="post" enctype="multipart/form-data">
#             <input type="file" name="file" accept="image/*" required>
#             <button type="submit">Sprawdź zdjęcie</button>
#         </form>
#     </body>
#     </html>
#     """


# @app.post("/predict", response_class=HTMLResponse)
# async def predict(file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("image/"):
#         return HTMLResponse(
#             "<h1>Wybierz plik graficzny.</h1><p><a href='/'>Wróć</a></p>",
#             status_code=400,
#         )

#     try:
#         image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     except (UnidentifiedImageError, OSError):
#         return HTMLResponse(
#             "<h1>Nie udało się odczytać zdjęcia.</h1><p><a href='/'>Wróć</a></p>",
#             status_code=400,
#         )

#     try:
#         classifier, labels = load_model()
#         image_tensor = image_transform(image).unsqueeze(0)

#         with torch.no_grad():
#             scores = classifier(image_tensor)
#             probabilities = torch.softmax(scores, dim=1)[0]
#             predicted_index = int(probabilities.argmax())

#         predicted_class = labels[predicted_index]
#         confidence = float(probabilities[predicted_index]) * 100
#         safe_filename = html.escape(file.filename or "zdjęcie")

#         return HTMLResponse(f"""
#         <!doctype html>
#         <html lang="pl">
#         <head><meta charset="utf-8"><title>Wynik klasyfikacji</title></head>
#         <body>
#             <h1>Wynik klasyfikacji</h1>
#             <p>Plik: {safe_filename}</p>
#             <p>Rozpoznana klasa: <strong>{html.escape(predicted_class)}</strong></p>
#             <p>Pewność modelu: {confidence:.1f}%</p>
#             <p><a href="/">Sprawdź kolejne zdjęcie</a></p>
#         </body>
#         </html>
#         """)

#     except Exception as error:
#         return HTMLResponse(
#             f"<h1>Nie udało się wykonać klasyfikacji</h1>"
#             f"<p>{html.escape(str(error))}</p>"
#             f"<p><a href='/'>Wróć</a></p>",
#             status_code=500,
#         )


import html
import io
import os

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from google.cloud import storage
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torch import nn


BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
MODEL_OBJECT = "rose-leaf/models/rose_leaf_resnet18.pth"

app = FastAPI()
model = None
class_names = None

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Pobiera wytrenowany model z GCS i przygotowuje go do predykcji."""
    global model, class_names

    if model is not None:
        return model, class_names

    if not BUCKET_NAME:
        raise RuntimeError("Brak zmiennej GCS_BUCKET_NAME.")

    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_OBJECT)

    if not blob.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu gs://{BUCKET_NAME}/{MODEL_OBJECT}"
        )

    checkpoint_bytes = blob.download_as_bytes()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="pl">
    <head>
        <meta charset="utf-8">
        <title>Klasyfikator liści</title>
    </head>
    <body>
        <h1>Klasyfikator liści</h1>
        <p>
            Prześlij zdjęcie liścia. Model przypisze je do jednej z czterech
            klas ze zbioru: Healthy Leaf, Black Spot, Leaf Hole lub Dry Leaf.
        </p>
        <p>
            Aplikacja klasyfikuje liść według tych etykiet. Nie rozpoznaje
            gatunku rośliny.
        </p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Sprawdź zdjęcie</button>
        </form>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return HTMLResponse(
            "<h1>Wybierz plik graficzny.</h1><p><a href='/'>Wróć</a></p>",
            status_code=400,
        )

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return HTMLResponse(
            "<h1>Nie udało się odczytać zdjęcia.</h1><p><a href='/'>Wróć</a></p>",
            status_code=400,
        )

    try:
        classifier, labels = load_model()
        image_tensor = image_transform(image).unsqueeze(0)

        with torch.no_grad():
            scores = classifier(image_tensor)
            probabilities = torch.softmax(scores, dim=1)[0]
            predicted_index = int(probabilities.argmax())

        predicted_class = labels[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100
        safe_filename = html.escape(file.filename or "zdjęcie")

        return HTMLResponse(f"""
        <!doctype html>
        <html lang="pl">
        <head>
            <meta charset="utf-8">
            <title>Wynik klasyfikacji liścia</title>
        </head>
        <body>
            <h1>Wynik klasyfikacji liścia</h1>
            <p>Plik: {safe_filename}</p>
            <p>
                Klasa przypisana przez model:
                <strong>{html.escape(predicted_class)}</strong>
            </p>
            <p>Pewność modelu: {confidence:.1f}%</p>
            <p><a href="/">Sprawdź kolejne zdjęcie</a></p>
        </body>
        </html>
        """)

    except Exception as error:
        return HTMLResponse(
            f"<h1>Nie udało się wykonać klasyfikacji</h1>"
            f"<p>{html.escape(str(error))}</p>"
            f"<p><a href='/'>Wróć</a></p>",
            status_code=500,
        )
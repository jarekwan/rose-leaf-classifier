# Rose Leaf Classifier

A FastAPI application that classifies rose leaf images into four labels:
Healthy Leaf, Black Spot, Leaf Hole, and Dry Leaf.

The classifier uses a pretrained ResNet18 model with transfer learning.
The training job is defined in `train-job.yaml`; the prediction API is in `app.py`.

## Files

- `app.py` — FastAPI application for image predictions.
- `train.py` — model training code.
- `train-job.yaml` — Kubernetes Job configuration for training.
- `Dockerfile` — container image configuration.
- `requirements.txt` — Python dependencies.

The model predicts one of the four dataset labels. It does not identify the plant species.

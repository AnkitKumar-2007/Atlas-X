import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

from dataset import MammogramDataset, eval_transform
from models import create_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\OncoLens")

TEST_CSV = BASE_DIR / "processed" / "stratified_splits" / "test.csv"

MODEL_NAME = "convnext_tiny"

MODEL_PATH = BASE_DIR / "models" / f"{MODEL_NAME}_best.pth"

OUTPUT_DIR = BASE_DIR / "processed" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(test_df)}")
print(f"Unique patients: {test_df['patient_id'].nunique()}")

test_dataset = MammogramDataset(
    csv_file=TEST_CSV,
    transform=eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
print(f"Checkpoint: {MODEL_PATH}")
print(f"Device: {DEVICE}")

if not MODEL_PATH.exists():
    print("\nERROR: Model checkpoint not found!")
    sys.exit()

model = create_model(
    MODEL_NAME,
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle either a raw state_dict or a checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully!")


# ============================================================
# RUN TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST EVALUATION")
print("=" * 70)

all_labels = []
all_probs = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs).squeeze(1)

        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_predictions = np.array(all_predictions)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nBreakdown:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")


# ============================================================
# SAVE METRICS
# ============================================================

results = pd.DataFrame([{
    "model": MODEL_NAME,
    "test_images": len(test_df),
    "accuracy": accuracy,
    "precision": precision,
    "recall_sensitivity": recall,
    "specificity": specificity,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "true_positives": tp
}])

results_path = OUTPUT_DIR / f"{MODEL_NAME}_test_results.csv"

results.to_csv(
    results_path,
    index=False
)

print(f"\nResults saved to:")
print(results_path)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Non-malignant",
        "Malignant"
    ]
)

display.plot()

plt.title(
    f"Confusion Matrix - {MODEL_NAME}"
)

plt.tight_layout()

cm_path = OUTPUT_DIR / f"{MODEL_NAME}_confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

print(f"\nConfusion matrix saved to:")
print(cm_path)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probs
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"ROC Curve - {MODEL_NAME}"
)

plt.legend()

plt.tight_layout()

roc_path = OUTPUT_DIR / f"{MODEL_NAME}_roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()

print(f"\nROC curve saved to:")
print(roc_path)


print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETE!")
print("=" * 70)
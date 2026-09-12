from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import eval_transform
from models import create_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "convnext_tiny"

MODEL_PATH = Path(
    r"D:\OncoLens\models\convnext_tiny_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256

# ============================================================
# TEST CSV
# ============================================================

# IMPORTANT:
# Change this filename only if your test CSV has a different name.
TEST_CSV = Path(
    r"D:\OncoLens\processed\splits\test.csv"
)

# Which image to use from the test dataset?
IMAGE_INDEX = 0

OUTPUT_DIR = Path(
    r"D:\OncoLens\processed\gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output
    ):
        self.activations = output

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    def generate(self, image_tensor):

        self.model.zero_grad()

        output = self.model(image_tensor)

        # Binary classifier score
        score = output.squeeze()

        score.backward()

        # Average gradients spatially
        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted activations
        cam = torch.sum(
            weights * self.activations,
            dim=1,
            keepdim=True
        )

        # Keep positive evidence
        cam = F.relu(cam)

        # Resize to model input size
        cam = F.interpolate(
            cam,
            size=(
                image_tensor.shape[2],
                image_tensor.shape[3]
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # Normalize
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return (
            cam
            .detach()
            .cpu()
            .numpy()
        )

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

if not TEST_CSV.exists():

    print(f"\nERROR: Test CSV not found:")
    print(TEST_CSV)

    print("\nUpdate TEST_CSV in this script.")
    sys.exit()

df = pd.read_csv(TEST_CSV)

print(f"Test CSV: {TEST_CSV}")
print(f"Total images: {len(df)}")

# Check required column
if "image_path_local" not in df.columns:

    print("\nERROR:")
    print("Column 'image_path_local' not found.")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    sys.exit()

if IMAGE_INDEX >= len(df):

    print("\nERROR:")
    print(
        f"IMAGE_INDEX {IMAGE_INDEX} "
        f"is outside the dataset."
    )

    sys.exit()

row = df.iloc[IMAGE_INDEX]

IMAGE_PATH = Path(
    row["image_path_local"]
)

print()
print(f"Selected image index: {IMAGE_INDEX}")
print(f"Image path: {IMAGE_PATH}")

if "pathology" in df.columns:
    print(
        f"Actual label: "
        f"{row['pathology']}"
    )

if not IMAGE_PATH.exists():

    print("\nERROR: Image file does not exist:")
    print(IMAGE_PATH)

    sys.exit()


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING CONVNEXT-TINY")
print("=" * 70)

model = create_model(
    MODEL_NAME
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Support both checkpoint formats
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print(f"Device: {DEVICE}")


# ============================================================
# SELECT TARGET LAYER
# ============================================================

print()
print("=" * 70)
print("SETTING UP GRAD-CAM")
print("=" * 70)

# ConvNeXt-Tiny feature extractor
target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)

print("Grad-CAM initialized successfully!")


# ============================================================
# LOAD IMAGE
# ============================================================

print()
print("=" * 70)
print("LOADING MAMMOGRAM")
print("=" * 70)

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = eval_transform(
    original_image
)

image_tensor = (
    image_tensor
    .unsqueeze(0)
    .to(DEVICE)
)

print(f"Original size: {original_image.size}")
print(f"Model input shape: {image_tensor.shape}")


# ============================================================
# CLASSIFICATION
# ============================================================

print()
print("=" * 70)
print("RUNNING CLASSIFICATION")
print("=" * 70)

with torch.no_grad():

    output = model(
        image_tensor
    )

    probability = torch.sigmoid(
        output
    ).item()

prediction = (
    "MALIGNANT"
    if probability >= 0.5
    else "NON-MALIGNANT"
)

print(f"Prediction: {prediction}")
print(
    f"Malignancy probability: "
    f"{probability:.4f}"
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print()
print("=" * 70)
print("GENERATING GRAD-CAM")
print("=" * 70)

heatmap = gradcam.generate(
    image_tensor
)

print("Grad-CAM generated successfully!")


# ============================================================
# PREPARE IMAGE FOR VISUALIZATION
# ============================================================

original_cv = cv2.imread(
    str(IMAGE_PATH),
    cv2.IMREAD_COLOR
)

if original_cv is None:

    print("\nERROR: OpenCV could not read image.")
    gradcam.remove_hooks()
    sys.exit()

original_cv = cv2.cvtColor(
    original_cv,
    cv2.COLOR_BGR2RGB
)

# The Grad-CAM coordinates correspond to 256 x 256
original_resized = cv2.resize(
    original_cv,
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ============================================================
# CREATE HEATMAP VISUALIZATION
# ============================================================

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

overlay = cv2.addWeighted(
    original_resized,
    0.60,
    heatmap_color,
    0.40,
    0
)


# ============================================================
# EXTRACT ACTIVATION REGION
# ============================================================

print()
print("=" * 70)
print("EXTRACTING ROI")
print("=" * 70)

# Keep strong Grad-CAM activations
THRESHOLD = 0.60

binary_mask = (
    heatmap >= THRESHOLD
).astype(np.uint8) * 255

# Clean small noisy regions
kernel = np.ones(
    (5, 5),
    np.uint8
)

binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    kernel
)

binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# FIND ROI BOUNDING BOX
# ============================================================

contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

roi_box = None

if len(contours) == 0:

    print("No strong activation region found.")

else:

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # Add a little padding around the suspicious area
    PADDING = 10

    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)

    x2 = min(
        IMAGE_SIZE - 1,
        x + w + PADDING
    )

    y2 = min(
        IMAGE_SIZE - 1,
        y + h + PADDING
    )

    roi_box = (
        x1,
        y1,
        x2,
        y2
    )

    print("ROI Bounding Box:")
    print(f"x1 = {x1}")
    print(f"y1 = {y1}")
    print(f"x2 = {x2}")
    print(f"y2 = {y2}")


# ============================================================
# DRAW ROI
# ============================================================

roi_image = original_resized.copy()

if roi_box is not None:

    x1, y1, x2, y2 = roi_box

    cv2.rectangle(
        roi_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

base_name = IMAGE_PATH.stem

original_output = (
    OUTPUT_DIR /
    f"{base_name}_original.png"
)

heatmap_output = (
    OUTPUT_DIR /
    f"{base_name}_heatmap.png"
)

overlay_output = (
    OUTPUT_DIR /
    f"{base_name}_overlay.png"
)

mask_output = (
    OUTPUT_DIR /
    f"{base_name}_activation_mask.png"
)

roi_output = (
    OUTPUT_DIR /
    f"{base_name}_roi.png"
)

# Save images
Image.fromarray(
    original_resized
).save(original_output)

Image.fromarray(
    heatmap_color
).save(heatmap_output)

Image.fromarray(
    overlay
).save(overlay_output)

cv2.imwrite(
    str(mask_output),
    binary_mask
)

Image.fromarray(
    roi_image
).save(roi_output)


# ============================================================
# SAVE RESULTS INFORMATION
# ============================================================

results_output = (
    OUTPUT_DIR /
    f"{base_name}_results.txt"
)

with open(
    results_output,
    "w"
) as f:

    f.write(
        f"Image: {IMAGE_PATH}\n"
    )

    if "pathology" in df.columns:

        f.write(
            f"Actual pathology: "
            f"{row['pathology']}\n"
        )

    f.write(
        f"Prediction: {prediction}\n"
    )

    f.write(
        f"Malignancy probability: "
        f"{probability:.4f}\n"
    )

    if roi_box is not None:

        f.write(
            f"ROI Bounding Box: "
            f"{roi_box}\n"
        )

    else:

        f.write(
            "ROI Bounding Box: None\n"
        )


# ============================================================
# FINISHED
# ============================================================

print(f"\nOriginal:")
print(original_output)

print(f"\nGrad-CAM heatmap:")
print(heatmap_output)

print(f"\nOverlay:")
print(overlay_output)

print(f"\nActivation mask:")
print(mask_output)

print(f"\nROI bounding box:")
print(roi_output)

print(f"\nResults:")
print(results_output)


gradcam.remove_hooks()

print()
print("=" * 70)
print("GRAD-CAM → ROI PIPELINE COMPLETE!")
print("=" * 70)
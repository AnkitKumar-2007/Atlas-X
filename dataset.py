from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageOps
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

LABEL_MAPPING = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}


# ============================================================
# CUSTOM RESIZE + PAD
# ============================================================

class ResizeWithPadding:
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a square of the requested size.
    """

    def __init__(self, size=256, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        width, height = image.size

        # Scale image while preserving aspect ratio
        scale = min(
            self.size / width,
            self.size / height
        )

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Calculate required padding
        pad_left = (self.size - new_width) // 2
        pad_top = (self.size - new_height) // 2

        pad_right = self.size - new_width - pad_left
        pad_bottom = self.size - new_height - pad_top

        image = ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ),
            fill=self.fill
        )

        return image


# ============================================================
# TRAINING TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    # Ensure grayscale input
    transforms.Grayscale(num_output_channels=3),

    # Resize while preserving aspect ratio
    ResizeWithPadding(
        size=IMAGE_SIZE,
        fill=0
    ),

    # Mild augmentation
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    # Convert to PyTorch tensor
    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORMS
# ============================================================

eval_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    ResizeWithPadding(
        size=IMAGE_SIZE,
        fill=0
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class MammogramDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        # Create binary labels
        self.df["label"] = self.df["pathology"].map(
            LABEL_MAPPING
        )

        # Safety check
        if self.df["label"].isna().any():

            unknown_labels = (
                self.df.loc[
                    self.df["label"].isna(),
                    "pathology"
                ]
                .unique()
            )

            raise ValueError(
                f"Unknown pathology labels found: "
                f"{unknown_labels}"
            )

        # Convert labels to integer
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(
            row["image_path_local"]
        )

        # Load image
        image = Image.open(image_path)

        # Ensure image is loaded into memory
        image.load()

        # Apply preprocessing
        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


# ============================================================
# DATALOADER HELPER
# ============================================================

def create_dataloader(
    csv_file,
    transform,
    batch_size=16,
    shuffle=False,
    num_workers=0
):

    dataset = MammogramDataset(
        csv_file=csv_file,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, loader
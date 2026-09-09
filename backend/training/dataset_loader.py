"""
dataset_loader.py — Multi-dataset loader for RetinaScan AI training.
Supports: EyePACS, APTOS 2019, IDRiD, DRIVE, CHASE_DB1, STARE, DIARETDB1, ORIGA.
All datasets expected under data/train/ as documented in AGENT.md Section 6.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision.transforms as T

DATA_ROOT = Path("data/train")


# ── Base dataset ──────────────────────────────────────────────────────────────

class FundusDataset(Dataset):
    """Base class for fundus image datasets."""

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform
        self.samples = []   # list of (image_path, label) tuples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── EyePACS ───────────────────────────────────────────────────────────────────

class EyePACSDataset(FundusDataset):
    """
    PRIMARY GRADING DATASET.
    Source: https://www.kaggle.com/c/diabetic-retinopathy-detection
    Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    Size: ~88,702 images, ~35GB
    """

    def __init__(self, transform=None, split: str = "train"):
        super().__init__(transform)
        root = DATA_ROOT / "eyepacs"
        df = pd.read_csv(root / "trainLabels.csv")
        cutoff = int(len(df) * 0.8)
        df = df[:cutoff] if split == "train" else df[cutoff:]

        self.samples = [
            (str(root / "images" / f"{row['image']}.jpeg"), int(row["level"]))
            for _, row in df.iterrows()
        ]


# ── APTOS 2019 ────────────────────────────────────────────────────────────────

class APTOSDataset(FundusDataset):
    """
    INDIAN CLINICAL DATASET ★ (Aravind Eye Hospital)
    Source: https://www.kaggle.com/c/aptos2019-blindness-detection
    License: CC BY 4.0 | Size: 3,662 images, ~6GB
    MOST RELEVANT for Indian PHC deployment.
    """

    def __init__(self, transform=None):
        super().__init__(transform)
        root = DATA_ROOT / "aptos2019"
        df = pd.read_csv(root / "train.csv")
        self.samples = [
            (str(root / "train_images" / f"{row['id_code']}.png"), int(row["diagnosis"]))
            for _, row in df.iterrows()
        ]


# ── IDRiD ─────────────────────────────────────────────────────────────────────

class IDRiDDataset(FundusDataset):
    """
    INDIAN DIABETIC RETINOPATHY IMAGE DATASET ★★ (MOST IMPORTANT for India)
    Source: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    License: CC BY 4.0 | Size: 516 grading images + 81 pixel-level segmentation images
    Indian patients, Indian cameras, Indian clinical conditions.
    """

    def __init__(self, transform=None, split: str = "train"):
        super().__init__(transform)
        sub = "a. Training Set" if split == "train" else "b. Testing Set"
        csv_name = (
            "a. IDRiD_Disease Grading_Training Labels.csv" if split == "train"
            else "b. IDRiD_Disease Grading_Testing Labels.csv"
        )
        root = DATA_ROOT / "idrid" / "B. Disease Grading" / sub
        df = pd.read_csv(root / csv_name)
        self.samples = [
            (str(root / f"{row['Image name']}.jpg"), int(row["Retinopathy grade"]))
            for _, row in df.iterrows()
        ]


# ── Vessel segmentation datasets ──────────────────────────────────────────────

class DRIVEDataset(Dataset):
    """
    VESSEL SEGMENTATION — DRIVE Dataset
    Source: https://drive.grand-challenge.org/
    License: CC BY 4.0 | Size: 40 images (20 train, 20 test)
    """

    def __init__(self, transform=None, split: str = "training"):
        self.root = DATA_ROOT / "drive" / split
        self.transform = transform
        self.image_files = sorted((self.root / "images").glob("*.tif"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.root / "1st_manual" / (img_path.stem.replace("_training", "_manual1") + ".gif")
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L") if mask_path.exists() else Image.new("L", image.size)
        if self.transform:
            image = self.transform(image)
        import torchvision.transforms.functional as TF
        mask = TF.to_tensor(mask)
        return image, mask


# ── Combined training loader ───────────────────────────────────────────────────

def get_combined_train_loader(batch_size: int = 32) -> DataLoader:
    """Combines all grading datasets for training (EyePACS + APTOS + IDRiD)."""
    transform = T.Compose([
        T.Resize((380, 380)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = [
        EyePACSDataset(transform=transform, split="train"),
        APTOSDataset(transform=transform),
        IDRiDDataset(transform=transform, split="train"),
    ]

    combined = ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

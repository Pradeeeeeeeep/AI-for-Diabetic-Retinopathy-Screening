"""
augmentation.py — Augmentation pipeline for RetinaScan AI training.
Implements domain-specific augmentations for fundus imagery.
"""

import numpy as np
import cv2
import random
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class FundusAugmentation:
    """
    Fundus-specific augmentation pipeline.

    Standard augmentations:
    - Horizontal / vertical flip (retina is symmetric)
    - Random rotation ±10° (camera angle variation)
    - Colour jitter (camera/lighting variation)
    - Random crop + resize

    Advanced augmentations (for robustness):
    - Simulated blur (motion blur, out-of-focus)
    - Brightness perturbation (field conditions)
    - Circular masking (remove border artefacts)
    - MixUp between same-grade images
    """

    def __init__(self, image_size: int = 380, mode: str = "train"):
        self.image_size = image_size
        self.mode = mode

    def get_train_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.image_size + 20, self.image_size + 20)),
            T.RandomCrop((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_val_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def simulate_low_quality(image_bgr: np.ndarray, level: int = 1) -> np.ndarray:
        """
        Simulate poor capture conditions (used for quality assessor training).
        Level 1 = mild degradation, Level 3 = severe.
        """
        result = image_bgr.copy()

        # Blur
        if level >= 1:
            ksize = random.choice([3, 5, 7]) * level
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        # Brightness perturbation
        if level >= 2:
            factor = random.uniform(0.3, 0.6) if random.random() > 0.5 else random.uniform(1.5, 2.5)
            result = np.clip(result.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Add noise
        if level >= 3:
            noise = np.random.normal(0, 25, result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def apply_circular_mask(image_bgr: np.ndarray) -> np.ndarray:
        """Mask pixels outside retinal circular boundary to black."""
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 5
        cv2.circle(mask, center, radius, 255, -1)
        result = image_bgr.copy()
        result[mask == 0] = 0
        return result

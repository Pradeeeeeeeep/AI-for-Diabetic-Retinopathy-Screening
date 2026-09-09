"""
train_vessel_unet.py — Training script for U-Net vessel segmentation.

Usage:
  python backend/training/train_vessel_unet.py \
    --data_dirs data/train/drive data/train/chase_db1 data/train/stare \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_path backend/models/weights/vessel_unet.pth

Expected: Dice score > 0.82 on DRIVE test set.
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from backend.models.unet import UNet
from backend.training.dataset_loader import DRIVEDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_vessel_unet")


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
    return 1.0 - dice.mean()


def combined_loss(pred, target):
    return dice_loss(pred, target) + nn.BCEWithLogitsLoss()(pred, target)


def train_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Vessel U-Net", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimiser.zero_grad()
        pred = model(images)
        loss = combined_loss(pred, masks)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", default=["data/train/drive"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_path", default="backend/models/weights/vessel_unet.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training U-Net on {device}")

    model = UNet(in_channels=1, out_channels=1).to(device)

    # Use DRIVE as primary dataset
    import torchvision.transforms as T
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    train_ds = DRIVEDataset(transform=transform, split="training")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimiser, device)
        scheduler.step()
        logger.info(f"Epoch {epoch:03d}/{args.epochs} | Loss: {loss:.4f}")

        if epoch % 10 == 0:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            logger.info(f"  → Checkpoint saved to {output_path}")

    logger.info("Vessel U-Net training complete.")


if __name__ == "__main__":
    main()

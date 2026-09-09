"""
train_grader.py — Training script for EfficientNet-B4 DR Grader.

Usage:
  # Full training:
  python backend/training/train_grader.py \
    --datasets eyepacs aptos2019 idrid \
    --epochs 50 --batch_size 32 --lr 1e-4 \
    --output_path backend/models/weights/dr_grader_final.pth

  # Fine-tuning (Indian domain adaptation):
  python backend/training/train_grader.py \
    --mode finetune \
    --checkpoint backend/models/weights/dr_grader_final.pth \
    --datasets aptos2019 idrid \
    --epochs 10 --lr 5e-6 \
    --freeze_backbone_epochs 5 \
    --output_path backend/models/weights/dr_grader_india.pth
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

from backend.modules.dr_grader import DRGrader
from backend.training.dataset_loader import get_combined_train_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_grader")

# ── Class weights (handle class imbalance) ────────────────────────────────────
CLASS_WEIGHTS = torch.tensor([0.5, 1.5, 1.0, 2.0, 2.5])   # No DR is most common


def ordinal_loss(logits, targets, num_classes=5):
    """
    Ordinal cross-entropy: penalises predictions far from the true grade
    more heavily than adjacent-grade errors (clinically appropriate).
    """
    base_loss = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(logits.device))(
        logits, targets
    )
    # Distance penalty
    probs = torch.softmax(logits, dim=1)
    grade_vals = torch.arange(num_classes, dtype=torch.float32, device=logits.device)
    pred_grade = (probs * grade_vals).sum(dim=1)
    true_grade = targets.float()
    ordinal_penalty = ((pred_grade - true_grade) ** 2).mean() * 0.3
    return base_loss + ordinal_penalty


def focal_loss(logits, targets, gamma=2.0):
    """Focal loss to focus on hard examples."""
    ce = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(logits.device), reduction="none")(
        logits, targets
    )
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        # Dummy lesion features (zero) during early training
        lesion_features = torch.zeros(images.size(0), 8, device=device)

        optimiser.zero_grad()
        logits = model(images, lesion_features)
        loss = ordinal_loss(logits, labels) + focal_loss(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "finetune"], default="train")
    parser.add_argument("--datasets", nargs="+", default=["eyepacs", "aptos2019", "idrid"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0)
    parser.add_argument("--ordinal_loss_weight", type=float, default=0.3)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="backend/models/weights/dr_grader_final.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DRGrader(num_classes=5, pretrained=(args.mode == "train"))
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)

    if args.mode == "finetune" and args.freeze_backbone_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for initial fine-tuning epochs")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = get_combined_train_loader(batch_size=args.batch_size)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze_backbone_epochs
        if (args.mode == "finetune" and epoch == args.freeze_backbone_epochs + 1):
            for param in model.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone unfrozen")

        loss, acc = train_one_epoch(model, train_loader, optimiser, device)
        scheduler.step()

        logger.info(f"Epoch {epoch:03d}/{args.epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            logger.info(f"  → Saved best model to {output_path} (acc={acc:.4f})")

    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

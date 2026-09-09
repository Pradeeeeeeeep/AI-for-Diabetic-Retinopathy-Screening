"""
calibration.py — Temperature Scaling for confidence calibration.
After training, run this to find the optimal temperature T such that
model confidence scores are well-calibrated (reliability diagram is diagonal).

Usage:
  python backend/training/calibration.py \
    --model_path backend/models/weights/dr_grader_final.pth \
    --val_data messidor2 \
    --output_temperature backend/models/weights/temperature.json
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from backend.modules.dr_grader import DRGrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibration")


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling (Guo et al., 2017 ICML).
    Learns a single scalar T that divides logits before softmax.
    T > 1 → softer (more uncertain) predictions.
    T < 1 → sharper (overconfident) predictions.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)


def calibrate(model_path: str, val_loader: DataLoader, device: torch.device) -> float:
    """Find optimal temperature T on the validation set."""
    # Load base model
    model = DRGrader(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Collect all logits on val set
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            lesion_features = torch.zeros(images.size(0), 8, device=device)
            logits = model(images, lesion_features)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Learn temperature
    ts = TemperatureScaling()
    optimiser = optim.LBFGS([ts.temperature], lr=0.01, max_iter=500)
    criterion = nn.CrossEntropyLoss()

    def eval_step():
        optimiser.zero_grad()
        loss = criterion(ts(all_logits), all_labels)
        loss.backward()
        return loss

    optimiser.step(eval_step)
    T = float(ts.temperature.item())
    logger.info(f"Optimal temperature: {T:.4f}")
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_data", default="messidor2")
    parser.add_argument("--output_temperature", default="backend/models/weights/temperature.json")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: Replace with real val loader for your chosen dataset
    logger.warning("Using dummy validation loader — replace with real dataset for production calibration.")
    from torch.utils.data import TensorDataset
    dummy_images = torch.randn(64, 3, 380, 380)
    dummy_labels = torch.randint(0, 5, (64,))
    val_loader = DataLoader(TensorDataset(dummy_images, dummy_labels), batch_size=args.batch_size)

    temperature = calibrate(args.model_path, val_loader, device)

    output_path = Path(args.output_temperature)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)

    logger.info(f"Temperature saved to {output_path}")


if __name__ == "__main__":
    main()

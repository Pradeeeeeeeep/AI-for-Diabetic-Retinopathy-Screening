"""
train_ma_svm.py — Training script for Microaneurysm SVM false-positive reducer.

Usage:
  python backend/training/train_ma_svm.py \
    --data_dir data/train/diaretdb1 \
    --output_path backend/models/weights/svm_ma_classifier.pkl

Expected: Sensitivity > 0.78, Specificity > 0.90 on held-out DIARETDB1.
"""

import argparse
import logging
import pickle
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

from backend.modules.lesion_segmentor import MicroaneurysmDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_ma_svm")


def load_diaretdb1(data_dir: str):
    """
    Load DIARETDB1 images and ground-truth MA masks.
    Returns: (features_list, labels_list)
      label=1: true microaneurysm
      label=0: false positive
    """
    import cv2
    data_path = Path(data_dir)
    detector = MicroaneurysmDetector()

    all_features = []
    all_labels = []

    images_dir = data_path / "images"
    gt_dir = data_path / "groundtruths" / "redsmalldots"

    for img_file in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        green = img[:, :, 1]
        gt_file = gt_dir / img_file.name
        gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE) if gt_file.exists() else None

        candidates = detector.detect(green)
        for c in candidates:
            x, y, r = c["x"], c["y"], c["radius"]
            # Check if candidate overlaps with ground truth
            if gt_mask is not None:
                region = gt_mask[
                    max(0, y - r):min(gt_mask.shape[0], y + r),
                    max(0, x - r):min(gt_mask.shape[1], x + r),
                ]
                is_true_ma = int(region.mean() > 10)
            else:
                is_true_ma = 0

            all_features.append(c["features"])
            all_labels.append(is_true_ma)

    return np.array(all_features), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/train/diaretdb1")
    parser.add_argument("--output_path", default="backend/models/weights/svm_ma_classifier.pkl")
    args = parser.parse_args()

    logger.info(f"Loading DIARETDB1 from {args.data_dir}")
    X, y = load_diaretdb1(args.data_dir)
    logger.info(f"Dataset: {len(X)} samples, {y.sum()} true MAs, {len(y) - y.sum()} FPs")

    # SVM pipeline with standard scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True)),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    logger.info(f"5-fold CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train on full dataset
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    logger.info("\n" + classification_report(y, preds, target_names=["FP", "True MA"]))

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"SVM classifier saved to {output_path}")


if __name__ == "__main__":
    main()

"""
quality_assessor.py — Module 1: Image Quality Assessment & Enhancement
Scores fundus images on sharpness, illumination, and field-of-view.
Applies CLAHE enhancement for borderline-quality images.
"""

import cv2
import numpy as np
from typing import Optional


class ImageQualityAssessor:
    """
    Scores fundus images on 3 axes and applies enhancement.
    Returns: (quality_score, enhanced_image, feedback_message)

    Scoring axes:
      • Sharpness    — Laplacian variance (detects blur)
      • Illumination — Green channel mean pixel value
      • Field of view — Hough circle detection for retinal boundary
    """

    THRESHOLDS = {
        "sharpness_min": 100.0,     # Laplacian variance
        "brightness_min": 40,        # Mean pixel value
        "brightness_max": 220,
        "accept_threshold": 0.70,    # Composite score → ACCEPTED
        "borderline_threshold": 0.40,# Composite score → ENHANCED
    }

    def assess(self, image_bgr: np.ndarray) -> dict:
        """
        Full quality assessment pipeline.

        Args:
            image_bgr: OpenCV BGR image (H x W x 3, uint8)

        Returns:
            dict with keys: status, composite_score, sharpness,
                            illumination, field_of_view, enhanced_image, feedback
        """
        if image_bgr is None or image_bgr.size == 0:
            return self._rejected("No image data received.")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # ── 1. Sharpness: Laplacian variance ──────────────────────────────────
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = float(min(lap_var / 500.0, 1.0))

        # ── 2. Illumination: mean pixel in green channel ───────────────────────
        green = image_bgr[:, :, 1]
        mean_brightness = float(green.mean())
        if 40 <= mean_brightness <= 220:
            illumination_score = 1.0
        else:
            dist = min(abs(mean_brightness - 40), abs(mean_brightness - 220))
            illumination_score = float(max(0.0, 1.0 - dist / 100.0))

        # ── 3. Field of View: detect retinal circular boundary ─────────────────
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30,
            minRadius=int(min(gray.shape) * 0.3),
            maxRadius=int(min(gray.shape) * 0.6),
        )
        fov_score = 1.0 if circles is not None else 0.5

        # ── 4. Composite (weighted sum) ────────────────────────────────────────
        composite = (
            0.4 * sharpness_score +
            0.4 * illumination_score +
            0.2 * fov_score
        )

        # ── 5. Decision ────────────────────────────────────────────────────────
        if composite >= self.THRESHOLDS["accept_threshold"]:
            status = "ACCEPTED"
            enhanced = self._enhance(image_bgr)
            feedback = "Image quality: Good. Analysis proceeding."
        elif composite >= self.THRESHOLDS["borderline_threshold"]:
            status = "ENHANCED"
            enhanced = self._enhance(image_bgr)
            feedback = (
                "Image enhanced (borderline quality). "
                "Results may be less reliable — consider recapture."
            )
        else:
            return self._rejected(
                "Image too poor to grade. "
                "Please recapture with better focus and lighting."
            )

        return {
            "status": status,
            "composite_score": round(composite, 3),
            "sharpness": round(sharpness_score, 3),
            "illumination": round(illumination_score, 3),
            "field_of_view": round(fov_score, 3),
            "enhanced_image": enhanced,
            "feedback": feedback,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """CLAHE + illumination normalisation + Gaussian denoising."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_ch)

        # Illumination normalisation via morphological opening (background subtraction)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(l_enhanced, cv2.MORPH_OPEN, kernel)
        l_normalised = cv2.subtract(l_enhanced, bg)
        l_normalised = cv2.normalize(l_normalised, None, 0, 255, cv2.NORM_MINMAX)

        merged = cv2.merge([l_normalised, a_ch, b_ch])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Gaussian denoising
        result = cv2.GaussianBlur(result, (3, 3), 0)
        return result

    def _rejected(self, feedback: str) -> dict:
        return {
            "status": "REJECTED",
            "composite_score": 0.0,
            "sharpness": 0.0,
            "illumination": 0.0,
            "field_of_view": 0.0,
            "enhanced_image": None,
            "feedback": feedback,
        }

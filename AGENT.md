# AGENT.md — Explainable AI for Diabetic Retinopathy Screening
## Web Application — Complete Agent Specification

---

## 1. PROJECT OVERVIEW

| Field | Value |
|---|---|
| **System Name** | RetinaScan AI — Explainable DR Screening Platform |
| **Target Deployment** | Web Application (browser-based, mobile-responsive) |
| **Primary Users** | Ophthalmologists, Optometrists, Rural PHC Staff |
| **Patient Scale** | 100,000+ patients/year per district |
| **Clinical Target** | Sensitivity >90%, Specificity >85% for referable DR (Level 2+) |
| **Tech Stack** | Python (FastAPI) + React/Next.js frontend + PyTorch backend |
| **Data Location** | `data/train/` (see Section 6 for full dataset manifest) |

---

## 2. SYSTEM ARCHITECTURE — HOW IT DETECTS DR

### 2.1 End-to-End Detection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEB APPLICATION                          │
│                                                                 │
│  [Upload Fundus Image]                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │  MODULE 1:       │  • Sharpness check (Laplacian variance)  │
│  │  Image Quality   │  • Illumination check (mean pixel)       │
│  │  Assessment      │  • Field-of-view check (circle detect)   │
│  └────────┬────────┘  • CLAHE enhancement if borderline        │
│           │                                                     │
│     PASS / ENHANCE / REJECT                                     │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  MODULE 2:       │  • Vessel segmentation (U-Net)           │
│  │  Lesion          │  • Microaneurysm detection (LoG+SVM)     │
│  │  Segmentation    │  • Exudate segmentation                  │
│  └────────┬────────┘  • Hemorrhage classification              │
│           │            • Optic disc localization               │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  MODULE 3:       │  • EfficientNet-B4 backbone              │
│  │  DR Grading      │  • 5-class ICDR (0–4)                   │
│  │  Engine          │  • Calibrated confidence score           │
│  └────────┬────────┘  • Ordinal regression loss               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  MODULE 4:       │  • Grad-CAM heatmap overlay              │
│  │  Explainability  │  • Lesion-to-ICDR criterion mapping     │
│  └────────┬────────┘  • PDF report generation                 │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  DASHBOARD       │  • Results display                       │
│  │  (Clinician UI)  │  • Review queue management              │
│  └─────────────────┘  • Referral workflow                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. MODULE SPECIFICATIONS

### MODULE 1: Image Quality Assessment & Enhancement

**Purpose:** Automatically score and enhance fundus images before analysis.

**Detection Logic:**

```python
# File: backend/modules/quality_assessor.py

import cv2
import numpy as np

class ImageQualityAssessor:
    """
    Scores fundus images on 3 axes and applies enhancement.
    Returns: (quality_score, enhanced_image, feedback_message)
    """

    THRESHOLDS = {
        "sharpness_min": 100.0,    # Laplacian variance
        "brightness_min": 40,       # Mean pixel value
        "brightness_max": 220,
        "accept_threshold": 0.70,   # Composite score
        "borderline_threshold": 0.40,
    }

    def assess(self, image_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- Sharpness: Laplacian variance ---
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(lap_var / 500.0, 1.0)

        # --- Illumination: mean pixel in green channel ---
        green = image_bgr[:, :, 1]
        mean_brightness = green.mean()
        if 40 <= mean_brightness <= 220:
            illumination_score = 1.0
        else:
            dist = min(abs(mean_brightness - 40), abs(mean_brightness - 220))
            illumination_score = max(0, 1 - dist / 100)

        # --- Field of View: detect retinal circular boundary ---
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30,
                                    minRadius=int(min(gray.shape)*0.3),
                                    maxRadius=int(min(gray.shape)*0.6))
        fov_score = 1.0 if circles is not None else 0.5

        # --- Composite score (weighted) ---
        composite = (0.4 * sharpness_score +
                     0.4 * illumination_score +
                     0.2 * fov_score)

        # --- Decision ---
        if composite >= self.THRESHOLDS["accept_threshold"]:
            status = "ACCEPTED"
            enhanced = self._enhance(image_bgr)
            feedback = "Image quality: Good"
        elif composite >= self.THRESHOLDS["borderline_threshold"]:
            status = "ENHANCED"
            enhanced = self._enhance(image_bgr)
            feedback = "Image enhanced (borderline quality). Results may be less reliable."
        else:
            status = "REJECTED"
            enhanced = None
            feedback = "Image too poor to grade. Please recapture with better focus/lighting."

        return {
            "status": status,
            "composite_score": round(composite, 3),
            "sharpness": round(sharpness_score, 3),
            "illumination": round(illumination_score, 3),
            "field_of_view": round(fov_score, 3),
            "enhanced_image": enhanced,
            "feedback": feedback,
        }

    def _enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """CLAHE + illumination normalization + denoising."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Illumination normalization via morphological top-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(l_enhanced, cv2.MORPH_OPEN, kernel)
        l_normalized = cv2.subtract(l_enhanced, bg)
        l_normalized = cv2.normalize(l_normalized, None, 0, 255, cv2.NORM_MINMAX)

        merged = cv2.merge([l_normalized, a, b])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Gaussian denoising
        result = cv2.GaussianBlur(result, (3, 3), 0)
        return result
```

---

### MODULE 2: Retinal Lesion Segmentation

**Architecture:** U-Net for pixel-level segmentation; custom LoG filter for microaneurysms.

```python
# File: backend/modules/lesion_segmentor.py

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from skimage import measure, feature

class MicroaneurysmDetector:
    """
    Multi-scale Laplacian of Gaussian (LoG) blob detection
    for sub-pixel microaneurysm candidates, followed by
    SVM false-positive reduction.
    
    WHY LoG: MAs are 10–100 micron dark spots (~5–15 pixels at
    standard fundus resolution). LoG matched filters at σ=1–4px
    respond maximally to circular blobs at the right scale.
    """

    SIGMA_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    THRESHOLD = 0.015  # LoG response threshold

    def detect(self, green_channel: np.ndarray) -> list[dict]:
        """Returns list of candidate microaneurysm bounding boxes."""
        # Invert (MAs are dark, LoG detects bright blobs after invert)
        inverted = 255 - green_channel
        normalized = inverted.astype(np.float32) / 255.0

        blobs = feature.blob_log(
            normalized,
            min_sigma=min(self.SIGMA_RANGE),
            max_sigma=max(self.SIGMA_RANGE),
            num_sigma=len(self.SIGMA_RANGE),
            threshold=self.THRESHOLD,
            overlap=0.5,
        )

        candidates = []
        for blob in blobs:
            y, x, sigma = blob
            radius = int(sigma * np.sqrt(2))
            candidates.append({
                "x": int(x), "y": int(y),
                "radius": radius,
                "features": self._extract_features(green_channel, int(x), int(y), radius)
            })

        return candidates  # Pass to SVM classifier for FP reduction

    def _extract_features(self, img, x, y, r):
        """12 features for SVM false-positive reduction."""
        h, w = img.shape
        x1, y1 = max(0, x-r*2), max(0, y-r*2)
        x2, y2 = min(w, x+r*2), min(h, y+r*2)
        patch = img[y1:y2, x1:x2].astype(np.float32)

        if patch.size == 0:
            return [0.0] * 12

        center_mean = img[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)].mean()
        bg_mean = patch.mean()

        return [
            center_mean,                        # 1. Center intensity
            bg_mean,                            # 2. Background intensity
            bg_mean - center_mean,              # 3. Contrast
            patch.std(),                        # 4. Texture variance
            float(r),                           # 5. Radius
            float(r ** 2 * np.pi),             # 6. Area
            center_mean / (bg_mean + 1e-5),     # 7. Contrast ratio
            patch.min(),                        # 8. Min pixel
            patch.max(),                        # 9. Max pixel
            np.median(patch),                   # 10. Median
            patch.max() - patch.min(),          # 11. Dynamic range
            float(np.sum(patch < center_mean)), # 12. Dark pixel count
        ]


class VesselSegmentor:
    """
    U-Net for retinal vessel segmentation.
    Trained on: DRIVE + CHASE_DB1 + STARE datasets.
    Input: Green channel (512x512)
    Output: Binary vessel mask
    """

    def __init__(self, model_path: str):
        self.model = self._build_unet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def _build_unet(self) -> nn.Module:
        """Standard U-Net with skip connections (4 encoder/decoder levels)."""
        # See: backend/models/unet.py for full implementation
        pass

    def segment(self, green_channel: np.ndarray) -> np.ndarray:
        """Returns binary vessel mask (0=background, 1=vessel)."""
        # Preprocessing
        img = cv2.resize(green_channel, (512, 512))
        img_tensor = torch.FloatTensor(img / 255.0).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = torch.sigmoid(self.model(img_tensor))
        
        mask = (pred.squeeze().numpy() > 0.5).astype(np.uint8)
        return cv2.resize(mask, green_channel.shape[::-1],
                          interpolation=cv2.INTER_NEAREST)


class ExudateSegmentor:
    """
    Hard exudates: bright, waxy deposits near leaking vessels.
    Method: Thresholding + morphological operations after OD masking.
    Soft exudates (cotton-wool spots): texture + grey-level features.
    """

    def segment_hard_exudates(self, image_bgr, od_mask):
        green = image_bgr[:, :, 1]
        # Remove optic disc (which is also bright)
        masked = green.copy()
        masked[od_mask > 0] = 0
        
        # Hard exudates are bright outliers
        threshold = green.mean() + 2.5 * green.std()
        binary = (masked > threshold).astype(np.uint8)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cleaned

    def segment_hemorrhages(self, image_bgr, vessel_mask):
        """Dark irregular blobs outside vessel map = hemorrhages."""
        red = image_bgr[:, :, 2]
        dark_mask = (red < red.mean() - red.std()).astype(np.uint8)
        # Exclude vessels
        non_vessel = cv2.bitwise_and(dark_mask,
                                      cv2.bitwise_not(vessel_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(non_vessel, cv2.MORPH_OPEN, kernel)
```

---

### MODULE 3: DR Severity Grading Engine

**Model:** EfficientNet-B4 with late fusion of segmentation feature maps.

```python
# File: backend/modules/dr_grader.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class DRGrader(nn.Module):
    """
    EfficientNet-B4 backbone + lesion feature fusion.
    
    ICDR Grading Scale:
      Level 0 — No DR
      Level 1 — Mild NPDR (microaneurysms only)
      Level 2 — Moderate NPDR (referable threshold)
      Level 3 — Severe NPDR (>20 hemorrhages per quadrant,
                              venous beading, IRMA)
      Level 4 — Proliferative DR (neovascularization, vitreous
                                   hemorrhage, tractional detachment)
    
    Training: EyePACS (88,702) + APTOS2019 (3,662) + Messidor-2 (1,748)
    Fine-tuning: Indian Retinal DB subset (domain adaptation)
    Loss: Ordinal cross-entropy + focal loss (class imbalance)
    """

    DR_CLASSES = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR"
    }

    REFERRAL_THRESHOLD = 2  # Level 2+ = refer to ophthalmologist

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        
        # Backbone
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove default head

        # Lesion feature branch (from segmentation counts)
        self.lesion_fc = nn.Sequential(
            nn.Linear(8, 64),   # 8 lesion features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, image: torch.Tensor,
                lesion_features: torch.Tensor) -> torch.Tensor:
        img_features = self.backbone(image)
        lesion_out = self.lesion_fc(lesion_features)
        combined = torch.cat([img_features, lesion_out], dim=1)
        return self.classifier(combined)

    def predict_with_confidence(self, image: torch.Tensor,
                                 lesion_features: torch.Tensor) -> dict:
        """
        Returns calibrated prediction with confidence score.
        Uses temperature scaling for calibration.
        """
        self.eval()
        TEMPERATURE = 1.5  # Learned during calibration phase

        with torch.no_grad():
            logits = self.forward(image, lesion_features)
            calibrated_logits = logits / TEMPERATURE
            probs = F.softmax(calibrated_logits, dim=1).squeeze()

        predicted_class = probs.argmax().item()
        confidence = probs[predicted_class].item()
        is_referable = predicted_class >= self.REFERRAL_THRESHOLD

        return {
            "grade": predicted_class,
            "grade_label": self.DR_CLASSES[predicted_class],
            "confidence": round(confidence * 100, 1),
            "is_referable": is_referable,
            "all_probs": {self.DR_CLASSES[i]: round(p.item() * 100, 1)
                          for i, p in enumerate(probs)},
            "low_confidence_flag": confidence < 0.65,
            "recommendation": "REFER TO OPHTHALMOLOGIST" if is_referable
                              else "ROUTINE FOLLOW-UP",
        }


class LesionFeatureExtractor:
    """
    Converts segmentation outputs → 8-dimensional feature vector
    for the grading model's lesion branch.
    """

    def extract(self, segmentation_results: dict) -> np.ndarray:
        ma = segmentation_results.get("microaneurysms", [])
        exudates = segmentation_results.get("hard_exudates")
        hemorrhages = segmentation_results.get("hemorrhages")
        vessels = segmentation_results.get("vessels")
        
        he_mask = hemorrhages if hemorrhages is not None else np.zeros((1,1))
        ex_mask = exudates if exudates is not None else np.zeros((1,1))
        v_mask = vessels if vessels is not None else np.zeros((1,1))

        features = np.array([
            float(len(ma)),                          # 1. MA count
            float(np.sum(ex_mask > 0)) / 1000.0,    # 2. Exudate area (norm)
            float(np.sum(he_mask > 0)) / 1000.0,    # 3. Hemorrhage area
            float(np.sum(v_mask > 0)) / 10000.0,    # 4. Vessel density
            float(len(ma) > 0),                      # 5. MA present (binary)
            float(np.sum(ex_mask > 0) > 50),         # 6. Exudates present
            float(np.sum(he_mask > 0) > 50),         # 7. Hemorrhages present
            self._count_quadrant_hemorrhages(he_mask), # 8. Quadrant HE count
        ], dtype=np.float32)

        return features

    def _count_quadrant_hemorrhages(self, hemo_mask: np.ndarray) -> float:
        """ICDR criterion: >20 hemorrhages in any quadrant → Level 3."""
        h, w = hemo_mask.shape
        quadrants = [
            hemo_mask[:h//2, :w//2],   # Top-left
            hemo_mask[:h//2, w//2:],   # Top-right
            hemo_mask[h//2:, :w//2],   # Bottom-left
            hemo_mask[h//2:, w//2:],   # Bottom-right
        ]
        counts = []
        for q in quadrants:
            labeled, n = ndimage.label(q)
            counts.append(float(n))
        return max(counts) if counts else 0.0
```

---

### MODULE 4: Explainability Engine

```python
# File: backend/modules/explainability.py

import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

class GradCAMGenerator:
    """
    Gradient-weighted Class Activation Mapping.
    
    HOW IT WORKS:
    1. Run forward pass → get class prediction
    2. Backprop gradient of target class score to final conv layer
    3. Global-average-pool the gradients → importance weights
    4. Weighted sum of feature maps → coarse attention map
    5. ReLU → upsample to input resolution
    6. Overlay as heatmap on fundus image
    
    Result: Clinician sees WHICH regions drove the AI decision.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks on target conv layer
        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor,
                 target_class: int) -> np.ndarray:
        """Returns heatmap as numpy array (H x W), values 0–1."""
        self.model.eval()
        output = self.model(image_tensor, torch.zeros(1, 8))  # dummy lesion feat
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().numpy()
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def overlay_on_image(self, original_image_bgr: np.ndarray,
                          cam: np.ndarray) -> np.ndarray:
        """Returns BGR image with heatmap overlay."""
        h, w = original_image_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(original_image_bgr, 0.6, heatmap, 0.4, 0)
        return overlay


class LesionEvidenceMapper:
    """
    Maps detected lesions to ICDR criteria.
    Provides clinical justification text for each finding.
    """

    ICDR_CRITERIA = {
        0: {
            "label": "No DR",
            "criteria": "No abnormalities detected",
            "color": "#22c55e",
        },
        1: {
            "label": "Mild NPDR",
            "criteria": "Microaneurysms only",
            "color": "#84cc16",
        },
        2: {
            "label": "Moderate NPDR",
            "criteria": "More than microaneurysms but less than severe NPDR",
            "color": "#f59e0b",
        },
        3: {
            "label": "Severe NPDR",
            "criteria": ">20 intraretinal hemorrhages in each quadrant; "
                        "definite venous beading; prominent IRMA",
            "color": "#f97316",
        },
        4: {
            "label": "Proliferative DR",
            "criteria": "Neovascularization, vitreous/pre-retinal hemorrhage",
            "color": "#ef4444",
        },
    }

    def generate_evidence_report(self, grade: int,
                                  segmentation_results: dict) -> dict:
        ma_count = len(segmentation_results.get("microaneurysms", []))
        has_exudates = segmentation_results.get("hard_exudates_present", False)
        has_hemorrhages = segmentation_results.get("hemorrhages_present", False)
        max_quadrant_he = segmentation_results.get("max_quadrant_hemorrhages", 0)
        has_neovasc = segmentation_results.get("neovascularization", False)

        findings = []

        if ma_count > 0:
            findings.append({
                "lesion": "Microaneurysm",
                "count": ma_count,
                "supports": "Level ≥1 DR",
                "icdr_criterion": "Presence of microaneurysms",
                "severity": "mild",
            })

        if has_exudates:
            findings.append({
                "lesion": "Hard Exudates",
                "count": "Present",
                "supports": "Level ≥2 DR",
                "icdr_criterion": "Lipid deposits indicating vascular leakage",
                "severity": "moderate",
            })

        if has_hemorrhages:
            findings.append({
                "lesion": "Intraretinal Hemorrhages",
                "count": f"Max {int(max_quadrant_he)} in worst quadrant",
                "supports": f"Level {'≥3' if max_quadrant_he > 20 else '≥2'} DR",
                "icdr_criterion": "Intraretinal hemorrhages",
                "severity": "severe" if max_quadrant_he > 20 else "moderate",
            })

        if has_neovasc:
            findings.append({
                "lesion": "Neovascularization",
                "count": "Detected",
                "supports": "Level 4 (Proliferative DR)",
                "icdr_criterion": "New vessel formation on disc or elsewhere",
                "severity": "critical",
            })

        return {
            "grade": grade,
            "grade_info": self.ICDR_CRITERIA[grade],
            "findings": findings,
            "finding_count": len(findings),
            "referral_urgency": self._get_urgency(grade),
        }

    def _get_urgency(self, grade: int) -> str:
        urgency_map = {
            0: "No referral needed — annual screening",
            1: "No referral needed — 6-month follow-up",
            2: "Refer within 3 months",
            3: "Refer within 2 weeks",
            4: "URGENT — refer within 24–48 hours",
        }
        return urgency_map[grade]


class ReportGenerator:
    """Generates PDF annotated report for ophthalmologist review."""

    def generate_pdf(self, patient_id: str, grade_result: dict,
                     evidence: dict, heatmap_image_bgr: np.ndarray,
                     original_image_bgr: np.ndarray) -> bytes:
        """Returns PDF as bytes, to be sent to frontend for download."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Header
        story.append(Paragraph(
            f"<b>DR Screening Report — Patient ID: {patient_id}</b>",
            styles["Title"]
        ))

        # Grade result
        story.append(Paragraph(
            f"<b>DR Grade:</b> {grade_result['grade_label']} "
            f"(Level {grade_result['grade']})",
            styles["Heading2"]
        ))
        story.append(Paragraph(
            f"<b>Confidence:</b> {grade_result['confidence']}%",
            styles["Normal"]
        ))
        story.append(Paragraph(
            f"<b>Recommendation:</b> {grade_result['recommendation']}",
            styles["Normal"]
        ))
        story.append(Paragraph(
            f"<b>Referral Urgency:</b> {evidence['referral_urgency']}",
            styles["Normal"]
        ))

        # Findings table
        story.append(Paragraph("<b>Clinical Findings:</b>", styles["Heading3"]))
        for finding in evidence["findings"]:
            story.append(Paragraph(
                f"• <b>{finding['lesion']}</b> ({finding['count']}): "
                f"{finding['supports']} — {finding['icdr_criterion']}",
                styles["Normal"]
            ))

        # Add heatmap image
        heatmap_pil = Image.fromarray(
            cv2.cvtColor(heatmap_image_bgr, cv2.COLOR_BGR2RGB)
        )
        img_buffer = io.BytesIO()
        heatmap_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        story.append(Paragraph("<b>Grad-CAM Attention Map:</b>",
                               styles["Heading3"]))
        story.append(RLImage(img_buffer, width=400, height=300))

        doc.build(story)
        return buffer.getvalue()
```

---

## 4. WEBAPP STRUCTURE

```
retinascan-app/
├── AGENT.md                          ← THIS FILE
├── README.md
│
├── backend/
│   ├── main.py                       ← FastAPI app entry point
│   ├── requirements.txt
│   ├── config.py                     ← Paths, thresholds, model configs
│   │
│   ├── modules/
│   │   ├── quality_assessor.py       ← Module 1
│   │   ├── lesion_segmentor.py       ← Module 2
│   │   ├── dr_grader.py              ← Module 3
│   │   └── explainability.py         ← Module 4
│   │
│   ├── models/
│   │   ├── unet.py                   ← U-Net architecture
│   │   ├── efficientnet_grader.py    ← DR grader model
│   │   └── weights/                  ← Trained .pth files
│   │       ├── vessel_unet.pth
│   │       ├── dr_grader_final.pth
│   │       └── svm_ma_classifier.pkl
│   │
│   ├── training/
│   │   ├── train_grader.py           ← Training script
│   │   ├── train_vessel_unet.py
│   │   ├── train_ma_svm.py
│   │   ├── dataset_loader.py         ← Multi-dataset loader
│   │   ├── augmentation.py           ← Augmentation pipeline
│   │   └── calibration.py            ← Temperature scaling
│   │
│   └── api/
│       ├── routes.py                 ← API endpoints
│       └── schemas.py                ← Pydantic models
│
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   └── src/
│       ├── app/
│       │   ├── page.tsx              ← Landing / upload page
│       │   ├── dashboard/page.tsx    ← Clinician dashboard
│       │   └── report/[id]/page.tsx  ← Detailed report view
│       ├── components/
│       │   ├── ImageUploader.tsx
│       │   ├── GradCAMViewer.tsx
│       │   ├── LesionEvidenceCard.tsx
│       │   ├── GradeGauge.tsx
│       │   ├── ConfidenceBar.tsx
│       │   └── ReviewQueue.tsx
│       └── lib/
│           └── api.ts                ← Frontend API client
│
└── data/
    └── train/                        ← ALL TRAINING DATA (see Section 6)
```

---

## 5. API ENDPOINTS

```
POST   /api/analyze            ← Main analysis endpoint
GET    /api/report/{id}        ← Get saved report
GET    /api/report/{id}/pdf    ← Download PDF report
GET    /api/queue              ← Get review queue
POST   /api/queue/{id}/review  ← Submit ophthalmologist review
GET    /api/stats              ← Dashboard statistics
```

### Main Analysis Endpoint

```python
# backend/api/routes.py

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2

router = APIRouter()

@router.post("/api/analyze")
async def analyze_fundus_image(
    file: UploadFile = File(...),
    patient_id: str = "unknown",
    eye: str = "right"  # "right" or "left"
):
    """
    Full pipeline: Quality → Segmentation → Grading → Explainability.
    Returns JSON with all results + base64 heatmap.
    Typical processing time: 8–12 seconds on GPU, 25–40s on CPU.
    """
    # Read image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Module 1: Quality
    quality_result = quality_assessor.assess(image_bgr)
    if quality_result["status"] == "REJECTED":
        return JSONResponse(status_code=422, content={
            "error": "image_rejected",
            "feedback": quality_result["feedback"]
        })

    working_image = quality_result["enhanced_image"]

    # Module 2: Segmentation
    green = working_image[:, :, 1]
    ma_candidates = ma_detector.detect(green)
    vessel_mask = vessel_segmentor.segment(green)
    hard_exudates = exudate_segmentor.segment_hard_exudates(
        working_image, optic_disc_mask=np.zeros_like(green)
    )
    hemorrhages = exudate_segmentor.segment_hemorrhages(
        working_image, vessel_mask
    )

    seg_results = {
        "microaneurysms": ma_candidates,
        "hard_exudates_present": hard_exudates.sum() > 50,
        "hemorrhages_present": hemorrhages.sum() > 50,
        "max_quadrant_hemorrhages": lesion_extractor._count_quadrant_hemorrhages(hemorrhages),
        "neovascularization": False,  # Advanced check in full system
    }

    # Module 3: Grading
    lesion_features = lesion_extractor.extract(seg_results)
    image_tensor = preprocess_for_model(working_image)
    lesion_tensor = torch.FloatTensor(lesion_features).unsqueeze(0)
    grade_result = dr_grader.predict_with_confidence(image_tensor, lesion_tensor)

    # Module 4: Explainability
    cam = gradcam.generate(image_tensor, grade_result["grade"])
    heatmap_overlay = gradcam.overlay_on_image(working_image, cam)
    evidence = evidence_mapper.generate_evidence_report(
        grade_result["grade"], seg_results
    )

    # Encode heatmap for frontend
    _, buffer = cv2.imencode('.jpg', heatmap_overlay)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

    # Save to DB
    report_id = save_report(patient_id, grade_result, evidence, heatmap_b64)

    return {
        "report_id": report_id,
        "quality": quality_result,
        "grade": grade_result,
        "evidence": evidence,
        "heatmap_b64": heatmap_b64,
        "lesion_counts": {
            "microaneurysms": len(ma_candidates),
            "hard_exudates": "present" if seg_results["hard_exudates_present"] else "absent",
            "hemorrhages": "present" if seg_results["hemorrhages_present"] else "absent",
        },
        "processing_notes": (
            "⚠️ Low confidence — mandatory human review"
            if grade_result["low_confidence_flag"] else None
        )
    }
```

---

## 6. TRAINING DATA — `data/train/` DIRECTORY

All datasets live under `data/train/`. The loader reads from this path automatically.

```
data/
└── train/
    │
    ├── eyepacs/                         ← PRIMARY GRADING DATASET
    │   ├── images/                      ← 88,702 fundus images (.jpeg)
    │   │   ├── 10_left.jpeg
    │   │   ├── 10_right.jpeg
    │   │   └── ...
    │   ├── trainLabels.csv              ← patient,level (0–4)
    │   └── README.txt
    │   Source: https://www.kaggle.com/c/diabetic-retinopathy-detection
    │   License: Kaggle competition data (non-commercial research)
    │   Size: ~88K images, ~35GB
    │   Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    │
    ├── aptos2019/                       ← INDIAN CLINICAL DATASET ★
    │   ├── train_images/                ← 3,662 fundus images (.png)
    │   │   ├── 000c1434d8d7.png
    │   │   └── ...
    │   ├── train.csv                    ← id_code,diagnosis (0–4)
    │   └── README.txt
    │   Source: https://www.kaggle.com/c/aptos2019-blindness-detection
    │   License: CC BY 4.0
    │   Size: 3,662 images, ~6GB
    │   Note: Captured at Aravind Eye Hospital, India — MOST RELEVANT
    │          for Indian PHC deployment (Remidio / Forus cameras)
    │
    ├── messidor2/                       ← BENCHMARK DATASET
    │   ├── images/                      ← 1,748 fundus images (.tiff)
    │   │   ├── Messidor-2/
    │   │   └── ...
    │   ├── messidor_data.csv            ← image_id,adjudicated_grade,adjudicated_dme
    │   └── README.txt
    │   Source: https://www.adcis.net/en/third-party/messidor2/
    │   License: Research use (requires registration)
    │   Size: 1,748 images, ~4GB
    │   Note: Used primarily for benchmarking; high-quality images
    │
    ├── drive/                           ← VESSEL SEGMENTATION
    │   ├── training/
    │   │   ├── images/                  ← 20 fundus images (.tif)
    │   │   └── 1st_manual/             ← 20 vessel masks (.gif)
    │   ├── test/
    │   │   ├── images/
    │   │   └── 1st_manual/
    │   └── README.txt
    │   Source: https://drive.grand-challenge.org/
    │   License: CC BY 4.0
    │   Size: 40 images (20 train, 20 test)
    │   Use: Train vessel segmentation U-Net
    │
    ├── chase_db1/                       ← VESSEL SEGMENTATION (supplement)
    │   ├── images/                      ← 28 fundus images (.jpg)
    │   ├── 1stAnnotation/              ← Vessel masks (expert 1)
    │   ├── 2ndAnnotation/              ← Vessel masks (expert 2)
    │   └── README.txt
    │   Source: https://researchdata.kingston.ac.uk/96/
    │   License: CC BY 4.0
    │   Size: 28 images
    │
    ├── stare/                           ← VESSEL SEGMENTATION (supplement)
    │   ├── images/                      ← 20 images (.ppm)
    │   ├── labels-vk/                  ← Vessel labels (Valentina Kouznetsova)
    │   ├── labels-ah/                  ← Vessel labels (Adam Hoover)
    │   └── README.txt
    │   Source: http://cecas.clemson.edu/~ahoover/stare/
    │   License: Research use
    │   Size: 20 images
    │
    ├── diaretdb1/                       ← MICROANEURYSM TRAINING
    │   ├── images/                      ← 89 fundus images (.png)
    │   ├── groundtruths/
    │   │   ├── hardexudates/           ← Pixel-level masks
    │   │   ├── softexudates/
    │   │   ├── hemorrhages/
    │   │   └── redsmalldots/           ← Microaneurysms + small hemorrhages
    │   └── README.txt
    │   Source: https://www.it.lut.fi/project/imageret/diaretdb1/
    │   License: Research use
    │   Size: 89 images with pixel-level lesion annotations
    │   Use: Train microaneurysm detector + lesion segmentors
    │
    ├── idrid/                           ← LESION SEGMENTATION (Indian)
    │   ├── A. Segmentation/
    │   │   ├── 1. Original Images/
    │   │   │   ├── a. Training Set/    ← 54 images (.jpg)
    │   │   │   └── b. Testing Set/     ← 27 images
    │   │   └── 2. All Segmentation Groundtruths/
    │   │       ├── a. Training Set/
    │   │       │   ├── 1. Microaneurysms/
    │   │       │   ├── 2. Haemorrhages/
    │   │       │   ├── 3. Hard Exudates/
    │   │       │   ├── 4. Soft Exudates/
    │   │       │   └── 5. Optic Disc/
    │   │       └── b. Testing Set/
    │   ├── B. Disease Grading/
    │   │   ├── a. Training Set/        ← 413 images with ICDR labels
    │   │   └── b. Testing Set/         ← 103 images
    │   └── README.txt
    │   Source: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    │   License: CC BY 4.0
    │   Size: 516 grading images + 81 pixel-level segmentation images
    │   Note: ★★ MOST IMPORTANT for Indian deployment — Indian patients,
    │              Indian cameras, Indian clinical conditions
    │
    ├── origa/                           ← OPTIC DISC LOCALIZATION
    │   ├── images/                      ← 650 fundus images
    │   ├── masks/                       ← OD/cup segmentation masks
    │   └── README.txt
    │   Source: http://www.seri.com.sg/
    │   License: Research use
    │   Size: 650 images
    │   Use: Train optic disc localizer
    │
    └── synthetic/                       ← AUGMENTED / SYNTHETIC DATA
        ├── low_quality_simulated/       ← Deliberately degraded images
        │   ├── blur_level1/             ← Simulated motion blur
        │   ├── blur_level2/
        │   ├── underexposed/            ← Dark images (rural conditions)
        │   ├── overexposed/
        │   └── artifacts/               ← Dust, lens flare simulation
        └── generation_script.py         ← Script that creates these from
                                            EyePACS/APTOS source images
        Note: Generated by applying random degradation transforms to
              clean images. Essential for training the quality assessor
              and robustness under field conditions.
```

### Dataset Summary Table

| Dataset | Images | Labels | Use Case | Indian? |
|---|---|---|---|---|
| EyePACS | 88,702 | DR grade 0–4 | Primary grader training | No |
| APTOS 2019 | 3,662 | DR grade 0–4 | Fine-tuning (Indian) | ✅ Yes |
| Messidor-2 | 1,748 | DR grade 0–3 | Benchmarking | No |
| IDRiD | 516 | Grade + lesion masks | Lesion + grade | ✅ Yes |
| DRIVE | 40 | Vessel masks | Vessel U-Net | No |
| CHASE_DB1 | 28 | Vessel masks | Vessel U-Net | No |
| STARE | 20 | Vessel masks | Vessel U-Net | No |
| DIARETDB1 | 89 | Lesion pixel masks | MA/exudate detection | No |
| ORIGA | 650 | OD masks | Optic disc localization | No |
| Synthetic | ~5,000 | Inherited | Quality assessor training | N/A |

### Dataset Loader

```python
# backend/training/dataset_loader.py

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision.transforms as T

DATA_ROOT = "data/train"

class EyePACSDataset(Dataset):
    def __init__(self, transform=None, split="train"):
        self.root = os.path.join(DATA_ROOT, "eyepacs")
        df = pd.read_csv(os.path.join(self.root, "trainLabels.csv"))
        # 80/20 split
        cutoff = int(len(df) * 0.8)
        self.df = df[:cutoff] if split == "train" else df[cutoff:]
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, "images", f"{row['image']}.jpeg")
        image = Image.open(img_path).convert("RGB")
        label = int(row["level"])
        if self.transform:
            image = self.transform(image)
        return image, label


class APTOSDataset(Dataset):
    """APTOS 2019 — Indian fundus images from Aravind Eye Hospital."""

    def __init__(self, transform=None):
        self.root = os.path.join(DATA_ROOT, "aptos2019")
        self.df = pd.read_csv(os.path.join(self.root, "train.csv"))
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, "train_images",
                                f"{row['id_code']}.png")
        image = Image.open(img_path).convert("RGB")
        label = int(row["diagnosis"])
        if self.transform:
            image = self.transform(image)
        return image, label


class IDRiDDataset(Dataset):
    """Indian Diabetic Retinopathy Image Dataset — lesion + grade labels."""

    def __init__(self, transform=None, split="train"):
        self.root = os.path.join(DATA_ROOT, "idrid",
                                 "B. Disease Grading",
                                 "a. Training Set" if split == "train"
                                 else "b. Testing Set")
        self.df = pd.read_csv(
            os.path.join(self.root,
                         "a. IDRiD_Disease Grading_Training Labels.csv"
                         if split == "train"
                         else "b. IDRiD_Disease Grading_Testing Labels.csv")
        )
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root,
                                f"{row['Image name']}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = int(row["Retinopathy grade"])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_combined_train_loader(batch_size=32):
    """Combines all grading datasets for training."""
    transform = T.Compose([
        T.Resize((380, 380)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    datasets = [
        EyePACSDataset(transform=transform, split="train"),
        APTOSDataset(transform=transform),
        IDRiDDataset(transform=transform, split="train"),
    ]

    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size,
                      shuffle=True, num_workers=4, pin_memory=True)
```

---

## 7. TRAINING PROCEDURE

### Step 1 — Train Vessel U-Net

```bash
python backend/training/train_vessel_unet.py \
  --data_dirs data/train/drive data/train/chase_db1 data/train/stare \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_path backend/models/weights/vessel_unet.pth
```

Expected: Dice score > 0.82 on DRIVE test set.

### Step 2 — Train Microaneurysm SVM

```bash
python backend/training/train_ma_svm.py \
  --data_dir data/train/diaretdb1 \
  --output_path backend/models/weights/svm_ma_classifier.pkl
```

Expected: Sensitivity > 0.78, specificity > 0.90 on held-out DIARETDB1.

### Step 3 — Train DR Grader (EfficientNet-B4)

```bash
python backend/training/train_grader.py \
  --datasets eyepacs aptos2019 idrid \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --warmup_epochs 5 \
  --focal_loss_gamma 2 \
  --ordinal_loss_weight 0.3 \
  --output_path backend/models/weights/dr_grader_final.pth
```

**Class weights** (to handle imbalance):
```python
CLASS_WEIGHTS = {
    0: 0.5,   # No DR — overrepresented
    1: 1.5,   # Mild — underrepresented
    2: 1.0,   # Moderate
    3: 2.0,   # Severe — rare
    4: 2.5,   # Proliferative — rarest
}
```

### Step 4 — Calibrate Confidence (Temperature Scaling)

```bash
python backend/training/calibration.py \
  --model_path backend/models/weights/dr_grader_final.pth \
  --val_data messidor2 \
  --output_temperature backend/models/weights/temperature.json
```

### Step 5 — Domain Adaptation (Indian Images)

Fine-tune the trained grader on APTOS + IDRiD only:

```bash
python backend/training/train_grader.py \
  --mode finetune \
  --checkpoint backend/models/weights/dr_grader_final.pth \
  --datasets aptos2019 idrid \
  --epochs 10 \
  --lr 5e-6 \
  --freeze_backbone_epochs 5 \
  --output_path backend/models/weights/dr_grader_india.pth
```

---

## 8. FRONTEND — WEBAPP DESIGN

### Design System

```
Typography: Inter (UI) + IBM Plex Mono (confidence scores/data)
Palette:
  --bg:          #0f172a  (deep navy — clinical, serious)
  --surface:     #1e293b  (card background)
  --border:      #334155
  --primary:     #38bdf8  (sky blue — trust, medical)
  --grade-0:     #22c55e  (green — no DR)
  --grade-1:     #84cc16  (lime — mild)
  --grade-2:     #f59e0b  (amber — moderate)
  --grade-3:     #f97316  (orange — severe)
  --grade-4:     #ef4444  (red — proliferative)
  --text:        #f1f5f9
  --muted:       #94a3b8
```

### Page Structure

```
/                          ← Upload page
  [Logo + Header]
  [Drag & drop fundus image upload]
  [Patient ID input]
  [Eye selector: Left / Right]
  [Analyze button]

/result/{report_id}        ← Results page (primary clinician view)
  [Two-panel layout]
  LEFT:
    Original image | Grad-CAM heatmap toggle
    Lesion overlay (MA dots, exudate regions, hemorrhage areas)
    
  RIGHT:
    DR Grade badge (color-coded 0–4)
    Confidence bar (calibrated %)
    ⚠️ Low-confidence flag (if < 65%)
    Recommendation banner (REFER / ROUTINE)
    Referral urgency
    Clinical Findings accordion:
      • Each lesion with count + ICDR criterion
    [Download PDF Report button]
    [Mark as Reviewed by Ophthalmologist]

/dashboard                 ← Admin / review queue
  [Stats: screened today, referable rate, avg confidence]
  [Review Queue Table: pending human review]
  [Filter by grade, date, confidence]
```

### Key React Components

```tsx
// frontend/src/components/GradeGauge.tsx
// Circular gauge showing grade 0–4 with severity color

// frontend/src/components/GradCAMViewer.tsx
// Toggle between original + heatmap with opacity slider

// frontend/src/components/LesionEvidenceCard.tsx
// Expandable card per lesion type mapping to ICDR criteria

// frontend/src/components/ConfidenceBar.tsx
// Horizontal bar + uncertainty flag display

// frontend/src/components/ReviewQueue.tsx
// Table of pending reports requiring ophthalmologist sign-off
```

---

## 9. CLINICAL VALIDATION TARGETS

| Metric | Target | Validated On |
|---|---|---|
| Sensitivity (referable DR, Level 2+) | >90% | IDRiD test + Messidor-2 |
| Specificity (referable DR) | >85% | IDRiD test + Messidor-2 |
| AUC (binary: referable vs not) | >0.95 | Messidor-2 |
| Quadratic Weighted Kappa | >0.80 | EyePACS test |
| Image rejection rate | <5% for gradeable images | Synthetic quality set |
| Report generation time | <12s on GPU | Benchmarked |
| Ophthalmologist review time | <30s per case | UX target |
| Grad-CAM clinical usefulness rating | >4.0/5.0 | Ophthalmologist study |

---

## 10. DEPLOYMENT ARCHITECTURE (WEB)

```
┌─────────────┐     HTTPS      ┌──────────────────┐
│   Browser   │ ←────────────→ │  Next.js Frontend │
│  (Clinician)│                │  (Vercel / nginx) │
└─────────────┘                └────────┬─────────┘
                                        │ REST API
                               ┌────────▼─────────┐
                               │  FastAPI Backend  │
                               │  (Uvicorn + GPU)  │
                               └────────┬─────────┘
                                        │
                    ┌───────────────────┼────────────────────┐
                    │                   │                    │
           ┌────────▼───────┐  ┌───────▼──────┐  ┌─────────▼──────┐
           │  PyTorch Model  │  │  PostgreSQL   │  │  File Storage  │
           │  (GPU inference)│  │  (reports DB) │  │  (images/PDF)  │
           └────────────────┘  └──────────────┘  └────────────────┘
```

### Environment Variables

```env
# .env
MODEL_GRADER_PATH=backend/models/weights/dr_grader_india.pth
MODEL_VESSEL_PATH=backend/models/weights/vessel_unet.pth
MODEL_SVM_PATH=backend/models/weights/svm_ma_classifier.pkl
TEMPERATURE_PATH=backend/models/weights/temperature.json
DATABASE_URL=postgresql://user:pass@localhost/retinascan
STORAGE_BACKEND=local  # or "s3"
STORAGE_PATH=data/outputs/reports
USE_GPU=true
DEVICE=cuda:0
LOG_LEVEL=INFO
```

---

## 11. PERFORMANCE BENCHMARKS (EXPECTED)

| Hardware | Processing Time | Throughput |
|---|---|---|
| NVIDIA T4 GPU | 8–12s per image | ~5 images/min |
| CPU only (8-core) | 35–55s per image | ~1.5 images/min |
| NVIDIA A100 | 3–5s per image | ~15 images/min |

For 100,000 patients/year (district scale):
- = 274 patients/day (both eyes = 548 images/day)
- At 5 images/min: 548 images ÷ 5 = ~110 minutes GPU time/day
- Ophthalmologist reviews (30% referable): 164 cases × 30s = ~82 min/day
- Single part-time ophthalmologist handles telemedicine review remotely

---

## 12. QUICK START

```bash
# 1. Clone and install
git clone https://github.com/your-org/retinascan-ai
cd retinascan-ai

# 2. Backend setup
pip install -r backend/requirements.txt

# 3. Download datasets (run dataset download scripts)
python scripts/download_aptos.py        # Kaggle API key required
python scripts/download_idrid.py        # IEEE DataPort account required
python scripts/generate_synthetic.py    # Generates data/train/synthetic/

# 4. Train models (or download pretrained weights)
python backend/training/train_grader.py --config configs/full_train.yaml

# 5. Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start frontend
cd frontend && npm install && npm run dev

# 7. Open browser
open http://localhost:3000
```

---

## 13. REFERENCES & CITATIONS

1. **ICDR Scale** — Wilkinson CP et al. (2003). Proposed international clinical diabetic retinopathy disease severity scale. *Ophthalmology*, 110(9), 1677–1682.
2. **EfficientNet** — Tan M, Le QV (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
3. **Grad-CAM** — Selvaraju RR et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
4. **Google DR Algorithm** — Gulshan V et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy. *JAMA*, 316(22), 2402–2410.
5. **IDRiD Dataset** — Porwal P et al. (2018). Indian diabetic retinopathy image dataset (IDRiD). *IEEE DataPort*.
6. **APTOS 2019** — Karthik M et al. (2019). APTOS 2019 blindness detection. *Kaggle*.
7. **Temperature Scaling** — Guo C et al. (2017). On calibration of modern neural networks. *ICML*.
8. **Sparse BagNet** — Djoumessi K et al. (2025). An inherently interpretable AI model improves screening speed and accuracy for early DR. *PLOS Digital Health*.
9. **XAI for DR** — Nibhanupudi S et al. (2026). Explainable AI for DR Screening: Enhancing clinician trust. *IEEE CCIC*.
10. **MadhuNetrAI** — RPC, AIIMS New Delhi (2025). India's AI-driven community DR screening programme.

---

*Last updated: 2026 | RetinaScan AI v1.0 | For research and clinical validation use only.*
*This system is intended as a decision-support tool. All referable cases must be reviewed by a licensed ophthalmologist.*

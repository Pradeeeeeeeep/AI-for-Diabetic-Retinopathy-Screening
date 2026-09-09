"""
explainability.py — Module 4: Explainability Engine
- GradCAMGenerator: Gradient-weighted Class Activation Mapping
- LesionEvidenceMapper: Maps lesion detections to ICDR criteria
- ReportGenerator: PDF clinical report generation
"""

import numpy as np
import cv2
import io
import base64
from typing import Optional
from PIL import Image

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ── Grad-CAM Generator ────────────────────────────────────────────────────────

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

    Result: Clinician sees WHICH retinal regions drove the AI decision.
    """

    def __init__(self, model=None, target_layer_name: str = ""):
        self.model = model
        self.gradients: Optional[np.ndarray] = None
        self.activations: Optional[np.ndarray] = None
        self._simulation = model is None or not TORCH_AVAILABLE

        if not self._simulation and target_layer_name:
            try:
                target_layer = dict(model.named_modules())[target_layer_name]
                target_layer.register_forward_hook(self._save_activation)
                target_layer.register_full_backward_hook(self._save_gradient)
            except Exception:
                self._simulation = True

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor,
        target_class: int,
        original_image_bgr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns heatmap as numpy array (H x W), values 0–1."""
        if self._simulation:
            return self._simulate_cam(original_image_bgr, target_class)

        self.model.eval()
        output = self.model(image_tensor, torch.zeros(1, 8))

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def overlay_on_image(
        self,
        original_image_bgr: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Returns BGR image with Jet heatmap overlay."""
        h, w = original_image_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        overlay = cv2.addWeighted(original_image_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def _simulate_cam(
        self,
        image_bgr: Optional[np.ndarray],
        grade: int,
    ) -> np.ndarray:
        """
        Generates a plausible simulated Grad-CAM heatmap.
        Higher grades produce more widespread/intense activations.
        """
        if image_bgr is None:
            cam = np.zeros((224, 224), dtype=np.float32)
        else:
            h, w = image_bgr.shape[:2]
            cam = np.zeros((h, w), dtype=np.float32)

        h, w = cam.shape
        np.random.seed(grade * 42)

        # Number and intensity of activation blobs scale with severity
        n_blobs = max(1, grade * 2 + np.random.randint(1, 4))
        for _ in range(n_blobs):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            radius = np.random.randint(20, max(30, min(h, w) // 5))
            intensity = 0.4 + grade * 0.15 + np.random.uniform(0, 0.2)

            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            blob = np.exp(-(dist ** 2) / (2 * (radius / 2) ** 2)) * intensity
            cam += blob

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


# ── Lesion Evidence Mapper ─────────────────────────────────────────────────────

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
            "criteria": (
                ">20 intraretinal hemorrhages in each quadrant; "
                "definite venous beading; prominent IRMA"
            ),
            "color": "#f97316",
        },
        4: {
            "label": "Proliferative DR",
            "criteria": "Neovascularization, vitreous/pre-retinal hemorrhage",
            "color": "#ef4444",
        },
    }

    URGENCY_MAP = {
        0: "No referral needed — annual screening",
        1: "No referral needed — 6-month follow-up",
        2: "Refer within 3 months",
        3: "Refer within 2 weeks",
        4: "URGENT — refer within 24–48 hours",
    }

    def generate_evidence_report(
        self,
        grade: int,
        segmentation_results: dict,
    ) -> dict:
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
            "referral_urgency": self.URGENCY_MAP[grade],
        }


# ── PDF Report Generator ──────────────────────────────────────────────────────

class ReportGenerator:
    """Generates annotated PDF report for ophthalmologist review."""

    def generate_pdf(
        self,
        patient_id: str,
        eye: str,
        grade_result: dict,
        evidence: dict,
        heatmap_image_bgr: Optional[np.ndarray],
        original_image_bgr: Optional[np.ndarray],
        report_date: str = "",
    ) -> bytes:
        """Returns PDF as bytes to be sent to frontend for download."""
        if not REPORTLAB_AVAILABLE:
            return b""

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        story = []

        # ── Header ─────────────────────────────────────────────────────────────
        story.append(Paragraph(
            "<b>RetinaScan AI — Diabetic Retinopathy Screening Report</b>",
            styles["Title"],
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#334155")))
        story.append(Spacer(1, 0.3 * cm))

        # ── Patient info ───────────────────────────────────────────────────────
        info_data = [
            ["Patient ID:", patient_id],
            ["Eye:", eye.title()],
            ["Date:", report_date or "N/A"],
            ["AI Version:", "RetinaScan AI v1.0 (Simulation Mode)"],
        ]
        info_table = Table(info_data, colWidths=[4 * cm, 12 * cm])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5 * cm))

        # ── Grade result ───────────────────────────────────────────────────────
        grade_color = evidence["grade_info"]["color"]
        story.append(Paragraph(
            f"<b>DR Grade:</b> "
            f"<font color='{grade_color}'><b>{grade_result['grade_label']} "
            f"(Level {grade_result['grade']})</b></font>",
            styles["Heading2"],
        ))
        story.append(Paragraph(
            f"<b>Confidence:</b> {grade_result['confidence']}%"
            + (" ⚠️ Low confidence — mandatory human review" if grade_result.get('low_confidence_flag') else ""),
            styles["Normal"],
        ))
        story.append(Paragraph(
            f"<b>Recommendation:</b> {grade_result['recommendation']}",
            styles["Normal"],
        ))
        story.append(Paragraph(
            f"<b>Referral Urgency:</b> {evidence['referral_urgency']}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(f"<b>ICDR Criterion:</b> {evidence['grade_info']['criteria']}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

        # ── Clinical findings ──────────────────────────────────────────────────
        story.append(Paragraph("<b>Clinical Findings:</b>", styles["Heading3"]))
        if evidence["findings"]:
            for f in evidence["findings"]:
                story.append(Paragraph(
                    f"• <b>{f['lesion']}</b> ({f['count']}): "
                    f"{f['supports']} — {f['icdr_criterion']}",
                    styles["Normal"],
                ))
        else:
            story.append(Paragraph("• No significant lesions detected.", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

        # ── Probability breakdown ──────────────────────────────────────────────
        story.append(Paragraph("<b>Grade Probability Breakdown:</b>", styles["Heading3"]))
        prob_data = [["DR Level", "Probability (%)"]] + [
            [label, f"{prob:.1f}%"] for label, prob in grade_result.get("all_probs", {}).items()
        ]
        prob_table = Table(prob_data, colWidths=[8 * cm, 4 * cm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.5 * cm))

        # ── Images ─────────────────────────────────────────────────────────────
        def bgr_to_rl_image(img_bgr: np.ndarray, label: str, width=8*cm, height=6*cm):
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            elements = [
                Paragraph(f"<b>{label}</b>", styles["Heading3"]),
                RLImage(buf, width=width, height=height),
            ]
            return elements

        if original_image_bgr is not None and heatmap_image_bgr is not None:
            story.append(Paragraph("<b>Image Analysis:</b>", styles["Heading2"]))
            story += bgr_to_rl_image(original_image_bgr, "Original Fundus Image")
            story += bgr_to_rl_image(heatmap_image_bgr, "Grad-CAM Attention Map")

        # ── Disclaimer ─────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8")))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "<i>This report is generated by an AI decision-support system. "
            "All referable cases MUST be reviewed by a licensed ophthalmologist "
            "before clinical action. RetinaScan AI v1.0 | Research use only.</i>",
            styles["Normal"],
        ))

        doc.build(story)
        return buffer.getvalue()

"""
routes.py — FastAPI route handlers for RetinaScan AI.

Endpoints:
  POST /api/analyze             Full image analysis pipeline
  GET  /api/report/{id}         Get saved report JSON
  GET  /api/report/{id}/pdf     Download annotated PDF
  GET  /api/queue               Review queue (pending human review)
  POST /api/queue/{id}/review   Submit ophthalmologist review
  GET  /api/stats               Dashboard statistics
"""

import os
import cv2
import base64
import uuid
import json
import numpy as np
from datetime import datetime, date
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse

from backend.api.schemas import (
    AnalysisResponse, ReportSummary, ReviewRequest, DashboardStats,
    ErrorResponse,
)
from backend.modules.quality_assessor import ImageQualityAssessor
from backend.modules.lesion_segmentor import (
    MicroaneurysmDetector, VesselSegmentor, ExudateSegmentor,
)
from backend.modules.dr_grader import LesionFeatureExtractor, load_grader
from backend.modules.explainability import (
    GradCAMGenerator, LesionEvidenceMapper, ReportGenerator,
)
import backend.config as cfg

router = APIRouter()

# ── Module singletons (initialised once at startup) ───────────────────────────
_quality_assessor = ImageQualityAssessor()
_ma_detector = MicroaneurysmDetector()
_vessel_segmentor = VesselSegmentor(
    model_path=cfg.MODEL_VESSEL_PATH if not cfg.SIMULATION_MODE else None
)
_exudate_segmentor = ExudateSegmentor()
_lesion_extractor = LesionFeatureExtractor()
_grader = load_grader(cfg.MODEL_GRADER_PATH if not cfg.SIMULATION_MODE else None)
_gradcam = GradCAMGenerator()   # Simulation mode by default
_evidence_mapper = LesionEvidenceMapper()
_report_gen = ReportGenerator()

# ── In-memory report store (replace with DB in production) ────────────────────
_report_store: dict = {}


# ══ POST /api/analyze ══════════════════════════════════════════════════════════

@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_fundus_image(
    file: UploadFile = File(...),
    patient_id: str = Form(default="unknown"),
    eye: str = Form(default="right"),
):
    """
    Full analysis pipeline:
      Image Quality → Lesion Segmentation → DR Grading → Explainability

    Returns JSON with results + base64 Grad-CAM heatmap.
    Typical processing time: 8–12s on GPU, 25–40s on CPU.
    """
    # ── Read image ────────────────────────────────────────────────────────────
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_image", "feedback": "Could not decode image. Please upload a valid JPEG/PNG fundus image."},
        )

    # ── Module 1: Quality assessment ──────────────────────────────────────────
    quality_result = _quality_assessor.assess(image_bgr)

    if quality_result["status"] == "REJECTED":
        return JSONResponse(
            status_code=422,
            content={
                "error": "image_rejected",
                "feedback": quality_result["feedback"],
            },
        )

    working_image = quality_result.pop("enhanced_image")

    # ── Module 2: Lesion segmentation ─────────────────────────────────────────
    green = working_image[:, :, 1]

    ma_candidates = _ma_detector.detect(green)
    vessel_mask = _vessel_segmentor.segment(green)
    hard_exudates = _exudate_segmentor.segment_hard_exudates(working_image)
    hemorrhages = _exudate_segmentor.segment_hemorrhages(working_image, vessel_mask)

    seg_results = {
        "microaneurysms": ma_candidates,
        "hard_exudates": hard_exudates,
        "hemorrhages": hemorrhages,
        "vessels": vessel_mask,
        "hard_exudates_present": bool(int(hard_exudates.sum()) > 50),
        "hemorrhages_present": bool(int(hemorrhages.sum()) > 50),
        "max_quadrant_hemorrhages": float(
            _lesion_extractor._count_quadrant_hemorrhages(hemorrhages)
        ),
        "neovascularization": False,   # Advanced check in full system
    }

    # ── Module 3: Grading ─────────────────────────────────────────────────────
    lesion_features = _lesion_extractor.extract(seg_results)
    grade_result = _grader.predict_with_confidence(working_image, lesion_features)

    # ── Module 4: Explainability ──────────────────────────────────────────────
    cam = _gradcam.generate(None, grade_result["grade"], working_image)
    heatmap_overlay = _gradcam.overlay_on_image(working_image, cam)
    evidence = _evidence_mapper.generate_evidence_report(
        grade_result["grade"], seg_results
    )

    # Encode heatmap for frontend
    _, buffer = cv2.imencode(".jpg", heatmap_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    # Also encode original for PDF
    _, orig_buffer = cv2.imencode(".jpg", working_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    orig_b64 = base64.b64encode(orig_buffer).decode("utf-8")

    # ── Persist report ────────────────────────────────────────────────────────
    report_id = str(uuid.uuid4())[:8].upper()
    report_entry = {
        "report_id": report_id,
        "patient_id": patient_id,
        "eye": eye,
        "grade": grade_result,
        "evidence": evidence,
        "quality": quality_result,
        "heatmap_b64": heatmap_b64,
        "orig_b64": orig_b64,
        "lesion_counts": {
            "microaneurysms": len(ma_candidates),
            "hard_exudates": "present" if seg_results["hard_exudates_present"] else "absent",
            "hemorrhages": "present" if seg_results["hemorrhages_present"] else "absent",
        },
        "created_at": datetime.now().isoformat(),
        "reviewed": False,
        "reviewer_notes": None,
        "simulation_mode": cfg.SIMULATION_MODE,
    }
    _report_store[report_id] = report_entry

    return {
        "report_id": report_id,
        "patient_id": patient_id,
        "eye": eye,
        "quality": quality_result,
        "grade": grade_result,
        "evidence": evidence,
        "heatmap_b64": heatmap_b64,
        "lesion_counts": report_entry["lesion_counts"],
        "processing_notes": (
            "⚠️ Low confidence — mandatory human review"
            if grade_result["low_confidence_flag"] else None
        ),
        "simulation_mode": cfg.SIMULATION_MODE,
    }


# ══ GET /api/report/{id} ════════════════════════════════════════════════════════

@router.get("/api/report/{report_id}")
async def get_report(report_id: str):
    """Get full saved report by ID."""
    entry = _report_store.get(report_id.upper())
    if not entry:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    # Return without raw base64 blobs (use dedicated endpoints)
    return {k: v for k, v in entry.items() if k not in ("orig_b64",)}


# ══ GET /api/report/{id}/pdf ════════════════════════════════════════════════════

@router.get("/api/report/{report_id}/pdf")
async def download_pdf_report(report_id: str):
    """Generate and download annotated PDF report."""
    entry = _report_store.get(report_id.upper())
    if not entry:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    # Decode stored images
    heatmap_bgr = _b64_to_bgr(entry.get("heatmap_b64", ""))
    orig_bgr = _b64_to_bgr(entry.get("orig_b64", ""))

    pdf_bytes = _report_gen.generate_pdf(
        patient_id=entry["patient_id"],
        eye=entry["eye"],
        grade_result=entry["grade"],
        evidence=entry["evidence"],
        heatmap_image_bgr=heatmap_bgr,
        original_image_bgr=orig_bgr,
        report_date=entry.get("created_at", "")[:10],
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="retinascan_{report_id}.pdf"'
        },
    )


# ══ GET /api/queue ══════════════════════════════════════════════════════════════

@router.get("/api/queue")
async def get_review_queue(
    grade_filter: Optional[int] = None,
    min_confidence: Optional[float] = None,
):
    """Return reports pending ophthalmologist review, newest first."""
    queue = [
        entry for entry in _report_store.values()
        if not entry["reviewed"] and entry["grade"]["is_referable"]
    ]

    if grade_filter is not None:
        queue = [e for e in queue if e["grade"]["grade"] == grade_filter]
    if min_confidence is not None:
        queue = [e for e in queue if e["grade"]["confidence"] >= min_confidence]

    # Sort newest first
    queue.sort(key=lambda e: e["created_at"], reverse=True)

    return {
        "total": len(queue),
        "items": [_entry_to_summary(e) for e in queue],
    }


# ══ POST /api/queue/{id}/review ════════════════════════════════════════════════

@router.post("/api/queue/{report_id}/review")
async def submit_review(report_id: str, review: ReviewRequest):
    """Submit ophthalmologist review for a report."""
    entry = _report_store.get(report_id.upper())
    if not entry:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    entry["reviewed"] = True
    entry["reviewer_id"] = review.reviewer_id
    entry["reviewer_notes"] = review.notes
    entry["reviewed_at"] = datetime.now().isoformat()

    if review.override_grade is not None:
        entry["grade"]["grade_override"] = review.override_grade

    return {"status": "reviewed", "report_id": report_id}


# ══ GET /api/stats ══════════════════════════════════════════════════════════════

@router.get("/api/stats", response_model=DashboardStats)
async def get_stats():
    """Dashboard statistics."""
    all_reports = list(_report_store.values())
    today = date.today().isoformat()

    screened_today = sum(
        1 for e in all_reports if e["created_at"][:10] == today
    )
    referable = [e for e in all_reports if e["grade"]["is_referable"]]
    referable_rate = (
        round(len(referable) / len(all_reports) * 100, 1)
        if all_reports else 0.0
    )
    avg_conf = (
        round(
            sum(e["grade"]["confidence"] for e in all_reports) / len(all_reports), 1
        )
        if all_reports else 0.0
    )
    pending = sum(
        1 for e in all_reports
        if not e["reviewed"] and e["grade"]["is_referable"]
    )

    grade_dist: dict = {str(i): 0 for i in range(5)}
    for e in all_reports:
        grade_dist[str(e["grade"]["grade"])] += 1

    return DashboardStats(
        screened_today=screened_today,
        screened_total=len(all_reports),
        referable_rate=referable_rate,
        avg_confidence=avg_conf,
        pending_review=pending,
        grade_distribution=grade_dist,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entry_to_summary(entry: dict) -> dict:
    return {
        "report_id": entry["report_id"],
        "patient_id": entry["patient_id"],
        "eye": entry["eye"],
        "grade": entry["grade"]["grade"],
        "grade_label": entry["grade"]["grade_label"],
        "confidence": entry["grade"]["confidence"],
        "is_referable": entry["grade"]["is_referable"],
        "recommendation": entry["grade"]["recommendation"],
        "referral_urgency": entry["evidence"]["referral_urgency"],
        "created_at": entry["created_at"],
        "reviewed": entry["reviewed"],
        "reviewer_notes": entry.get("reviewer_notes"),
    }


def _b64_to_bgr(b64_str: str) -> Optional[np.ndarray]:
    if not b64_str:
        return None
    try:
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

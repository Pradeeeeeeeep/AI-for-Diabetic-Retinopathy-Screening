"""
schemas.py — Pydantic request/response models for RetinaScan AI API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class EyeChoice(str, Enum):
    left = "left"
    right = "right"


# ── Quality Assessment ─────────────────────────────────────────────────────────

class QualityResult(BaseModel):
    status: str                          # ACCEPTED | ENHANCED | REJECTED
    composite_score: float
    sharpness: float
    illumination: float
    field_of_view: float
    feedback: str


# ── DR Grading ─────────────────────────────────────────────────────────────────

class GradeResult(BaseModel):
    grade: int                           # 0–4 ICDR level
    grade_label: str                     # "No DR", "Mild NPDR", etc.
    confidence: float                    # Calibrated % confidence
    is_referable: bool
    all_probs: Dict[str, float]          # All 5 class probabilities
    low_confidence_flag: bool
    recommendation: str


# ── Lesion Finding ─────────────────────────────────────────────────────────────

class LesionFinding(BaseModel):
    lesion: str
    count: Any                           # int or "Present" / "Detected"
    supports: str
    icdr_criterion: str
    severity: str                        # mild | moderate | severe | critical


class GradeInfo(BaseModel):
    label: str
    criteria: str
    color: str                           # CSS hex


class EvidenceReport(BaseModel):
    grade: int
    grade_info: GradeInfo
    findings: List[LesionFinding]
    finding_count: int
    referral_urgency: str


# ── Lesion Counts ──────────────────────────────────────────────────────────────

class LesionCounts(BaseModel):
    microaneurysms: int
    hard_exudates: str                   # "present" | "absent"
    hemorrhages: str


# ── Analysis Response (main /api/analyze endpoint) ────────────────────────────

class AnalysisResponse(BaseModel):
    report_id: str
    patient_id: str
    eye: str
    quality: QualityResult
    grade: GradeResult
    evidence: EvidenceReport
    heatmap_b64: str                     # Base64-encoded JPEG heatmap
    lesion_counts: LesionCounts
    processing_notes: Optional[str] = None
    simulation_mode: bool = False


# ── Report ─────────────────────────────────────────────────────────────────────

class ReportSummary(BaseModel):
    report_id: str
    patient_id: str
    eye: str
    grade: int
    grade_label: str
    confidence: float
    is_referable: bool
    recommendation: str
    referral_urgency: str
    created_at: str
    reviewed: bool
    reviewer_notes: Optional[str] = None


# ── Review Queue ───────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=1)
    notes: Optional[str] = None
    override_grade: Optional[int] = Field(None, ge=0, le=4)


# ── Dashboard Stats ────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    screened_today: int
    screened_total: int
    referable_rate: float                # 0–100 %
    avg_confidence: float
    pending_review: int
    grade_distribution: Dict[str, int]


# ── Error Response ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    feedback: str

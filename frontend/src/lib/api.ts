/**
 * RetinaScan AI — API Client
 * Connects to FastAPI backend running on http://localhost:8000
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface QualityResult {
  status: "ACCEPTED" | "ENHANCED" | "REJECTED";
  composite_score: number;
  sharpness: number;
  illumination: number;
  field_of_view: number;
  feedback: string;
}

export interface GradeResult {
  grade: number; // 0–4
  grade_label: string; // "No DR", "Mild NPDR", etc.
  confidence: number; // calibrated percentage
  is_referable: boolean;
  all_probs: Record<string, number>;
  low_confidence_flag: boolean;
  recommendation: string;
}

export interface LesionFinding {
  lesion: string;
  count: string | number;
  supports: string;
  icdr_criterion: string;
  severity: "mild" | "moderate" | "severe" | "critical";
}

export interface GradeInfo {
  label: string;
  criteria: string;
  color: string;
}

export interface EvidenceReport {
  grade: number;
  grade_info: GradeInfo;
  findings: LesionFinding[];
  finding_count: number;
  referral_urgency: string;
}

export interface LesionCounts {
  microaneurysms: number;
  hard_exudates: string;
  hemorrhages: string;
}

export interface AnalysisResponse {
  report_id: string;
  patient_id: string;
  eye: string;
  quality: QualityResult;
  grade: GradeResult;
  evidence: EvidenceReport;
  heatmap_b64: string;
  lesion_counts: LesionCounts;
  processing_notes?: string;
  simulation_mode: boolean;
}

export interface ReportSummary {
  report_id: string;
  patient_id: string;
  eye: string;
  grade: number;
  grade_label: string;
  confidence: number;
  is_referable: boolean;
  recommendation: string;
  referral_urgency: string;
  created_at: string;
  reviewed: boolean;
  reviewer_notes?: string;
}

export interface DashboardStats {
  screened_today: number;
  screened_total: number;
  referable_rate: number;
  avg_confidence: number;
  pending_review: number;
  grade_distribution: Record<string, number>;
}

export async function checkBackendHealth(): Promise<{ status: string; version: string; simulation_mode: boolean } | null> {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function analyzeFundusImage(
  file: File,
  patientId: string,
  eye: string
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientId || "PATIENT_UNKNOWN");
  formData.append("eye", eye || "right");

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let errorMsg = `Server returned status ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) errorMsg = err.detail;
      else if (err.feedback) errorMsg = `${err.error}: ${err.feedback}`;
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  return await res.json();
}

export async function getReport(reportId: string): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/report/${reportId}`, {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Report not found (${res.status})`);
  }

  return await res.json();
}

export function getPdfDownloadUrl(reportId: string): string {
  return `${API_BASE}/api/report/${reportId}/pdf`;
}

export async function getReviewQueue(
  gradeFilter?: number,
  minConfidence?: number
): Promise<ReportSummary[]> {
  const params = new URLSearchParams();
  if (gradeFilter !== undefined && gradeFilter !== null) {
    params.append("grade_filter", gradeFilter.toString());
  }
  if (minConfidence !== undefined && minConfidence !== null) {
    params.append("min_confidence", minConfidence.toString());
  }

  const url = `${API_BASE}/api/queue${params.toString() ? `?${params.toString()}` : ""}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch review queue");
  const data = await res.json();
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.items)) return data.items;
  return [];
}

export async function submitReview(
  reportId: string,
  reviewerId: string,
  notes?: string,
  overrideGrade?: number
): Promise<{ status: string; report_id: string; reviewed: boolean }> {
  const body: { reviewer_id: string; notes?: string; override_grade?: number } = {
    reviewer_id: reviewerId,
  };
  if (notes) body.notes = notes;
  if (overrideGrade !== undefined && overrideGrade !== null) {
    body.override_grade = overrideGrade;
  }

  const res = await fetch(`${API_BASE}/api/queue/${reportId}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`Review submission failed (${res.status})`);
  }
  return await res.json();
}

export async function getDashboardStats(): Promise<DashboardStats> {
  const res = await fetch(`${API_BASE}/api/stats`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch dashboard statistics");
  return await res.json();
}

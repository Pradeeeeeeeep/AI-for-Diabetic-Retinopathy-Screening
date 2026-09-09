"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Download,
  CheckCircle,
  Clock,
  Printer,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Share2,
  User,
  Calendar,
  Stethoscope,
  FileCheck,
} from "lucide-react";
import { getReport, getPdfDownloadUrl, AnalysisResponse } from "@/lib/api";
import GradeGauge from "@/components/GradeGauge";
import ConfidenceBar from "@/components/ConfidenceBar";
import GradCAMViewer from "@/components/GradCAMViewer";
import LesionEvidenceCard from "@/components/LesionEvidenceCard";
import ReviewModal from "@/components/ReviewModal";

export default function ResultPage() {
  const params = useParams();
  const router = useRouter();
  const reportId = Array.isArray(params.id) ? params.id[0] : params.id;

  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isReviewOpen, setIsReviewOpen] = useState<boolean>(false);
  const [reviewed, setReviewed] = useState<boolean>(false);

  const fetchReport = async () => {
    if (!reportId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getReport(reportId);
      setData(res);
    } catch (err: any) {
      setError(err.message || "Failed to load clinical report");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
  }, [reportId]);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-24 flex flex-col items-center justify-center space-y-4 text-center">
        <RefreshCw className="w-8 h-8 text-sky-600 animate-spin" />
        <h2 className="text-xl font-bold text-slate-800">Generating Clinical Examination Report...</h2>
        <p className="text-xs text-slate-500 font-mono">Report ID: {reportId}</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-xl mx-auto px-4 py-24 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-rose-50 border border-rose-200 text-rose-600 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <h2 className="text-xl font-bold text-slate-900">Clinical Report Not Found</h2>
        <p className="text-sm text-slate-600">{error || "Unable to retrieve patient report."}</p>
        <div className="pt-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-sky-600 text-sm font-bold text-white hover:bg-sky-700 transition-colors shadow-xs"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Return to OPD Screening</span>
          </Link>
        </div>
      </div>
    );
  }

  const originalImageSrc = `data:image/jpeg;base64,${data.heatmap_b64}`;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">
      {/* Breadcrumb & Action Toolbar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Link href="/" className="hover:text-sky-600 transition-colors flex items-center gap-1 font-medium">
              <ArrowLeft className="w-3.5 h-3.5" />
              OPD Examination
            </Link>
            <span>/</span>
            <span className="text-slate-800 font-mono font-semibold">{data.report_id}</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 flex items-center gap-3">
            <span>Retinal Examination & Triage Sheet</span>
            {data.simulation_mode && (
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-sky-50 border border-sky-200 text-sky-800 font-mono font-medium">
                Clinical Validation
              </span>
            )}
          </h1>
        </div>

        {/* Doctor Action Buttons */}
        <div className="flex flex-wrap items-center gap-3 w-full sm:w-auto">
          {/* Download Clinical PDF Slip */}
          <a
            href={getPdfDownloadUrl(data.report_id)}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-800 text-xs font-bold border border-slate-300 transition-all shadow-xs"
          >
            <Download className="w-4 h-4 text-sky-600" />
            <span>Download Official Clinical PDF</span>
          </a>

          {/* Doctor Sign-Off & Stamp */}
          <button
            onClick={() => setIsReviewOpen(true)}
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all shadow-xs cursor-pointer ${
              reviewed
                ? "bg-emerald-50 text-emerald-800 border border-emerald-300"
                : "bg-sky-600 hover:bg-sky-700 text-white"
            }`}
          >
            <Stethoscope className="w-4 h-4" />
            <span>{reviewed ? "Doctor Signed Off (Stamped)" : "Doctor Sign-Off & Stamp"}</span>
          </button>
        </div>
      </div>

      {/* Patient Meta Strip (Indian Healthcare Format) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-2xl bg-white border border-slate-200 shadow-xs text-xs">
        <div>
          <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
            Patient / ABHA Record
          </span>
          <span className="font-bold text-slate-900 mt-0.5 block truncate">
            {data.patient_id}
          </span>
        </div>

        <div>
          <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
            Examined Eye (Oculus)
          </span>
          <span className="font-bold text-slate-900 mt-0.5 block capitalize">
            {data.eye === "right" ? "Right Eye (OD — Oculus Dexter)" : "Left Eye (OS — Oculus Sinister)"}
          </span>
        </div>

        <div>
          <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
            Retinal Photo Quality
          </span>
          <span
            className={`font-bold mt-0.5 block ${
              data.quality.status === "ACCEPTED"
                ? "text-emerald-700"
                : data.quality.status === "ENHANCED"
                ? "text-sky-700"
                : "text-rose-700"
            }`}
          >
            {data.quality.status} (Quality Index: {(data.quality.composite_score * 100).toFixed(0)}%)
          </span>
        </div>

        <div>
          <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
            Clarity & Illumination
          </span>
          <span className="font-mono text-slate-700 mt-0.5 block">
            {data.quality.sharpness.toFixed(0)} Var • {data.quality.illumination.toFixed(0)} Lum
          </span>
        </div>
      </div>

      {/* TWO-PANEL DOCTOR WORKSPACE */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT PANEL (7 cols): Grad-CAM Heatmap Viewer */}
        <div className="lg:col-span-7 space-y-4">
          <GradCAMViewer
            originalImageSrc={originalImageSrc}
            heatmapB64={data.heatmap_b64}
            patientId={data.patient_id}
            eye={data.eye}
            qualityStatus={data.quality.status}
          />
        </div>

        {/* RIGHT PANEL (5 cols): Diagnostic Verdict, Confidence, Lesion Evidence */}
        <div className="lg:col-span-5 space-y-5">
          {/* DR Grade Gauge */}
          <GradeGauge
            grade={data.grade.grade}
            gradeLabel={data.grade.grade_label}
            isReferable={data.grade.is_referable}
            recommendation={data.grade.recommendation}
            referralUrgency={data.evidence.referral_urgency}
          />

          {/* Calibrated Confidence Bar */}
          <ConfidenceBar
            confidence={data.grade.confidence}
            lowConfidenceFlag={data.grade.low_confidence_flag}
            allProbs={data.grade.all_probs}
          />

          {/* Lesion Findings Accordion */}
          <LesionEvidenceCard
            findings={data.evidence.findings}
            findingCount={data.evidence.finding_count}
            gradeCriteria={data.evidence.grade_info.criteria}
            lesionCounts={data.lesion_counts}
          />
        </div>
      </div>

      {/* Doctor Sign-off Dialog Modal */}
      <ReviewModal
        reportId={data.report_id}
        initialGrade={data.grade.grade}
        isOpen={isReviewOpen}
        onClose={() => setIsReviewOpen(false)}
        onSuccess={() => {
          setReviewed(true);
        }}
      />
    </div>
  );
}

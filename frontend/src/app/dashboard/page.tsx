"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  Users,
  AlertTriangle,
  Sparkles,
  CheckCircle,
  Clock,
  RefreshCw,
  Eye,
  BarChart3,
  Layers,
  Building2,
  Stethoscope,
} from "lucide-react";
import { getDashboardStats, getReviewQueue, DashboardStats, ReportSummary } from "@/lib/api";
import ReviewQueue from "@/components/ReviewQueue";

const GRADE_LABELS: Record<string, { label: string; color: string }> = {
  "0": { label: "Level 0 (No DR)", color: "bg-emerald-500" },
  "1": { label: "Level 1 (Mild)", color: "bg-lime-500" },
  "2": { label: "Level 2 (Moderate)", color: "bg-amber-500" },
  "3": { label: "Level 3 (Severe)", color: "bg-orange-500" },
  "4": { label: "Level 4 (PDR)", color: "bg-rose-500" },
};

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [queue, setQueue] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsData, queueData] = await Promise.all([
        getDashboardStats(),
        getReviewQueue(),
      ]);
      setStats(statsData);
      setQueue(queueData);
    } catch (err: any) {
      setError(err.message || "Failed to load dashboard data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-50 border border-sky-200 text-sky-800 text-xs font-bold mb-2">
            <Building2 className="w-3.5 h-3.5 text-sky-600" />
            <span>District Tele-Ophthalmology Network (NPCBVI)</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Screening Analytics & Doctor Review Queue
          </h1>
          <p className="text-xs sm:text-sm text-slate-600 mt-1">
            Real-time diabetic retinopathy triage, specialist referrals, and ophthalmologist sign-off queue.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white hover:bg-slate-50 text-slate-700 text-xs font-bold border border-slate-300 transition-all shadow-xs disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-sky-600" : ""}`} />
          <span>Refresh Records</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-medium">
          {error} — Make sure the FastAPI backend is running on port 8000.
        </div>
      )}

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Screened Today */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs uppercase font-bold tracking-wider">Screened Today</span>
            <Users className="w-4 h-4 text-sky-600" />
          </div>
          <div className="text-3xl font-black mono text-slate-900">
            {stats ? stats.screened_today : "—"}
          </div>
          <div className="text-[11px] text-slate-500">
            Total district screenings: <span className="text-slate-800 font-mono font-bold">{stats?.screened_total ?? 0}</span>
          </div>
        </div>

        {/* Referable Rate */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs uppercase font-bold tracking-wider">Referral Rate</span>
            <AlertTriangle className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-3xl font-black mono text-amber-600">
            {stats ? `${stats.referable_rate.toFixed(1)}%` : "—"}
          </div>
          <div className="text-[11px] text-slate-500">
            Patients graded Level 2+ (Moderate, Severe, PDR)
          </div>
        </div>

        {/* Average Calibrated Confidence */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs uppercase font-bold tracking-wider">AI Confidence</span>
            <Sparkles className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-3xl font-black mono text-emerald-600">
            {stats ? `${stats.avg_confidence.toFixed(1)}%` : "—"}
          </div>
          <div className="text-[11px] text-slate-500">
            Calibrated softmax reliability score
          </div>
        </div>

        {/* Pending Review */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs uppercase font-bold tracking-wider">Doctor Queue</span>
            <Clock className="w-4 h-4 text-rose-600" />
          </div>
          <div className="text-3xl font-black mono text-rose-600">
            {stats ? stats.pending_review : queue.length}
          </div>
          <div className="text-[11px] text-slate-500">
            Referable cases awaiting doctor sign-off
          </div>
        </div>
      </div>

      {/* Grade Distribution Bar */}
      {stats && stats.grade_distribution && (
        <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-sky-600" />
              Patient Grade Distribution (ICDR 5-Class Scale)
            </h3>
            <span className="text-xs text-slate-500 font-mono font-bold">
              Total Evaluated = {stats.screened_total}
            </span>
          </div>

          <div className="w-full h-3 rounded-full bg-slate-100 overflow-hidden flex border border-slate-200">
            {Object.entries(stats.grade_distribution).map(([gradeKey, count]) => {
              const total = stats.screened_total || 1;
              const pct = (count / total) * 100;
              const cfg = GRADE_LABELS[gradeKey] || { color: "bg-slate-400" };
              if (pct === 0) return null;
              return (
                <div
                  key={gradeKey}
                  className={`${cfg.color} h-full transition-all`}
                  style={{ width: `${pct}%` }}
                  title={`Level ${gradeKey}: ${count} (${pct.toFixed(1)}%)`}
                />
              );
            })}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1">
            {Object.entries(GRADE_LABELS).map(([key, item]) => {
              const count = stats.grade_distribution[key] || 0;
              const total = stats.screened_total || 1;
              const pct = ((count / total) * 100).toFixed(0);

              return (
                <div key={key} className="flex items-center gap-2 text-xs text-slate-700">
                  <div className={`w-3 h-3 rounded-full ${item.color} shrink-0`} />
                  <div className="truncate">
                    <span className="font-bold">{item.label}: </span>
                    <span className="font-mono text-slate-500">{count} ({pct}%)</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Review Queue Component */}
      <ReviewQueue queue={queue} onRefresh={loadData} />
    </div>
  );
}

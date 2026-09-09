"use client";

import React, { useState } from "react";
import { CheckCircle, X, Stethoscope, UserCheck } from "lucide-react";
import { submitReview } from "@/lib/api";

interface ReviewModalProps {
  reportId: string;
  initialGrade: number;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function ReviewModal({
  reportId,
  initialGrade,
  isOpen,
  onClose,
  onSuccess,
}: ReviewModalProps) {
  const [reviewerId, setReviewerId] = useState<string>("Dr. S. K. Mukherjee (MCI Reg: 48921-WB)");
  const [overrideGrade, setOverrideGrade] = useState<number | undefined>(undefined);
  const [notes, setNotes] = useState<string>(
    "Clinical review confirmed superior-temporal microaneurysms and hard exudates. Advised OCT macula scan and HbA1c control."
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      await submitReview(reportId, reviewerId, notes, overrideGrade);
      onSuccess();
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to submit doctor review");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="w-full max-w-lg rounded-2xl bg-white border border-slate-200 shadow-2xl p-6">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center">
              <Stethoscope className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Ophthalmologist Clinical Review & Sign-Off
              </h3>
              <p className="text-xs text-slate-500">Official Tele-Retina Medical Sign-Off</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          {error && (
            <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-medium">
              {error}
            </div>
          )}

          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1">
              Consultant Doctor Name & Medical Council Registration No.
            </label>
            <input
              type="text"
              required
              value={reviewerId}
              onChange={(e) => setReviewerId(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors"
              placeholder="e.g. Dr. A. K. Sharma (MCI Reg: 34102)"
            />
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1">
              Clinical Grade Verification / Expert Override
            </label>
            <select
              value={overrideGrade ?? ""}
              onChange={(e) =>
                setOverrideGrade(e.target.value === "" ? undefined : Number(e.target.value))
              }
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors"
            >
              <option value="">Confirm AI Staging (Level {initialGrade})</option>
              <option value="0">Override to Level 0 — No Diabetic Retinopathy</option>
              <option value="1">Override to Level 1 — Mild NPDR</option>
              <option value="2">Override to Level 2 — Moderate NPDR</option>
              <option value="3">Override to Level 3 — Severe NPDR</option>
              <option value="4">Override to Level 4 — Proliferative DR (PDR)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1">
              Doctor&apos;s Prescription Remarks & Treatment Advice
            </label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors resize-none"
              placeholder="e.g. Laser photocoagulation advised. Regular HbA1c monitoring. Review in 1 month."
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-5 py-2.5 rounded-xl text-sm font-bold bg-sky-600 hover:bg-sky-700 text-white transition-colors shadow-xs disabled:opacity-50 flex items-center gap-2 cursor-pointer"
            >
              <UserCheck className="w-4 h-4" />
              <span>{isSubmitting ? "Signing off..." : "Sign Off & Stamp Report"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

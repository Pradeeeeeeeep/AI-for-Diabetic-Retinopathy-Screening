"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  FileText,
  AlertTriangle,
  CheckCircle,
  Eye,
  Filter,
  ArrowRight,
  Clock,
  UserCheck,
  Search,
} from "lucide-react";
import { ReportSummary } from "@/lib/api";

interface ReviewQueueProps {
  queue: ReportSummary[];
  onRefresh?: () => void;
}

const GRADE_BADGES: Record<number, { text: string; class: string }> = {
  0: { text: "Level 0 (No DR)", class: "bg-emerald-50 text-emerald-800 border-emerald-200" },
  1: { text: "Level 1 (Mild)", class: "bg-lime-50 text-lime-800 border-lime-200" },
  2: { text: "Level 2 (Moderate)", class: "bg-amber-50 text-amber-800 border-amber-200" },
  3: { text: "Level 3 (Severe)", class: "bg-orange-50 text-orange-800 border-orange-200" },
  4: { text: "Level 4 (Proliferative)", class: "bg-rose-50 text-rose-800 border-rose-200" },
};

export default function ReviewQueue({ queue, onRefresh }: ReviewQueueProps) {
  const [filterGrade, setFilterGrade] = useState<string>("all");
  const [searchPatient, setSearchPatient] = useState<string>("");

  const items = Array.isArray(queue) ? queue : [];
  const filtered = items.filter((item) => {
    if (filterGrade !== "all" && item.grade !== Number(filterGrade)) return false;
    if (
      searchPatient &&
      !item.patient_id.toLowerCase().includes(searchPatient.toLowerCase()) &&
      !item.report_id.toLowerCase().includes(searchPatient.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
      {/* Table Header Controls */}
      <div className="p-5 border-b border-slate-200 bg-slate-50/70 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Clock className="w-4 h-4 text-sky-700" />
            Ophthalmologist Review & Referral Queue
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {filtered.length} pending / referable tele-retina cases requiring specialist validation
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 w-full sm:w-auto">
          {/* Patient Search */}
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchPatient}
              onChange={(e) => setSearchPatient(e.target.value)}
              placeholder="Search Patient Name / ABHA..."
              className="w-full pl-8 pr-3 py-2 rounded-xl bg-white border border-slate-300 text-xs text-slate-900 focus:outline-none focus:border-sky-600 shadow-xs"
            />
          </div>

          {/* Grade Filter */}
          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <select
              value={filterGrade}
              onChange={(e) => setFilterGrade(e.target.value)}
              className="px-3 py-2 rounded-xl bg-white border border-slate-300 text-xs font-semibold text-slate-800 focus:outline-none focus:border-sky-600 shadow-xs"
            >
              <option value="all">All Severity Levels</option>
              <option value="0">Level 0 (No DR)</option>
              <option value="1">Level 1 (Mild)</option>
              <option value="2">Level 2 (Moderate)</option>
              <option value="3">Level 3 (Severe)</option>
              <option value="4">Level 4 (PDR)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table Body */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-100/70 text-slate-600 font-bold uppercase tracking-wider">
              <th className="py-3 px-4">Report ID</th>
              <th className="py-3 px-4">Patient / ABHA Details</th>
              <th className="py-3 px-4">Eye (Oculus)</th>
              <th className="py-3 px-4">DR Staging</th>
              <th className="py-3 px-4">AI Confidence</th>
              <th className="py-3 px-4">Referral Urgency</th>
              <th className="py-3 px-4">Doctor Status</th>
              <th className="py-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-12 text-center text-slate-500 font-medium">
                  No screening records matching the selected criteria.
                </td>
              </tr>
            ) : (
              filtered.map((row) => {
                const gradeInfo =
                  GRADE_BADGES[row.grade] || GRADE_BADGES[0];

                return (
                  <tr
                    key={row.report_id}
                    className="hover:bg-slate-50/80 transition-colors"
                  >
                    <td className="py-3.5 px-4 font-mono font-bold text-sky-700">
                      {row.report_id}
                    </td>

                    <td className="py-3.5 px-4 font-semibold text-slate-800">
                      {row.patient_id}
                    </td>

                    <td className="py-3.5 px-4 capitalize text-slate-600 font-medium">
                      {row.eye === "right" ? "Right Eye (OD)" : "Left Eye (OS)"}
                    </td>

                    <td className="py-3.5 px-4">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-md text-[11px] font-bold border ${gradeInfo.class}`}
                      >
                        {gradeInfo.text}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 font-mono font-bold">
                      <span
                        className={
                          row.confidence < 65 ? "text-amber-600" : "text-slate-800"
                        }
                      >
                        {row.confidence.toFixed(1)}%
                      </span>
                      {row.confidence < 65 && (
                        <span className="ml-1.5 text-[10px] text-amber-800 font-sans px-1.5 py-0.2 rounded-sm bg-amber-100 border border-amber-200 font-bold">
                          Uncertain
                        </span>
                      )}
                    </td>

                    <td className="py-3.5 px-4">
                      <span className="text-slate-800 font-semibold">
                        {row.referral_urgency || "Standard Routine"}
                      </span>
                    </td>

                    <td className="py-3.5 px-4">
                      {row.reviewed ? (
                        <span className="inline-flex items-center gap-1 text-emerald-700 text-[11px] font-bold bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
                          <CheckCircle className="w-3.5 h-3.5" />
                          Signed Off
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-amber-700 text-[11px] font-bold bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">
                          <Clock className="w-3.5 h-3.5" />
                          Pending Review
                        </span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      <Link
                        href={`/result/${row.report_id}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-50 hover:bg-sky-100 text-sky-800 font-bold border border-sky-200 transition-all text-xs"
                      >
                        <span>Open Report</span>
                        <ArrowRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

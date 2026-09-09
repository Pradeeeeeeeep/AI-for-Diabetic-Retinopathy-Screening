"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, FileText, CircleDot, CheckCircle2 } from "lucide-react";
import { LesionFinding } from "@/lib/api";

interface LesionEvidenceCardProps {
  findings: LesionFinding[];
  findingCount: number;
  gradeCriteria?: string;
  lesionCounts?: {
    microaneurysms: number;
    hard_exudates: string;
    hemorrhages: string;
  };
}

const SEVERITY_BADGE: Record<string, { label: string; class: string }> = {
  mild: { label: "Mild Stage", class: "bg-emerald-50 text-emerald-800 border-emerald-200" },
  moderate: { label: "Moderate Stage", class: "bg-amber-50 text-amber-800 border-amber-200" },
  severe: { label: "Severe Stage", class: "bg-orange-50 text-orange-800 border-orange-200" },
  critical: { label: "Critical (PDR)", class: "bg-rose-50 text-rose-800 border-rose-200" },
};

export default function LesionEvidenceCard({
  findings,
  findingCount,
  gradeCriteria,
  lesionCounts,
}: LesionEvidenceCardProps) {
  const [expanded, setExpanded] = useState<boolean>(true);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
      {/* Card Header Accordion */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-5 text-left bg-slate-50/70 hover:bg-slate-100/60 transition-colors border-b border-slate-200"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-sky-100 border border-sky-200 flex items-center justify-center text-sky-700">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-900">
              Retinal Lesions & ICDR Diagnostic Evidence
            </h4>
            <p className="text-xs text-slate-500">
              {findingCount} clinical biomarkers detected by AI pipeline
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-slate-200 text-slate-700">
            {findingCount} criteria
          </span>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-slate-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-500" />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {expanded && (
        <div className="p-5 space-y-4">
          {/* Quantitative Lesion Counter Row */}
          {lesionCounts && (
            <div className="grid grid-cols-3 gap-3 p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="border-r border-slate-200 pr-2">
                <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
                  Microaneurysms
                </div>
                <div className="text-xl font-black mono text-sky-700 mt-0.5">
                  {lesionCounts.microaneurysms}
                </div>
              </div>

              <div className="border-r border-slate-200 px-2">
                <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
                  Hard Exudates
                </div>
                <div className="text-sm font-bold capitalize text-slate-800 mt-1">
                  {lesionCounts.hard_exudates}
                </div>
              </div>

              <div className="pl-2">
                <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
                  Hemorrhages
                </div>
                <div className="text-sm font-bold capitalize text-slate-800 mt-1">
                  {lesionCounts.hemorrhages}
                </div>
              </div>
            </div>
          )}

          {/* ICDR Criteria Rule explanation */}
          {gradeCriteria && (
            <div className="p-3 rounded-xl bg-sky-50 border border-sky-200 text-xs text-sky-950 leading-relaxed">
              <span className="font-bold text-sky-900">Clinical Staging Criterion: </span>
              {gradeCriteria}
            </div>
          )}

          {/* Biomarker Finding Cards */}
          <div className="space-y-2.5">
            {findings.map((item, index) => {
              const badge =
                SEVERITY_BADGE[item.severity] || SEVERITY_BADGE.moderate;

              return (
                <div
                  key={index}
                  className="p-3.5 rounded-xl bg-slate-50/70 border border-slate-200 hover:border-slate-300 transition-colors"
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <div className="flex items-center gap-2">
                      <CircleDot className="w-3.5 h-3.5 text-sky-600" />
                      <span className="font-bold text-sm text-slate-900">
                        {item.lesion}
                      </span>
                      <span className="text-xs font-mono text-slate-600 px-2 py-0.2 rounded-md bg-white border border-slate-200">
                        Count: {item.count}
                      </span>
                    </div>

                    <span
                      className={`text-[10px] font-bold px-2.5 py-0.5 rounded-md border ${badge.class}`}
                    >
                      {badge.label}
                    </span>
                  </div>

                  <p className="text-xs text-slate-700 leading-relaxed">
                    <span className="font-semibold text-slate-800">Rule: </span>
                    {item.icdr_criterion}
                  </p>

                  <div className="mt-2 text-[11px] text-sky-700 font-semibold flex items-center gap-1.5">
                    <CheckCircle2 className="w-3 h-3 text-sky-600" />
                    <span>Evidence confirms: {item.supports}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

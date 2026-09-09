"use client";

import React from "react";
import { AlertCircle, ShieldCheck, Info } from "lucide-react";

interface ConfidenceBarProps {
  confidence: number;
  lowConfidenceFlag: boolean;
  allProbs?: Record<string, number>;
}

export default function ConfidenceBar({
  confidence,
  lowConfidenceFlag,
  allProbs,
}: ConfidenceBarProps) {
  const isWarning = lowConfidenceFlag || confidence < 65.0;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-sky-600" />
          <span className="text-xs uppercase font-bold tracking-wider text-slate-700">
            Model Confidence (Calibrated Softmax)
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 font-semibold">
            Temperature Scaled
          </span>
        </div>
        <span className="text-lg font-bold mono text-sky-700">
          {confidence.toFixed(1)}%
        </span>
      </div>

      {/* Main Confidence Bar */}
      <div className="relative w-full h-3 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            isWarning
              ? "bg-gradient-to-r from-amber-500 to-orange-500"
              : "bg-gradient-to-r from-sky-500 to-emerald-500"
          }`}
          style={{ width: `${Math.min(100, Math.max(0, confidence))}%` }}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-rose-500"
          style={{ left: "65%" }}
          title="65% Threshold for Human Review"
        />
      </div>

      {/* Uncertainty Notice for Doctor */}
      {isWarning && (
        <div className="flex items-start gap-2.5 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
          <div className="leading-relaxed">
            <span className="font-bold text-amber-950">Borderline Confidence Notice: </span>
            The AI confidence score ({confidence.toFixed(1)}%) is below the 65% certainty cutoff. 
            Clinical examination by a senior ophthalmologist or additional 7-field fundus photography is advised.
          </div>
        </div>
      )}

      {/* 5-Class Probability Distribution */}
      {allProbs && Object.keys(allProbs).length > 0 && (
        <div className="pt-2 border-t border-slate-100">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">
            Class Probability Breakdown
          </div>
          <div className="grid grid-cols-5 gap-2 text-center">
            {Object.entries(allProbs).map(([cls, prob]) => {
              const pct = (prob * 100).toFixed(1);
              return (
                <div
                  key={cls}
                  className="bg-slate-50 rounded-xl p-2 border border-slate-200 text-slate-800"
                >
                  <div className="text-[10px] text-slate-500 font-semibold truncate mb-0.5">
                    Level {cls.replace("level_", "")}
                  </div>
                  <div className="text-xs font-black mono text-slate-900">
                    {pct}%
                  </div>
                  <div className="w-full bg-slate-200 h-1.5 rounded-full mt-1.5 overflow-hidden">
                    <div
                      className="h-full bg-sky-600 rounded-full"
                      style={{ width: `${Math.min(100, prob * 100)}%` }}
                    />
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

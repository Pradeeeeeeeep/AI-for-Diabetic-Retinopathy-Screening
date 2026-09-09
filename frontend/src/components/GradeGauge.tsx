"use client";

import React from "react";
import { AlertTriangle, CheckCircle2, Stethoscope, Clock, ShieldAlert } from "lucide-react";

interface GradeGaugeProps {
  grade: number;
  gradeLabel: string;
  isReferable: boolean;
  recommendation?: string;
  referralUrgency?: string;
}

const GRADE_CONFIG: Record<
  number,
  {
    name: string;
    color: string;
    bgBadge: string;
    textBadge: string;
    borderBadge: string;
    patientAdvice: string;
  }
> = {
  0: {
    name: "Level 0 — No Diabetic Retinopathy",
    color: "#16a34a",
    bgBadge: "bg-emerald-50",
    textBadge: "text-emerald-800",
    borderBadge: "border-emerald-200",
    patientAdvice: "Fundus healthy. Advise strict glycemic (HbA1c < 7.0%) and BP control. Re-screen annually.",
  },
  1: {
    name: "Level 1 — Mild Non-Proliferative DR",
    color: "#65a30d",
    bgBadge: "bg-lime-50",
    textBadge: "text-lime-800",
    borderBadge: "border-lime-200",
    patientAdvice: "Microaneurysms only. Early disease stage. Reinforce medical management and re-screen in 9–12 months.",
  },
  2: {
    name: "Level 2 — Moderate Non-Proliferative DR",
    color: "#d97706",
    bgBadge: "bg-amber-50",
    textBadge: "text-amber-800",
    borderBadge: "border-amber-200",
    patientAdvice: "Refer to Ophthalmologist within 4–6 weeks for dilated biomicroscopy / OCT macula assessment.",
  },
  3: {
    name: "Level 3 — Severe Non-Proliferative DR (4-2-1 Rule)",
    color: "#ea580c",
    bgBadge: "bg-orange-50",
    textBadge: "text-orange-800",
    borderBadge: "border-orange-200",
    patientAdvice: "High risk of rapid progression to PDR. Urgent referral to Vitreo-Retina specialist within 2–4 weeks.",
  },
  4: {
    name: "Level 4 — Proliferative Diabetic Retinopathy (PDR)",
    color: "#dc2626",
    bgBadge: "bg-rose-50",
    textBadge: "text-rose-800",
    borderBadge: "border-rose-200",
    patientAdvice: "Neovascularization present. Sight-threatening emergency. Immediate referral for PRP Laser / Anti-VEGF therapy.",
  },
};

export default function GradeGauge({
  grade,
  gradeLabel,
  isReferable,
  recommendation,
  referralUrgency,
}: GradeGaugeProps) {
  const current = GRADE_CONFIG[grade] || GRADE_CONFIG[0];

  const radius = 58;
  const circumference = 2 * Math.PI * radius;
  const normalizedLevel = (grade + 1) / 5;
  const strokeDashoffset = circumference - normalizedLevel * (circumference * 0.75);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-5">
      {/* Header Clinical Classification Banner */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Stethoscope className="w-4 h-4 text-sky-600" />
          <span className="text-xs uppercase font-bold tracking-wider text-slate-500">
            ICDR International Clinical Staging
          </span>
        </div>

        {isReferable ? (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-50 text-rose-700 border border-rose-200">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
            SPECIALIST REFERRAL REQUIRED
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            ROUTINE TELE-RETINA SCREENING
          </span>
        )}
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6">
        {/* Circular Gauge */}
        <div className="relative w-36 h-36 flex items-center justify-center shrink-0">
          <svg className="w-full h-full transform -rotate-135" viewBox="0 0 160 160">
            <circle
              cx="80"
              cy="80"
              r={radius}
              fill="transparent"
              stroke="#f1f5f9"
              strokeWidth="14"
              strokeDasharray={circumference * 0.75}
              strokeDashoffset="0"
              strokeLinecap="round"
            />
            <circle
              cx="80"
              cy="80"
              r={radius}
              fill="transparent"
              stroke={current.color}
              strokeWidth="14"
              strokeDasharray={circumference * 0.75}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
            />
          </svg>

          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span
              className="text-4xl font-black mono leading-none"
              style={{ color: current.color }}
            >
              {grade}
            </span>
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mt-1">
              Stage / 4
            </span>
          </div>
        </div>

        {/* Diagnosis & Clinical Action */}
        <div className="flex-1 text-center sm:text-left space-y-2">
          <div className="inline-block">
            <span className="text-xs text-slate-500 font-semibold block">
              Doctor&apos;s Diagnostic Verdict:
            </span>
            <h3
              className="text-xl font-extrabold tracking-tight"
              style={{ color: current.color }}
            >
              {gradeLabel || current.name}
            </h3>
          </div>

          <p className="text-xs text-slate-700 leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-200">
            <span className="font-bold text-slate-900">Clinical Protocol: </span>
            {recommendation || current.patientAdvice}
          </p>

          <div className="flex items-center gap-2 pt-1 text-xs">
            <Clock className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-600 font-medium">Referral Urgency:</span>
            <span
              className={`font-bold px-2.5 py-0.5 rounded-md border text-xs ${current.bgBadge} ${current.textBadge} ${current.borderBadge}`}
            >
              {referralUrgency || "Standard Follow-Up"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

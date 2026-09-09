import React from "react";
import ImageUploader from "@/components/ImageUploader";
import {
  Stethoscope,
  Activity,
  CheckCircle2,
  ShieldCheck,
  Search,
  Users,
  Award,
} from "lucide-react";

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-10">
      {/* Clinician Welcome & Banner */}
      <div className="text-center space-y-3 max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-sky-50 border border-sky-200 text-sky-800 text-xs font-bold shadow-xs">
          <Activity className="w-3.5 h-3.5 text-sky-600" />
          <span>Tele-Ophthalmology & District Eye Health Screening</span>
        </div>

        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900">
          Diabetic Retinopathy Clinical Triage System
        </h1>

        <p className="text-sm sm:text-base text-slate-600 leading-relaxed max-w-2xl mx-auto">
          AI-powered clinical decision support designed for Indian ophthalmologists, optometrists, 
          and PHC doctors to rapidly grade fundus photographs and identify patients needing specialist intervention.
        </p>
      </div>

      {/* Main Upload Box */}
      <ImageUploader />

      {/* Clinical Workflow Standards (Aravind / Sankara / AIIMS Protocols) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 pt-8 border-t border-slate-200">
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2.5">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-slate-900">
            Image Quality Verification
          </h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Automated Laplacian sharpness check and illumination control. Non-gradable poor quality photographs are flagged immediately to save clinical time.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2.5">
          <div className="w-10 h-10 rounded-xl bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-700">
            <Search className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-slate-900">
            Explainable Lesion Heatmap
          </h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            High-resolution Grad-CAM overlays highlight microaneurysms, hemorrhages, and exudates directly on the fundus photograph for clinical validation.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-xs space-y-2.5">
          <div className="w-10 h-10 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-slate-900">
            Calibrated Triage & Referral
          </h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Standardized ICDR 5-level grading with clear doctor action plans: routine follow-up vs. urgent referral to a Vitreo-Retina specialist.
          </p>
        </div>
      </div>
    </div>
  );
}

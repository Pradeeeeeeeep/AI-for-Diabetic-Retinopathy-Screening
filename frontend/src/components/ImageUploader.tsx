"use client";

import React, { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  UploadCloud,
  FileImage,
  CheckCircle2,
  AlertCircle,
  Eye,
  Sparkles,
  ArrowRight,
  RefreshCw,
  User,
  Hash,
  Stethoscope,
  Info,
} from "lucide-react";
import { analyzeFundusImage } from "@/lib/api";

const DEMO_SAMPLES = [
  {
    name: "Severe NPDR (4-2-1 Rule)",
    subtext: "Multiple blot hemorrhages & venous beading",
    path: "/samples/sample_1_severe.png",
    tag: "High Priority Referral",
    color: "text-amber-700 bg-amber-50 border-amber-200",
  },
  {
    name: "Moderate NPDR",
    subtext: "Microaneurysms + hard exudates present",
    path: "/samples/sample_2_moderate.png",
    tag: "Refer to Specialist",
    color: "text-orange-700 bg-orange-50 border-orange-200",
  },
  {
    name: "Mild NPDR / Early Stage",
    subtext: "Isolated microaneurysms, good macula",
    path: "/samples/sample_3_mild.png",
    tag: "Routine Follow-up",
    color: "text-emerald-700 bg-emerald-50 border-emerald-200",
  },
];

export default function ImageUploader() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [patientName, setPatientName] = useState<string>("Ramesh K. Patel");
  const [patientId, setPatientId] = useState<string>("ABHA-9821-4402-1920");
  const [age, setAge] = useState<string>("56");
  const [gender, setGender] = useState<string>("Male");
  const [eye, setEye] = useState<"left" | "right">("right");
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisStep, setAnalysisStep] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileChange = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a standard retinal fundus image (PNG, JPG, JPEG, TIFF).");
      return;
    }
    setErrorMessage(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const loadSample = async (samplePath: string, sampleName: string) => {
    try {
      setErrorMessage(null);
      const res = await fetch(samplePath);
      const blob = await res.blob();
      const file = new File([blob], samplePath.split("/").pop() || "sample.png", {
        type: "image/png",
      });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } catch (err) {
      setErrorMessage("Failed to load sample image.");
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select or upload a retinal fundus photograph first.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);

    // Doctor-friendly progression steps
    setAnalysisStep("Step 1/4: Checking Image Clarity & Field of View (Laplacian)...");
    setTimeout(() => {
      setAnalysisStep("Step 2/4: Segmenting Blood Vessels, Exudates & Hemorrhages...");
    }, 1200);
    setTimeout(() => {
      setAnalysisStep("Step 3/4: ICDR 5-Level Staging via Deep Neural Network...");
    }, 2400);
    setTimeout(() => {
      setAnalysisStep("Step 4/4: Generating Grad-CAM Diagnostic Map & Report...");
    }, 3600);

    try {
      const fullPatientId = `${patientId} | ${patientName} (${age}y/${gender})`;
      const result = await analyzeFundusImage(selectedFile, fullPatientId, eye);
      router.push(`/result/${result.report_id}`);
    } catch (err: any) {
      setIsAnalyzing(false);
      setErrorMessage(err.message || "Analysis request failed. Please check backend connection.");
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Step 1: Fundus Image Input */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2.5">
            <span className="w-6 h-6 rounded-full bg-sky-600 text-white font-bold text-xs flex items-center justify-center">
              1
            </span>
            <h3 className="text-base font-bold text-slate-900">
              Upload or Select Patient Retinal Fundus Photo
            </h3>
          </div>
          <span className="text-xs text-slate-500 hidden sm:inline">
            Standard 45° / 50° Macula or Disc centered
          </span>
        </div>

        {/* Drag & Drop Area */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => {
            if (!previewUrl && fileInputRef.current) fileInputRef.current.click();
          }}
          className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-8 transition-all flex flex-col items-center justify-center text-center cursor-pointer ${
            isDragging
              ? "border-sky-500 bg-sky-50"
              : previewUrl
              ? "border-slate-300 bg-slate-50/50"
              : "border-slate-300 bg-slate-50 hover:bg-sky-50/40 hover:border-sky-400"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileChange(e.target.files[0]);
              }
            }}
          />

          {previewUrl ? (
            <div className="relative w-full flex flex-col items-center">
              <div className="relative max-h-72 max-w-sm rounded-xl overflow-hidden border border-slate-300 shadow-md bg-black">
                <img
                  src={previewUrl}
                  alt="Fundus Preview"
                  className="w-full h-auto object-contain"
                />
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (fileInputRef.current) fileInputRef.current.click();
                  }}
                  className="absolute top-2 right-2 px-3 py-1 rounded-lg bg-white/90 hover:bg-white text-xs font-bold text-slate-800 shadow-md transition-colors"
                >
                  Change Photo
                </button>
              </div>
              <p className="text-xs text-slate-600 font-mono mt-3">
                Selected: {selectedFile?.name} ({((selectedFile?.size || 0) / 1024).toFixed(0)} KB)
              </p>
            </div>
          ) : (
            <div className="py-6 space-y-3">
              <div className="w-14 h-14 rounded-2xl bg-sky-100 text-sky-700 mx-auto flex items-center justify-center shadow-xs">
                <UploadCloud className="w-7 h-7" />
              </div>
              <div>
                <h4 className="text-base font-bold text-slate-800">
                  Click here to browse or drag & drop fundus photograph
                </h4>
                <p className="text-xs text-slate-500 mt-1">
                  Works with all digital fundus cameras (Topcon, Zeiss, Forus 3nethra, Remidio, etc.)
                </p>
              </div>
              <button
                type="button"
                className="mt-2 px-4 py-2 rounded-xl text-xs font-bold bg-white text-sky-700 border border-slate-300 shadow-xs hover:bg-slate-50 transition-colors"
              >
                Choose Photo File
              </button>
            </div>
          )}
        </div>

        {/* Quick Demo Test Presets */}
        <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-sky-600" />
              One-Click Patient Test Samples
            </span>
            <span className="text-[11px] text-slate-500">Instant demonstration</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {DEMO_SAMPLES.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => loadSample(sample.path, sample.name)}
                className="p-3 rounded-xl bg-white hover:bg-sky-50/50 border border-slate-200 hover:border-sky-300 text-left transition-all shadow-xs group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 group-hover:text-sky-700">
                    {sample.name}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 mt-1">{sample.subtext}</p>
                <span
                  className={`inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-md border ${sample.color}`}
                >
                  {sample.tag}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Step 2: Patient Information (Indian Healthcare Context) */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-5">
        <div className="flex items-center gap-2.5 border-b border-slate-100 pb-3">
          <span className="w-6 h-6 rounded-full bg-sky-600 text-white font-bold text-xs flex items-center justify-center">
            2
          </span>
          <h3 className="text-base font-bold text-slate-900">
            Patient Clinical Details (OPD / ABHA)
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Patient Name */}
          <div className="sm:col-span-2">
            <label className="block text-xs font-bold text-slate-700 mb-1.5">
              Patient Full Name
            </label>
            <input
              type="text"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors"
              placeholder="e.g. Ramesh Patel"
            />
          </div>

          {/* ABHA / OPD No */}
          <div className="sm:col-span-2">
            <label className="block text-xs font-bold text-slate-700 mb-1.5">
              ABHA ID / Hospital OPD No.
            </label>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm font-mono text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors"
              placeholder="e.g. ABHA-9821-4402-1920"
            />
          </div>

          {/* Age */}
          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1.5">Age (Years)</label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors"
              placeholder="e.g. 56"
            />
          </div>

          {/* Gender */}
          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1.5">Gender</label>
            <select
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-sky-600 transition-colors"
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Examined Eye (OD / OS) */}
          <div className="sm:col-span-2">
            <label className="block text-xs font-bold text-slate-700 mb-1.5">
              Examined Oculus (Eye)
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setEye("right")}
                className={`py-2.5 px-3 rounded-xl text-xs font-bold transition-all border ${
                  eye === "right"
                    ? "bg-sky-600 text-white border-sky-600 shadow-sm"
                    : "bg-slate-50 text-slate-700 border-slate-300 hover:bg-slate-100"
                }`}
              >
                Right Eye (OD — Oculus Dexter)
              </button>
              <button
                type="button"
                onClick={() => setEye("left")}
                className={`py-2.5 px-3 rounded-xl text-xs font-bold transition-all border ${
                  eye === "left"
                    ? "bg-sky-600 text-white border-sky-600 shadow-sm"
                    : "bg-slate-50 text-slate-700 border-slate-300 hover:bg-slate-100"
                }`}
              >
                Left Eye (OS — Oculus Sinister)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error Notice */}
      {errorMessage && (
        <div className="flex items-center gap-2.5 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-medium">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Big Action Button */}
      <button
        type="button"
        disabled={isAnalyzing || !selectedFile}
        onClick={handleAnalyze}
        className="w-full py-4 px-6 rounded-2xl bg-sky-600 hover:bg-sky-700 text-white font-bold text-base shadow-md shadow-sky-600/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 cursor-pointer"
      >
        {isAnalyzing ? (
          <>
            <RefreshCw className="w-5 h-5 animate-spin" />
            <span className="font-semibold">{analysisStep || "Analyzing Patient Fundus..."}</span>
          </>
        ) : (
          <>
            <Stethoscope className="w-5 h-5" />
            <span>Examine Patient Retina & Generate Clinical Diagnosis</span>
            <ArrowRight className="w-5 h-5" />
          </>
        )}
      </button>
    </div>
  );
}

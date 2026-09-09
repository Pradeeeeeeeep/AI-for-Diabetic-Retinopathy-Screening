"use client";

import React, { useState } from "react";
import { Eye, Layers, Sliders, Columns, ZoomIn, ZoomOut, RefreshCw, CheckCircle2 } from "lucide-react";

interface GradCAMViewerProps {
  originalImageSrc: string;
  heatmapB64: string;
  patientId?: string;
  eye?: string;
  qualityStatus?: string;
}

export default function GradCAMViewer({
  originalImageSrc,
  heatmapB64,
  patientId,
  eye,
  qualityStatus,
}: GradCAMViewerProps) {
  const [viewMode, setViewMode] = useState<"overlay" | "side-by-side" | "original">("overlay");
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(65);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const heatmapSrc = heatmapB64.startsWith("data:")
    ? heatmapB64
    : `data:image/jpeg;base64,${heatmapB64}`;

  return (
    <div className="flex flex-col h-full rounded-2xl bg-white border border-slate-200 overflow-hidden shadow-xs">
      {/* Top Bar: View Mode Switcher & Quality Badge */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-slate-200 bg-slate-50/80">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-sky-700" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Grad-CAM Lesion Heatmap Viewer
          </span>
          {qualityStatus && (
            <span
              className={`text-[11px] font-bold px-2 py-0.5 rounded-md border ${
                qualityStatus === "ACCEPTED"
                  ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                  : qualityStatus === "ENHANCED"
                  ? "bg-sky-50 text-sky-800 border-sky-200"
                  : "bg-rose-50 text-rose-800 border-rose-200"
              }`}
            >
              Image QA: {qualityStatus}
            </span>
          )}
        </div>

        {/* View Mode Buttons */}
        <div className="flex items-center gap-1 bg-white p-1 rounded-xl border border-slate-300 shadow-xs">
          <button
            onClick={() => setViewMode("overlay")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold transition-all ${
              viewMode === "overlay"
                ? "bg-sky-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Heatmap Overlay</span>
          </button>

          <button
            onClick={() => setViewMode("side-by-side")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold transition-all ${
              viewMode === "side-by-side"
                ? "bg-sky-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>

          <button
            onClick={() => setViewMode("original")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold transition-all ${
              viewMode === "original"
                ? "bg-sky-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Original Fundus</span>
          </button>
        </div>
      </div>

      {/* Main Retinal Optical Viewport (Dark background inside for optimal medical contrast) */}
      <div className="relative flex-1 min-h-[420px] bg-slate-950 flex items-center justify-center p-4 overflow-hidden select-none">
        {viewMode === "side-by-side" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full h-full">
            {/* Original Panel */}
            <div className="flex flex-col items-center justify-center relative rounded-xl border border-slate-800 bg-black/60 p-2 overflow-hidden">
              <span className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded-md bg-white/90 text-slate-900 text-xs font-bold shadow-xs">
                Patient Fundus (Original)
              </span>
              <img
                src={originalImageSrc}
                alt="Original Fundus"
                className="max-h-[380px] w-auto object-contain rounded-lg transition-transform duration-200"
                style={{ transform: `scale(${zoomLevel})` }}
              />
            </div>

            {/* Grad-CAM Heatmap Panel */}
            <div className="flex flex-col items-center justify-center relative rounded-xl border border-slate-800 bg-black/60 p-2 overflow-hidden">
              <span className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded-md bg-sky-600 text-white text-xs font-bold shadow-xs">
                Grad-CAM Activation Focus
              </span>
              <img
                src={heatmapSrc}
                alt="Grad-CAM Heatmap"
                className="max-h-[380px] w-auto object-contain rounded-lg transition-transform duration-200"
                style={{ transform: `scale(${zoomLevel})` }}
              />
            </div>
          </div>
        ) : viewMode === "original" ? (
          <div className="relative max-w-full max-h-full flex items-center justify-center">
            <img
              src={originalImageSrc}
              alt="Original Fundus"
              className="max-h-[440px] w-auto object-contain rounded-xl shadow-2xl transition-transform duration-200"
              style={{ transform: `scale(${zoomLevel})` }}
            />
          </div>
        ) : (
          /* Overlay Blend Mode */
          <div className="relative max-w-full max-h-full flex items-center justify-center">
            <img
              src={originalImageSrc}
              alt="Original Fundus"
              className="max-h-[440px] w-auto object-contain rounded-xl shadow-2xl transition-transform duration-200"
              style={{ transform: `scale(${zoomLevel})` }}
            />
            <img
              src={heatmapSrc}
              alt="Grad-CAM Overlay"
              className="absolute inset-0 max-h-[440px] m-auto w-auto object-contain rounded-xl mix-blend-screen transition-opacity duration-150 pointer-events-none"
              style={{
                opacity: heatmapOpacity / 100,
                transform: `scale(${zoomLevel})`,
              }}
            />
          </div>
        )}
      </div>

      {/* Bottom Controls Bar */}
      <div className="p-4 border-t border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-4">
        {/* Heatmap Opacity Slider */}
        {viewMode === "overlay" ? (
          <div className="flex items-center gap-3 flex-1 min-w-[200px] max-w-md">
            <Sliders className="w-4 h-4 text-sky-700 shrink-0" />
            <span className="text-xs text-slate-700 font-bold whitespace-nowrap">
              Heatmap Intensity:
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-xs font-bold mono text-sky-800 w-9 text-right">
              {heatmapOpacity}%
            </span>
          </div>
        ) : (
          <div className="text-xs font-medium text-slate-600">
            {viewMode === "side-by-side"
              ? "Dual comparison viewport active"
              : "Showing unaugmented retinal photograph"}
          </div>
        )}

        {/* Heatmap Attention Legend */}
        <div className="flex items-center gap-2 text-xs text-slate-600 font-medium">
          <span>AI Focus:</span>
          <div className="flex items-center h-3 rounded-md overflow-hidden w-28 border border-slate-300">
            <div className="h-full w-1/4 bg-blue-600" title="Low attention" />
            <div className="h-full w-1/4 bg-cyan-400" />
            <div className="h-full w-1/4 bg-amber-400" />
            <div className="h-full w-1/4 bg-rose-600" title="High lesion attention" />
          </div>
          <span className="text-rose-700 font-bold">Pathology</span>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoomLevel((z) => Math.max(0.75, z - 0.25))}
            className="p-1.5 rounded-lg bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 transition-colors shadow-xs"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-xs font-mono font-bold text-slate-700 w-12 text-center">
            {Math.round(zoomLevel * 100)}%
          </span>
          <button
            onClick={() => setZoomLevel((z) => Math.min(2.5, z + 0.25))}
            className="p-1.5 rounded-lg bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 transition-colors shadow-xs"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoomLevel(1)}
            className="p-1.5 rounded-lg bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 transition-colors shadow-xs ml-1"
            title="Reset Zoom"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

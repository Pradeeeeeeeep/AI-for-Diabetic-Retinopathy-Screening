import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "RetinaScan AI — Tele-Ophthalmology DR Screening Portal",
  description:
    "Doctor-friendly Diabetic Retinopathy screening platform with Grad-CAM explainability, calibrated ICDR grading, and clinical referral reports.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 flex flex-col antialiased">
        <Navbar />
        <main className="flex-1 w-full">{children}</main>
        <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-600">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-800">RetinaScan AI</span>
              <span>— Clinical Decision Support for Indian Eye Care & PHC Tele-Ophthalmology</span>
            </div>
            <div className="font-mono text-[11px] text-slate-500 flex items-center gap-3">
              <span>ABDM & Tele-Retina Ready</span>
              <span>•</span>
              <span>ICDR 5-Class Standard</span>
              <span>•</span>
              <span>AIIMS & Sankara Protocol</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}

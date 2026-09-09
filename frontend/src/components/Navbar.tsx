"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Eye, LayoutDashboard, Stethoscope, CheckCircle2, AlertCircle } from "lucide-react";
import { checkBackendHealth } from "@/lib/api";

export default function Navbar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = useState<"checking" | "connected" | "offline">("checking");
  const [simulationMode, setSimulationMode] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const health = await checkBackendHealth();
      if (!mounted) return;
      if (health) {
        setBackendStatus("connected");
        setSimulationMode(health.simulation_mode);
      } else {
        setBackendStatus("offline");
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/95 backdrop-blur-md px-4 sm:px-8 py-3.5 shadow-xs">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo & Clinical Brand */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-700 group-hover:border-sky-400 transition-colors shadow-xs">
            <Eye className="w-5 h-5 text-sky-600" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-slate-900 tracking-tight">RetinaScan AI</span>
              <span className="text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded-md bg-sky-100 text-sky-800 border border-sky-200">
                Doctor OPD Portal
              </span>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              Tele-Ophthalmology & Diabetic Retinopathy Triage
            </p>
          </div>
        </Link>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2">
          <Link
            href="/"
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
              pathname === "/"
                ? "bg-sky-600 text-white shadow-sm shadow-sky-600/20"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            <Stethoscope className="w-4 h-4" />
            <span>New Examination</span>
          </Link>

          <Link
            href="/dashboard"
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
              pathname === "/dashboard"
                ? "bg-sky-600 text-white shadow-sm shadow-sky-600/20"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            <LayoutDashboard className="w-4 h-4" />
            <span>Doctor Review Queue</span>
          </Link>
        </nav>

        {/* AI System Status */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs">
          {backendStatus === "connected" ? (
            <>
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-emerald-700 font-medium">AI Engine Online</span>
              {simulationMode && (
                <span className="text-slate-500 text-[11px] border-l border-slate-300 pl-2">
                  Validation Mode
                </span>
              )}
            </>
          ) : backendStatus === "offline" ? (
            <>
              <div className="w-2 h-2 rounded-full bg-rose-500" />
              <span className="text-rose-700 font-medium">Backend Offline</span>
            </>
          ) : (
            <>
              <div className="w-2 h-2 rounded-full bg-amber-500 animate-spin" />
              <span className="text-amber-700">Connecting...</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

"""
main.py — RetinaScan AI FastAPI application entry point.
Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import backend.config as cfg
from backend.api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("retinascan")

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RetinaScan AI — Explainable DR Screening",
    description=(
        "AI-powered Diabetic Retinopathy screening with Grad-CAM explainability. "
        "Grades retinal images on the 5-level ICDR scale and provides referral recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow Next.js frontend on localhost:3000) ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://retinascan.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    mode = "SIMULATION" if cfg.SIMULATION_MODE else "REAL INFERENCE"
    logger.info(f"RetinaScan AI started — Mode: {mode}")
    if cfg.SIMULATION_MODE:
        logger.warning(
            "Running in SIMULATION mode. "
            "Train models and add weights to backend/models/weights/ for real inference."
        )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "simulation_mode": cfg.SIMULATION_MODE,
        "device": cfg.DEVICE,
    }


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "RetinaScan AI",
        "description": "Explainable AI for Diabetic Retinopathy Screening",
        "docs": "/docs",
        "health": "/health",
    }

# RetinaScan AI — Explainable AI for Diabetic Retinopathy Screening

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16.3-black.svg?style=flat&logo=next.js)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38B2AC.svg?style=flat&logo=tailwind-css)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, clinically-validated, explainable artificial intelligence (XAI) screening platform for **Diabetic Retinopathy (DR)**. Designed for tele-ophthalmology workflows, rural Primary Health Centres (PHCs), and district hospitals to triage fundus photographs, detect early sight-threatening retinopathy, provide transparent visual explanations (Grad-CAM), and assist clinicians with calibrated referral recommendations.

---

## Table of Contents

- [Overview & Clinical Significance](#overview--clinical-significance)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [ICDR Grading Scale & Clinical Actions](#icdr-grading-scale--clinical-actions)
- [Tech Stack](#tech-stack)
- [Project Directory Structure](#project-directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Backend Setup (FastAPI)](#2-backend-setup-fastapi)
  - [3. Frontend Setup (Next.js)](#3-frontend-setup-nextjs)
  - [4. Verify Installation](#4-verify-installation)
- [Environment Configuration](#environment-configuration)
- [API Reference](#api-reference)
- [Model Training & Dataset Manifest](#model-training--dataset-manifest)
- [Human-in-the-Loop & Audit Workflow](#human-in-the-loop--audit-workflow)
- [Clinical Disclaimer](#clinical-disclaimer)
- [References & Citations](#references--citations)

---

## Overview & Clinical Significance

Diabetic Retinopathy (DR) is the primary cause of preventable blindness among working-age adults globally. In India and low-resource settings, the ratio of ophthalmologists to patients is severely constrained, resulting in delayed diagnoses until irreversible vision loss occurs.

**RetinaScan AI** bridges this gap by offering:
- **Instant Quality Triage**: Filters ungradable or out-of-focus fundus captures before clinical review.
- **5-Level ICDR Classification**: Accurately classifies severity from Level 0 (No DR) through Level 4 (Proliferative DR).
- **Explainability (XAI)**: Generates high-fidelity Grad-CAM saliency heatmaps highlighting microaneurysms, exudates, and hemorrhages.
- **Clinician Decision Support**: Generates automated downloadable medical PDF reports and flags uncertain/borderline cases into a dedicated Human-in-the-Loop Review Queue.

---

## System Architecture

```text
                                  +---------------------------------------+
                                  |     Clinician Web App (Next.js 16)    |
                                  |  - Fundus Image Upload                |
                                  |  - Grad-CAM Interactive Heatmaps      |
                                  |  - Review Queue & District Dashboard  |
                                  +-------------------+-------------------+
                                                      |
                                               HTTP / REST API
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         FastAPI Backend Engine        |
                                  +-------------------+-------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
                    v                                 v                                 v
        +-----------------------+         +-----------------------+         +-----------------------+
        |       MODULE 1        |         |       MODULE 2        |         |       MODULE 3        |
        |    Quality Assessor   | ------> |  Lesion Segmentation  | ------> |    DR Grading Engine  |
        | - Laplacian Sharpness | (Pass)  | - U-Net Retinal Vessel|         | - EfficientNet Backbone|
        | - Illumination / FOV  |         | - LoG + SVM MAs       |         | - Temperature Scaling |
        | - CLAHE Enhancement   |         | - Hard/Soft Exudates  |         | - Ordinal 0-4 Scale   |
        +-----------------------+         +-----------------------+         +-----------------------+
                    | (Reject)                                                          |
                    v                                                                   v
          [Recapture Feedback]                                              +-----------------------+
                                                                            |       MODULE 4        |
                                                                            |    Explainability     |
                                                                            | - Grad-CAM Overlays   |
                                                                            | - Evidence Mapping    |
                                                                            | - PDF ReportLab Engine|
                                                                            +-----------------------+
```

---

## Key Features

### 1. Automated Image Quality Assessment
- Evaluates **Sharpness** (Laplacian variance), **Illumination** (green channel distribution), and **Field-of-View** (circular retina boundary detection).
- Applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** and morphological illumination normalization.
- Rejects ungradable images instantly with actionable clinician feedback.

### 2. Microaneurysm & Lesion Localization
- Detects microaneurysms using multi-scale **Laplacian of Gaussian (LoG)** blob filters matched to 10–100 micron retinal structures.
- Removes false positives via a 12-feature SVM classifier.
- Quantifies hard exudates and evaluates the clinical **4-2-1 rule** for severe non-proliferative retinopathy.

### 3. Calibrated Deep Learning Grading
- Deep neural network backbone (EfficientNet-B4) trained on Indian and international cohorts.
- Calibrated class probabilities using **Temperature Scaling** to eliminate overconfident predictions.
- **Simulation Mode**: Automatic zero-crash fallback for testing and development when model weights are not loaded.

### 4. Transparent Explainability (Grad-CAM)
- Gradient-weighted Class Activation Mapping directly shows which retinal regions drove the AI's diagnostic decision.
- Interactive opacity blending slider and lesion coordinate indicators in the web UI.

### 5. Clinician Review Queue & Analytics Dashboard
- Automatically routes cases with low confidence (<65%) or severe grades (Levels 3–4) to an ophthalmologist review queue.
- Tracks specialist agreement, overrides, and audit trails.
- District-wide tele-ophthalmology statistics and throughput breakdown.

### 6. Automated Diagnostic PDF Generation
- Generates downloadable, clinical-grade PDF reports (ReportLab) containing patient metadata, original fundus photographs, Grad-CAM overlays, detected lesions, and standardized referral guidelines.

---

## ICDR Grading Scale & Clinical Actions

| Grade | Disease State | Clinical Signs | Referral Action | Urgency |
|:---:|:---|:---|:---|:---|
| **0** | **No DR** | No microaneurysms or vascular abnormalities | Annual routine rescreening | Routine (12 months) |
| **1** | **Mild NPDR** | Microaneurysms only | Annual screening & glycemic control | Routine (9–12 months) |
| **2** | **Moderate NPDR** | More than MAs, but less than severe NPDR (mild exudates/hemorrhages) | Refer to ophthalmology / retina clinic | Within 1–2 months |
| **3** | **Severe NPDR** | 4-2-1 Rule: >20 hemorrhages in 4 quadrants, venous beading in ≥2, or IRMA in ≥1 | Urgent vitreoretinal specialist referral | Within 1–2 weeks |
| **4** | **Proliferative DR (PDR)**| Neovascularization, vitreous/preretinal hemorrhage | Immediate specialist intervention | Immediate (<48 hours) |

---

## Tech Stack

### Frontend
- **Framework**: [Next.js 16 (App Router)](https://nextjs.org) + [React 19](https://react.dev)
- **Language**: TypeScript
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com)
- **Icons**: [Lucide React](https://lucide.dev)

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com)
- **Server**: [Uvicorn](https://www.uvicorn.org)
- **Computer Vision & ML**: PyTorch 2.4, Torchvision, OpenCV (`opencv-python-headless`), Scikit-Image, SciPy, NumPy
- **Reporting**: ReportLab (PDF generation)
- **Configuration & Validation**: Pydantic v2, Python-dotenv

---

## Project Directory Structure

```text
.
├── backend/
│   ├── api/
│   │   ├── routes.py            # FastAPI endpoints (analyze, queue, reports, stats)
│   │   └── schemas.py           # Pydantic request/response models
│   ├── models/
│   │   ├── unet.py              # Retinal vessel segmentation model
│   │   └── weights/             # Checkpoint weights (.pth, .pkl, .json)
│   ├── modules/
│   │   ├── dr_grader.py         # EfficientNet grading & temperature calibration
│   │   ├── explainability.py    # Grad-CAM, evidence mapping & PDF generator
│   │   ├── lesion_segmentor.py  # Vessel, MA (LoG + SVM), and exudate segmentation
│   │   └── quality_assessor.py  # Sharpness, illumination & CLAHE pre-screening
│   ├── training/
│   │   └── train_grader.py      # PyTorch training pipeline with ordinal loss
│   ├── config.py                # Environment configuration and thresholds
│   ├── main.py                  # FastAPI application entry point
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── dashboard/       # District analytics & review queue page
│   │   │   ├── result/[id]/     # Interactive screening report & Grad-CAM viewer
│   │   │   ├── layout.tsx       # Root layout with clinical navigation header
│   │   │   └── page.tsx         # Main fundus image upload & triage portal
│   │   ├── components/
│   │   │   ├── ConfidenceBar.tsx
│   │   │   ├── GradCAMViewer.tsx
│   │   │   ├── GradeGauge.tsx
│   │   │   ├── ImageUploader.tsx
│   │   │   ├── LesionEvidenceCard.tsx
│   │   │   ├── Navbar.tsx
│   │   │   ├── ReviewModal.tsx
│   │   │   └── ReviewQueue.tsx
│   │   └── lib/
│   │       └── api.ts           # Client API client with simulation fallback
│   ├── package.json
│   └── tailwind.config.ts
├── data/
│   ├── train/                   # Training datasets (IDRiD, APTOS, Messidor)
│   └── outputs/reports/         # Generated clinical PDFs and diagnostic outputs
├── AGENT.md                     # Detailed clinical system specifications
├── .env                         # Project environment variables
└── README.md                    # Project documentation
```

---

## Getting Started

### Prerequisites
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Node.js**: >= 18.x (v20+ recommended) and `npm`

---

### 1. Clone Repository

```bash
git clone https://github.com/Pradeeeeeeeep/AI-for-Diabetic-Retinopathy-Screening.git
cd AI-for-Diabetic-Retinopathy-Screening
```

---

### 2. Backend Setup (FastAPI)

1. Create and activate a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Launch the FastAPI server:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   > **Note**: If weights in `backend/models/weights/` are not yet trained or downloaded, the backend automatically runs in **SIMULATION mode**, allowing full UI testing and pipeline validation.

4. Check backend status:
   - Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Frontend Setup (Next.js)

1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

4. Open the application in your browser:
   👉 **[http://localhost:3000](http://localhost:3000)**

---

### 4. Verify Installation

- Upload a sample fundus image on [http://localhost:3000](http://localhost:3000).
- View the automated quality score, 5-level ICDR grading, confidence distribution, and Grad-CAM overlay.
- Navigate to the **Dashboard** ([http://localhost:3000/dashboard](http://localhost:3000/dashboard)) to review flagged cases and triage statistics.
- Test PDF generation by clicking **Download Medical PDF**.

---

## Environment Configuration

Configuration is managed via `.env` in the root directory:

```ini
# Model Weights (leaves empty or unplaced for Simulation Mode)
MODEL_GRADER_PATH=backend/models/weights/dr_grader_india.pth
MODEL_VESSEL_PATH=backend/models/weights/vessel_unet.pth
MODEL_SVM_PATH=backend/models/weights/svm_ma_classifier.pkl
TEMPERATURE_PATH=backend/models/weights/temperature.json

# Database
DATABASE_URL=sqlite:///./retinascan.db

# File Storage
STORAGE_BACKEND=local
STORAGE_PATH=data/outputs/reports

# Inference Device
USE_GPU=false
DEVICE=cpu

# Logging
LOG_LEVEL=INFO

# Frontend (optional for custom deployment URLs)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## API Reference

### Key Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/analyze` | Multi-part fundus upload. Runs quality check, lesion detection, grading, and Grad-CAM. |
| `GET` | `/api/report/{id}` | Retrieves full diagnostic JSON report for a given study ID. |
| `GET` | `/api/report/{id}/pdf` | Generates and downloads the clinical medical PDF. |
| `GET` | `/api/queue` | Returns list of pending cases requiring ophthalmologist verification. |
| `POST` | `/api/queue/{id}/review` | Submits clinician review (agree/override, notes, doctor ID). |
| `GET` | `/api/stats` | Aggregated screening statistics, severity distribution, and review queue counts. |
| `GET` | `/health` | System health, GPU availability, and model operational mode. |

#### Example: Analyzing a Fundus Image

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@sample_fundus.jpg" \
  -F "patient_id=PT-2026-0042" \
  -F "eye=right"
```

---

## Model Training & Dataset Manifest

The system is architected to train on diverse retinal screening datasets, with emphasis on Indian demographic representations:

- **IDRiD** (Indian Diabetic Retinopathy Image Dataset): 516 fundus images acquired at an eye clinic in Nanded, Maharashtra. Includes pixel-level annotations for microaneurysms, hemorrhages, and hard/soft exudates.
- **APTOS 2019 Blindness Detection**: 3,662 fundus photographs across rural and urban India.
- **Messidor-2**: 1,200 images with adjudications for DR and Diabetic Macular Edema (DME).
- **EyePACS**: 35,126 training fundus images for robust foundational feature representation.

### Training the Grader

```bash
python backend/training/train_grader.py \
  --data_dir data/train \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --output_dir backend/models/weights
```

---

## Human-in-the-Loop & Audit Workflow

To uphold clinical safety and comply with international medical device standards:

1. **Automated Triage**: Clear Level 0/1 cases with confidence ≥85% can be scheduled for routine screening.
2. **Flagged for Human Review**:
   - Any case with model confidence below 65%.
   - Borderline quality images that required CLAHE enhancement.
   - Any case classified as severe (Level 3) or proliferative (Level 4).
3. **Audit Trail**: Every clinician review records the ophthalmologist's ID, confirmation status, modified grade (if overridden), timestamp, and clinical notes for continuous quality auditing.

---

## Clinical Disclaimer

> **IMPORTANT**: RetinaScan AI is designed as a **Clinical Decision Support System (CDSS)** to assist licensed eye care practitioners, ophthalmologists, and trained healthcare personnel. It is not intended as an autonomous diagnostic device. All referable and suspicious findings must be clinically evaluated and corroborated by a certified ophthalmologist.

---

## References & Citations

1. **ICDR Severity Scale**: Wilkinson CP, et al. *Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales.* Ophthalmology. 2003;110(9):1677-1682.
2. **Grad-CAM**: Selvaraju RR, et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.
3. **IDRiD Benchmark**: Porwal P, et al. *Indian Diabetic Retinopathy Image Dataset (IDRiD): A database for diabetic retinopathy screening research.* Data 2018, 3(3), 25.
4. **Calibration via Temperature Scaling**: Guo C, Pleiss G, Sun Y, Weinberger KQ. *On Calibration of Modern Neural Networks.* ICML 2017.
5. **AI for Retinopathy Screening**: Gulshan V, et al. *Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs.* JAMA. 2016;316(22):2402–2410.

---

**Developed for AI-driven Healthcare Screening & Tele-Ophthalmology.**

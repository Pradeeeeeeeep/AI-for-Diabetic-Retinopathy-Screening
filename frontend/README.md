# RetinaScan AI — Frontend Web Application

The modern web user interface for **RetinaScan AI**, built with Next.js 16, React 19, TypeScript, and Tailwind CSS v4.

For the full system architecture, backend setup, and clinical documentation, please refer to the [Root README](../README.md).

---

## Features

- **Fundus Upload & Triage Portal**: Instant client-side validation, drag-and-drop fundus photo upload, eye selection (OD/OS), and patient ID assignment.
- **Interactive Grad-CAM Viewer**: Real-time opacity blending slider to inspect AI saliency heatmaps against fundus vasculature and micro-lesions.
- **Lesion Evidence Breakdown**: Microaneurysm counter, exudate detection metrics, and clinical 4-2-1 rule evaluation.
- **Human-in-the-Loop Review Queue**: Ophthalmologist case verification, agreement tracking, diagnosis override, and clinical notes.
- **District Screening Analytics**: Metrics on screened patients, referable DR rates, and pending review counts.
- **Diagnostic PDF Generation**: One-click download of clinical report sheets.

---

## Getting Started

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment (Optional)

Create a `.env.local` file if you want to connect to a non-default backend endpoint:

```ini
NEXT_PUBLIC_API_URL=http://localhost:8000
```

*(If the backend is offline, the frontend automatically falls back to an integrated simulation demo so you can test all UI flows.)*

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Build for Production

```bash
npm run build
npm run start
```

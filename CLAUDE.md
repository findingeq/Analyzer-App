# Project Rules

## Development
- Work directly on `main` branch
- Push changes to `main` for automatic Railway deployment
- Railway deployment config: `railway.json`, `nixpacks.toml`, `Procfile`

## Architecture
- **Frontend:** React + Vite + TypeScript + Tailwind + ECharts (`/web`)
- **Backend:** Python FastAPI (`/api`)
- **Deployment:** Railway (single service serving both)

## React Frontend Features (Implemented)
- [x] Dashboard layout with sidebar and chart
- [x] CSV file upload (local)
- [x] Cloud session loading (Firebase)
- [x] Data source toggle (Local/Cloud)
- [x] VE binned line with gradient
- [x] Breath scatter dots
- [x] Interval shading (colored by status: green/yellow/red)
- [x] 3-segment slope lines (segment1 + segment2 + segment3)
- [x] Slope annotations (%/min labels)
- [x] CUSUM line with color change (green→red at alarm)
- [x] Below-chart interval metrics (aligned with intervals)
- [x] Click-to-zoom on intervals
- [x] Header metrics (run type format, cumulative drift)
- [x] Reset zoom button

## Backend API Endpoints
- `POST /api/files/parse` - Parse CSV metadata
- `POST /api/files/detect-intervals` - Detect intervals from power data
- `POST /api/analysis/run` - Run CUSUM analysis
- `GET /api/sessions` - List cloud sessions (Firebase)
- `GET /api/sessions/{id}` - Get session CSV content
- `POST /api/upload` - Upload CSV to cloud

## General Behavior
1. When user provides any instructions, carefully evaluate whether the instructions are logical and appropriate given the overall purpose of the app. If the instructions are not clear, ask clarifying questions.
2. Do not code anything automatically. Always ask for permission first before coding.
3. When user asks to change a specific part of the code, do not alter any other parts of the code without first confirming with user he wants those other parts to be altered.

## Interval Detection
- Single interval = VT1, Multiple intervals = VT2
- Always use grid-fitting to detect intervals that extend to end of recording
- All recoveries and intervals are the same duration across a single run (respectively)

## Signal Processing
- 3-stage hybrid filtering: Rolling median → Time binning → Hampel filter
- 2-hinge robust regression model for VE drift analysis

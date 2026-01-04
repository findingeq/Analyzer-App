"""
FastAPI Backend for VT Threshold Analyzer

Provides:
- CSV file parsing and interval detection
- CUSUM analysis endpoints
- Cloud session management (Firebase Storage)
- Static file serving for React frontend
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import routers
from .routers import files_router, analysis_router, calibration_router
from .routers.calibration import set_firebase_bucket
from .services.csv_parser import parse_csv_auto, detect_csv_format

app = FastAPI(
    title="VT Threshold Analyzer API",
    description="Backend API for respiratory data analysis",
    version="2.0.0"
)

# Include routers
app.include_router(files_router)
app.include_router(analysis_router)
app.include_router(calibration_router)

# Allow requests from iOS app and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase (optional for local development)
# In Cloud Run, credentials are passed via environment variable
# Locally, use the firebase-credentials.json file
bucket = None
FIREBASE_ENABLED = False

try:
    import firebase_admin
    from firebase_admin import credentials, storage

    if os.environ.get("FIREBASE_CREDENTIALS"):
        # Cloud Run: credentials passed as JSON string in env var
        cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS"])
        cred = credentials.Certificate(cred_dict)
        BUCKET_NAME = os.environ.get("FIREBASE_BUCKET", "vt-threshold-analyzer.firebasestorage.app")
        firebase_admin.initialize_app(cred, {"storageBucket": BUCKET_NAME})
        bucket = storage.bucket()
        FIREBASE_ENABLED = True
    else:
        # Local development: use credentials file if available
        cred_path = os.path.join(os.path.dirname(__file__), "..", "firebase-credentials.json")
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            BUCKET_NAME = os.environ.get("FIREBASE_BUCKET", "vt-threshold-analyzer.firebasestorage.app")
            firebase_admin.initialize_app(cred, {"storageBucket": BUCKET_NAME})
            bucket = storage.bucket()
            FIREBASE_ENABLED = True
        else:
            print("⚠️  Firebase credentials not found. Cloud storage endpoints disabled.")
            print("   Analysis endpoints will still work for local development.")
except ImportError:
    print("⚠️  firebase-admin not installed. Cloud storage endpoints disabled.")

# Set Firebase bucket reference for calibration router
set_firebase_bucket(bucket, FIREBASE_ENABLED)


class UploadRequest(BaseModel):
    filename: str
    csv_content: str


class UploadResponse(BaseModel):
    success: bool
    session_id: str
    message: str


class SessionSummary(BaseModel):
    """Summary metadata extracted from CSV for quick loading."""
    date: Optional[str] = None
    run_type: Optional[str] = None  # "VT1" or "VT2"
    vt1_threshold: Optional[float] = None
    vt2_threshold: Optional[float] = None
    speed: Optional[float] = None  # Average speed in mph
    duration_seconds: Optional[float] = None
    num_intervals: Optional[int] = None
    interval_duration_min: Optional[float] = None
    recovery_duration_min: Optional[float] = None
    intensity: Optional[str] = None  # "Moderate", "Heavy", "Severe"
    avg_pace_min_per_mile: Optional[float] = None
    # Analysis results (populated after analysis is run)
    observed_sigma_pct: Optional[float] = None  # Observed sigma % from MADSD
    observed_drift_pct: Optional[float] = None  # Observed drift % per minute
    exclude_from_calibration: bool = False  # Whether to exclude from ML calibration


class SessionInfo(BaseModel):
    session_id: str
    filename: str
    uploaded_at: str
    size_bytes: int
    summary: Optional[SessionSummary] = None


@app.get("/api/health")
def health():
    """Health check endpoint for the API."""
    return {"status": "ok", "service": "VT Threshold Analyzer API"}


def _extract_session_summary(csv_content: str) -> dict:
    """Extract summary metadata from CSV content for quick loading."""
    summary = {}

    try:
        breath_df, metadata, power_df, run_params = parse_csv_auto(csv_content)

        # Extract date from metadata or filename
        if 'date' in metadata:
            summary['date'] = metadata['date']

        # Get run parameters from iOS CSV
        if run_params:
            if 'run_type' in run_params:
                summary['run_type'] = run_params['run_type'].value if hasattr(run_params['run_type'], 'value') else str(run_params['run_type'])
            if 'vt1_threshold' in run_params:
                summary['vt1_threshold'] = run_params['vt1_threshold']
            if 'vt2_threshold' in run_params:
                summary['vt2_threshold'] = run_params['vt2_threshold']
            if 'speeds' in run_params and run_params['speeds']:
                summary['speed'] = sum(run_params['speeds']) / len(run_params['speeds'])
            if 'num_intervals' in run_params:
                summary['num_intervals'] = run_params['num_intervals']
            if 'interval_duration' in run_params:
                summary['interval_duration_min'] = run_params['interval_duration']
            if 'recovery_duration' in run_params:
                summary['recovery_duration_min'] = run_params['recovery_duration']

        # Calculate duration from breath data
        if len(breath_df) > 0 and 'breath_time' in breath_df.columns:
            summary['duration_seconds'] = float(breath_df['breath_time'].max())

        # Calculate average pace (min/mile) from speed (mph)
        if summary.get('speed') and summary['speed'] > 0:
            summary['avg_pace_min_per_mile'] = 60.0 / summary['speed']

        # Calculate intensity based on VE vs VT1/VT2 thresholds
        # This requires analyzing VE data against thresholds
        vt1 = summary.get('vt1_threshold')
        vt2 = summary.get('vt2_threshold')

        if vt1 and vt2 and 've_raw' in breath_df.columns:
            ve_values = breath_df['ve_raw'].dropna().values

            # Exclude recovery periods if phase column exists
            if 'phase' in breath_df.columns:
                workout_mask = breath_df['phase'].str.lower().str.contains('workout', na=False)
                ve_values = breath_df.loc[workout_mask, 've_raw'].dropna().values

            if len(ve_values) > 0:
                # Count time in each zone (using bin size of ~5 seconds)
                moderate_count = sum(ve_values <= vt1)
                heavy_count = sum((ve_values > vt1) & (ve_values <= vt2))
                severe_count = sum(ve_values > vt2)

                # Determine intensity by majority time
                max_count = max(moderate_count, heavy_count, severe_count)
                if max_count == moderate_count:
                    summary['intensity'] = 'Moderate'
                elif max_count == heavy_count:
                    summary['intensity'] = 'Heavy'
                else:
                    summary['intensity'] = 'Severe'

    except Exception as e:
        # If parsing fails, return empty summary
        print(f"Warning: Could not extract session summary: {e}")

    return summary


@app.post("/api/upload", response_model=UploadResponse)
def upload_csv(request: UploadRequest):
    """
    Receive CSV content from iOS app and store in Firebase Storage.
    Extracts summary metadata for quick loading on startup screen.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")
    try:
        # Generate unique session ID using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{request.filename}"

        # Upload CSV to Firebase Storage
        csv_blob = bucket.blob(f"sessions/{session_id}")
        csv_blob.upload_from_string(request.csv_content, content_type="text/csv")

        # Extract summary metadata from CSV
        summary = _extract_session_summary(request.csv_content)

        # Upload metadata with summary
        metadata = {
            "filename": request.filename,
            "uploaded_at": datetime.now().isoformat(),
            "size_bytes": len(request.csv_content),
            "summary": summary
        }
        meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")
        meta_blob.upload_from_string(json.dumps(metadata), content_type="application/json")

        return UploadResponse(
            success=True,
            session_id=session_id,
            message=f"Successfully uploaded {request.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions", response_model=list[SessionInfo])
def list_sessions():
    """
    List all uploaded sessions from Firebase Storage with summary metadata.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")
    sessions = []

    # List all metadata files
    blobs = bucket.list_blobs(prefix="sessions/")
    meta_blobs = [b for b in blobs if b.name.endswith(".meta.json")]

    for meta_blob in sorted(meta_blobs, key=lambda b: b.name, reverse=True):
        try:
            metadata = json.loads(meta_blob.download_as_text())
            session_id = meta_blob.name.replace("sessions/", "").replace(".meta.json", "")

            # Parse summary if available
            summary_data = metadata.get("summary")
            summary = None
            if summary_data:
                summary = SessionSummary(
                    date=summary_data.get("date"),
                    run_type=summary_data.get("run_type"),
                    vt1_threshold=summary_data.get("vt1_threshold"),
                    vt2_threshold=summary_data.get("vt2_threshold"),
                    speed=summary_data.get("speed"),
                    duration_seconds=summary_data.get("duration_seconds"),
                    num_intervals=summary_data.get("num_intervals"),
                    interval_duration_min=summary_data.get("interval_duration_min"),
                    recovery_duration_min=summary_data.get("recovery_duration_min"),
                    intensity=summary_data.get("intensity"),
                    avg_pace_min_per_mile=summary_data.get("avg_pace_min_per_mile"),
                    observed_sigma_pct=summary_data.get("observed_sigma_pct"),
                    observed_drift_pct=summary_data.get("observed_drift_pct"),
                    exclude_from_calibration=summary_data.get("exclude_from_calibration", False)
                )

            sessions.append(SessionInfo(
                session_id=session_id,
                filename=metadata.get("filename", session_id),
                uploaded_at=metadata.get("uploaded_at", ""),
                size_bytes=metadata.get("size_bytes", 0),
                summary=summary
            ))
        except Exception:
            continue

    return sessions


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """
    Get CSV content for a specific session from Firebase Storage.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")
    csv_blob = bucket.blob(f"sessions/{session_id}")

    if not csv_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    csv_content = csv_blob.download_as_text()
    return {"session_id": session_id, "csv_content": csv_content}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user_id: Optional[str] = None):
    """
    Delete a session and its metadata from Firebase Storage.
    If the session contributed to calibration, recalculates calibration
    from remaining sessions.
    """
    from .services.calibration import recalculate_calibration_from_contributions
    from .routers.calibration import _load_calibration_state, _save_calibration_state

    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")

    csv_blob = bucket.blob(f"sessions/{session_id}")
    meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")

    if not csv_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if this session contributed to calibration
    deleted_contribution = None
    if meta_blob.exists():
        try:
            metadata = json.loads(meta_blob.download_as_text())
            deleted_contribution = metadata.get('summary', {}).get('calibration_contribution')
        except Exception:
            pass

    # Delete the session files
    csv_blob.delete()
    if meta_blob.exists():
        meta_blob.delete()

    # If this session contributed to calibration and we have a user_id, recalculate
    calibration_updated = False
    if deleted_contribution and deleted_contribution.get('contributed') and user_id:
        try:
            # Get all remaining sessions and their contributions
            remaining_contributions = []
            for blob in bucket.list_blobs(prefix="sessions/"):
                if blob.name.endswith('.meta.json') and blob.name != f"sessions/{session_id}.meta.json":
                    try:
                        meta = json.loads(blob.download_as_text())
                        contrib = meta.get('summary', {}).get('calibration_contribution')
                        if contrib and contrib.get('contributed'):
                            remaining_contributions.append(contrib)
                    except Exception:
                        continue

            # Recalculate calibration from remaining contributions
            new_state = recalculate_calibration_from_contributions(remaining_contributions)

            # Preserve VE thresholds from existing state (they're user-set, not recalculated)
            existing_state = _load_calibration_state(user_id)
            new_state.vt1_ve = existing_state.vt1_ve
            new_state.vt2_ve = existing_state.vt2_ve
            new_state.enabled = existing_state.enabled

            # Save updated calibration
            _save_calibration_state(user_id, new_state)
            calibration_updated = True
        except Exception as e:
            print(f"Warning: Failed to recalculate calibration after session deletion: {e}")

    return {
        "success": True,
        "message": f"Deleted {session_id}",
        "calibration_updated": calibration_updated
    }


@app.get("/api/storage/list")
def list_all_blobs():
    """
    Debug endpoint: List all blobs in Firebase Storage to see folder structure.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")

    all_blobs = []
    for blob in bucket.list_blobs():
        all_blobs.append({
            "name": blob.name,
            "size": blob.size,
            "updated": blob.updated.isoformat() if blob.updated else None
        })

    return {"blobs": all_blobs, "total": len(all_blobs)}


@app.get("/api/storage/download/{path:path}")
def download_blob(path: str):
    """
    Debug endpoint: Download any blob by path.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")

    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"Blob not found: {path}")

    content = blob.download_as_text()
    lines = content.split('\n')

    return {
        "path": path,
        "first_20_lines": lines[:20],
        "total_lines": len(lines),
        "total_bytes": len(content)
    }


@app.get("/api/sessions/{session_id}/debug")
def debug_session(session_id: str):
    """
    Debug endpoint: Get first 20 lines of a cloud session CSV for comparison.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")
    csv_blob = bucket.blob(f"sessions/{session_id}")
    meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")

    if not csv_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    csv_content = csv_blob.download_as_text()
    lines = csv_content.split('\n')[:20]

    # Get metadata too
    metadata = None
    if meta_blob.exists():
        metadata = json.loads(meta_blob.download_as_text())

    return {
        "session_id": session_id,
        "first_20_lines": lines,
        "total_lines": len(csv_content.split('\n')),
        "total_bytes": len(csv_content),
        "metadata": metadata
    }


class CalibrationContributionData(BaseModel):
    """Calibration contribution data from a session."""
    contributed: bool
    run_type: Optional[str] = None
    sigma_pct: Optional[float] = None


class UpdateSessionAnalysisRequest(BaseModel):
    """Request to update session with analysis results."""
    session_id: str
    observed_sigma_pct: Optional[float] = None
    observed_drift_pct: Optional[float] = None
    calibration_contribution: Optional[CalibrationContributionData] = None


class UpdateSessionCalibrationRequest(BaseModel):
    """Request to update session calibration exclusion."""
    exclude_from_calibration: bool


@app.post("/api/sessions/{session_id}/analysis")
def update_session_analysis(session_id: str, request: UpdateSessionAnalysisRequest):
    """
    Update a session's metadata with analysis results (sigma %, drift %).
    Called after analysis is run to store observed values.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")

    meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")

    if not meta_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Load existing metadata
        metadata = json.loads(meta_blob.download_as_text())

        # Update summary with analysis results
        if "summary" not in metadata:
            metadata["summary"] = {}

        if request.observed_sigma_pct is not None:
            metadata["summary"]["observed_sigma_pct"] = request.observed_sigma_pct
        if request.observed_drift_pct is not None:
            metadata["summary"]["observed_drift_pct"] = request.observed_drift_pct
        if request.calibration_contribution is not None:
            metadata["summary"]["calibration_contribution"] = {
                "contributed": request.calibration_contribution.contributed,
                "run_type": request.calibration_contribution.run_type,
                "sigma_pct": request.calibration_contribution.sigma_pct,
            }

        # Save updated metadata
        meta_blob.upload_from_string(json.dumps(metadata), content_type="application/json")

        return {
            "success": True,
            "session_id": session_id,
            "message": "Analysis results saved"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")


@app.post("/api/sessions/{session_id}/calibration")
def update_session_calibration(session_id: str, request: UpdateSessionCalibrationRequest):
    """
    Update a session's calibration exclusion status.
    When excluded, the session won't contribute to ML calibration updates.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")

    meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")

    if not meta_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Load existing metadata
        metadata = json.loads(meta_blob.download_as_text())

        # Update summary with exclusion status
        if "summary" not in metadata:
            metadata["summary"] = {}

        metadata["summary"]["exclude_from_calibration"] = request.exclude_from_calibration

        # Save updated metadata
        meta_blob.upload_from_string(json.dumps(metadata), content_type="application/json")

        return {
            "success": True,
            "session_id": session_id,
            "exclude_from_calibration": request.exclude_from_calibration,
            "message": "Calibration exclusion updated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")


# =============================================================================
# Static File Serving for React Frontend
# =============================================================================

# Path to the built React frontend
STATIC_DIR = Path(__file__).parent.parent / "web" / "dist"

# Serve static files if the build exists
if STATIC_DIR.exists():
    # Mount static assets (JS, CSS, etc.)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Root route - serve React app
    @app.get("/")
    async def serve_root():
        """Serve the React SPA at root."""
        return FileResponse(STATIC_DIR / "index.html")

    # Catch-all route for SPA - must be last
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        # Don't serve index.html for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        # Check if requesting a specific file
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise serve index.html for SPA routing
        return FileResponse(STATIC_DIR / "index.html")
else:
    # No frontend build - serve API status at root
    @app.get("/")
    def root():
        return {"status": "ok", "service": "VT Threshold Analyzer API", "frontend": "not built"}

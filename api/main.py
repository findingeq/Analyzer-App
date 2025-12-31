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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import routers
from .routers import files_router, analysis_router

app = FastAPI(
    title="VT Threshold Analyzer API",
    description="Backend API for respiratory data analysis",
    version="2.0.0"
)

# Include routers
app.include_router(files_router)
app.include_router(analysis_router)

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


class UploadRequest(BaseModel):
    filename: str
    csv_content: str


class UploadResponse(BaseModel):
    success: bool
    session_id: str
    message: str


class SessionInfo(BaseModel):
    session_id: str
    filename: str
    uploaded_at: str
    size_bytes: int


@app.get("/api/health")
def health():
    """Health check endpoint for the API."""
    return {"status": "ok", "service": "VT Threshold Analyzer API"}


@app.post("/api/upload", response_model=UploadResponse)
def upload_csv(request: UploadRequest):
    """
    Receive CSV content from iOS app and store in Firebase Storage.
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

        # Upload metadata
        metadata = {
            "filename": request.filename,
            "uploaded_at": datetime.now().isoformat(),
            "size_bytes": len(request.csv_content)
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
    List all uploaded sessions from Firebase Storage.
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
            sessions.append(SessionInfo(
                session_id=session_id,
                filename=metadata.get("filename", session_id),
                uploaded_at=metadata.get("uploaded_at", ""),
                size_bytes=metadata.get("size_bytes", 0)
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
def delete_session(session_id: str):
    """
    Delete a session and its metadata from Firebase Storage.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase storage not configured")
    csv_blob = bucket.blob(f"sessions/{session_id}")
    meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")

    if not csv_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    csv_blob.delete()
    if meta_blob.exists():
        meta_blob.delete()

    return {"success": True, "message": f"Deleted {session_id}"}


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

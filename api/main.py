"""
FastAPI Upload Endpoint for VT Threshold Analyzer
Receives CSV uploads from iOS app and stores in Firebase Storage
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, storage

app = FastAPI(title="VT Threshold Analyzer API")

# Allow requests from iOS app and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase
# In Cloud Run, credentials are passed via environment variable
# Locally, use the firebase-credentials.json file
if os.environ.get("FIREBASE_CREDENTIALS"):
    # Cloud Run: credentials passed as JSON string in env var
    cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS"])
    cred = credentials.Certificate(cred_dict)
else:
    # Local development: use credentials file
    cred_path = os.path.join(os.path.dirname(__file__), "..", "firebase-credentials.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
    else:
        raise RuntimeError("Firebase credentials not found. Set FIREBASE_CREDENTIALS env var or provide firebase-credentials.json")

# Get bucket name from env or use default pattern
BUCKET_NAME = os.environ.get("FIREBASE_BUCKET", "vt-threshold-analyzer.firebasestorage.app")

firebase_admin.initialize_app(cred, {"storageBucket": BUCKET_NAME})
bucket = storage.bucket()


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


@app.get("/")
def root():
    return {"status": "ok", "service": "VT Threshold Analyzer API"}


@app.post("/api/upload", response_model=UploadResponse)
def upload_csv(request: UploadRequest):
    """
    Receive CSV content from iOS app and store in Firebase Storage.
    """
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
    csv_blob = bucket.blob(f"sessions/{session_id}")
    meta_blob = bucket.blob(f"sessions/{session_id}.meta.json")

    if not csv_blob.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    csv_blob.delete()
    if meta_blob.exists():
        meta_blob.delete()

    return {"success": True, "message": f"Deleted {session_id}"}

"""
FastAPI Upload Endpoint for VT Threshold Analyzer
Receives CSV uploads from iOS app and stores locally
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import json

app = FastAPI(title="VT Threshold Analyzer API")

# Allow requests from iOS app and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local storage directory
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


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
    Receive CSV content from iOS app and store locally.
    """
    try:
        # Generate unique session ID using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{request.filename}"

        # Save CSV file
        csv_path = UPLOADS_DIR / session_id
        csv_path.write_text(request.csv_content)

        # Save metadata
        metadata = {
            "filename": request.filename,
            "uploaded_at": datetime.now().isoformat(),
            "size_bytes": len(request.csv_content)
        }
        meta_path = UPLOADS_DIR / f"{session_id}.meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

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
    List all uploaded sessions.
    """
    sessions = []

    for meta_file in sorted(UPLOADS_DIR.glob("*.meta.json"), reverse=True):
        try:
            metadata = json.loads(meta_file.read_text())
            session_id = meta_file.stem.replace(".meta", "")
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
    Get CSV content for a specific session.
    """
    csv_path = UPLOADS_DIR / session_id

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "csv_content": csv_path.read_text()}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """
    Delete a session and its metadata.
    """
    csv_path = UPLOADS_DIR / session_id
    meta_path = UPLOADS_DIR / f"{session_id}.meta.json"

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    csv_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    return {"success": True, "message": f"Deleted {session_id}"}

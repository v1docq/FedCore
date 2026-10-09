from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
EXPORT_DIR = APP_DIR / "exports"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="FedCore Artifact Storage Demo",
    description="Educational artifact storage service. Model conversion is unavailable; use the measured runner export route.",
    version="0.1.0",
)


@app.get("/")
def index():
    return HTMLResponse(
        """
        <h1>FedCore Artifact Storage Demo</h1>
        <p>Stores files only. Model conversion is unavailable.</p>
        <p>Available endpoints:</p>
        <ul>
            <li><code>GET /health</code></li>
            <li><code>GET /files</code></li>
            <li><code>POST /upload</code></li>
            <li><code>POST /export</code></li>
            <li><code>POST /analyze_model</code></li>
        </ul>
        """
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fedcore-model-exporter-demo",
        "uploads_dir": str(UPLOAD_DIR),
        "exports_dir": str(EXPORT_DIR),
    }


@app.get("/files")
def files():
    uploads = sorted(path.name for path in UPLOAD_DIR.glob("*") if path.is_file())
    exports = sorted(path.name for path in EXPORT_DIR.glob("*") if path.is_file())

    return {
        "uploads": uploads,
        "exports": exports,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).name != file.filename:
        raise HTTPException(status_code=400, detail="Use a simple filename without directory components")
    destination = UPLOAD_DIR / file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
    }


@app.post("/export")
def export_model(filename: str, target_format: str = "onnx"):
    return JSONResponse(status_code=501, content={"status": "unsupported",
                        "message": "This educational storage service does not convert models. Use FedCore.export or the measured experiment runner."})


@app.post("/analyze_model")
def analyze_model(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Use a simple filename without directory components")
    source = UPLOAD_DIR / filename

    if not source.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"File '{filename}' was not found in uploads directory.",
            },
        )

    return {
        "status": "file_metadata_only",
        "filename": filename,
        "size_bytes": source.stat().st_size,
        "suffix": source.suffix,
    }

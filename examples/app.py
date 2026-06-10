from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
EXPORT_DIR = APP_DIR / "exports"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="FedCore Model Exporter Demo",
    description="Minimal demo service for uploading and exporting model artifacts.",
    version="0.1.0",
)


@app.get("/")
def index():
    return HTMLResponse(
        """
        <h1>FedCore Model Exporter Demo</h1>
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
    source = UPLOAD_DIR / filename

    if not source.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"File '{filename}' was not found in uploads directory.",
            },
        )

    export_name = f"{source.stem}.{target_format.lower()}"
    destination = EXPORT_DIR / export_name

    # Demo behavior: copy artifact and change extension.
    # In production this block can call FedCore export pipeline.
    shutil.copyfile(source, destination)

    return {
        "status": "exported",
        "source": str(source),
        "target_format": target_format,
        "exported_file": str(destination),
        "size_bytes": destination.stat().st_size,
    }


@app.post("/analyze_model")
def analyze_model(filename: str):
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
        "status": "analyzed",
        "filename": filename,
        "size_bytes": source.stat().st_size,
        "suffix": source.suffix,
    }

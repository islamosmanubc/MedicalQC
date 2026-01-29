"""FastAPI service stub for on-prem inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from src.infer.infer_study import infer_study

app = FastAPI(title="MedicalQC Inference")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
def infer(study_path: str, model_path: str, device: str = "cpu") -> dict[str, Any]:
    try:
        return infer_study(Path(study_path), Path(model_path), device)
    except Exception as exc:  # pragma: no cover - service stub
        raise HTTPException(status_code=400, detail=str(exc)) from exc

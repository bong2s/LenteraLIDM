"""
Route: /health
Mengecek apakah API dan model ML berjalan normal.
Dipakai oleh web dev untuk memastikan server ML hidup sebelum request prediksi.
"""

from fastapi import APIRouter
from typing import Dict

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check() -> Dict:
    """
    Cek status API.

    Returns:
        {"status": "ok", "message": "ML API berjalan normal"}
    """
    return {
        "status": "ok",
        "message": "Handwriting Analysis ML API berjalan normal",
        "version": "1.0.0",
    }


@router.get("/", tags=["System"])
def root() -> Dict:
    """
    Root endpoint — redirect ke dokumentasi.
    Buka /docs untuk melihat Swagger UI interaktif.
    """
    return {
        "message": "Handwriting Analysis ML API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
    }
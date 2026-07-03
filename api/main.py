"""
=============================================================
FILE: api/main.py  (Dataset Real v3 RIASEC Direct)
=============================================================
Entry point FastAPI server analisis tulisan tangan.
Jalankan: uvicorn api.main:app --reload --port 8000

Endpoint:
  POST /predict       — prediksi dari gambar + nilai akademik
  GET  /predict/demo  — contoh respons tanpa upload
  GET  /health        — cek status server
  GET  /docs          — Swagger UI interaktif
=============================================================
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor
from api.routes import predict, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT / "models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("  Handwriting Analysis ML API — Dataset Real v3")
    logger.info("=" * 55)
    logger.info(f"Model dir: {MODEL_DIR}")

    predictor = Predictor()
    predictor.load_models(MODEL_DIR)

    if predictor.is_ready:
        logger.info("✅ Model ML berhasil dimuat!")
    else:
        logger.warning(
            "⚠️  Model belum ada. Jalankan training dulu:\n"
            "   python train_and_save.py\n"
            "   Endpoint /predict/demo tetap bisa dipakai."
        )

    app.state.predictor = predictor
    logger.info("API siap — buka http://localhost:8000/docs")
    logger.info("=" * 55)

    yield

    logger.info("Server ML API berhenti.")


app = FastAPI(
    title="Handwriting Analysis ML API",
    description="""
## API Analisis Tulisan Tangan → Karakter RIASEC + Rekomendasi Jurusan

Dataset real: 235 data tulisan (CSV) + 140 data akademik.

### Output:
1. **Karakter RIASEC** — tipe kepribadian Holland langsung dari tulisan tangan
2. **Kemampuan Akademik** — Rumpun Ilmu yang diprediksi dari nilai akademik
3. **Perbandingan** — apakah RIASEC sejalan atau berbeda dengan kemampuan akademik
4. **Rekomendasi Jurusan** — 3 jurusan (sejalan) atau 5 jurusan (berbeda)
    """,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Error internal server.", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
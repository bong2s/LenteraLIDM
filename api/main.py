"""
=============================================================
FILE: api/main.py  (Dataset Real v2)
=============================================================
Entry point FastAPI server analisis tulisan tangan.
Jalankan: uvicorn api.main:app --reload --port 8000

Endpoint:
  POST /predict       — prediksi dari gambar + data siswa
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
    logger.info("  Handwriting Analysis ML API — Dataset Real v2")
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
## API Analisis Tulisan Tangan → Karakter Siswa + Rekomendasi Jurusan

Dataset real: 221 gambar Big Five + 140 data akademik + 3600 data kecerdasan Gardner.

### Output:
1. **Profil Big Five** — kepribadian OCEAN dari tulisan tangan
2. **Tipe RIASEC** — karakter karier Holland (6 tipe)
3. **Rumpun Ilmu** — kluster akademik yang paling cocok
4. **TOP-3 Program Studi** — rekomendasi jurusan + alasan + estimasi IPK
    """,
    version="2.0.0",
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
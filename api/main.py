"""
=============================================================
FILE: api/main.py
=============================================================
TUJUAN:
  Entry point FastAPI server.
  Jalankan dengan: uvicorn api.main:app --reload --port 8000

FITUR:
  - Auto-load model saat server start
  - Swagger UI otomatis di /docs
  - CORS aktif (web Laravel bisa akses)
  - Logging request
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

# Tambah root ke Python path agar import modul src/* berjalan
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predictor import HandwritingPredictor
from api.routes import predict, health

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Konfigurasi path model
# ------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT / "models"))


# ------------------------------------------------------------------
# Lifecycle: startup & shutdown
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Dijalankan saat server START dan STOP.
    - START: muat model ML ke memory
    - STOP: cleanup (opsional)
    """
    # --- STARTUP ---
    logger.info("=" * 50)
    logger.info("Memulai Handwriting Analysis ML API...")
    logger.info(f"Mencari model di: {MODEL_DIR}")

    predictor = HandwritingPredictor()
    loaded = predictor.load_models(MODEL_DIR)

    if loaded:
        logger.info("✅ Model ML berhasil dimuat!")
    else:
        logger.warning(
            "⚠️  Model belum tersedia. Jalankan training dulu:\n"
            "   cd ml-handwriting && python train_and_save.py\n"
            "   Endpoint /predict/demo tetap tersedia untuk testing."
        )

    # Simpan predictor di app state agar bisa diakses dari routes
    app.state.predictor = predictor

    logger.info("API siap menerima request.")
    logger.info("Buka http://localhost:8000/docs untuk dokumentasi.")
    logger.info("=" * 50)

    yield  # Server berjalan di sini

    # --- SHUTDOWN ---
    logger.info("Server ML API berhenti.")


# ------------------------------------------------------------------
# Inisialisasi FastAPI App
# ------------------------------------------------------------------
app = FastAPI(
    title="Handwriting Analysis ML API",
    description="""
## API Analisis Tulisan Tangan → Karakter Siswa + Rekomendasi Jurusan

API ini menerima **gambar tulisan tangan** siswa dan menganalisisnya menggunakan
Machine Learning untuk menghasilkan:

1. **Profil Karakter** — Tipe kepribadian RIASEC siswa (Realistic, Investigative, Artistic,
   Social, Enterprising, Conventional)
2. **Top-3 Rekomendasi Jurusan** — Jurusan kuliah yang paling cocok beserta alasannya
3. **Analisis Tulisan** — Fitur-fitur yang diekstrak dari gambar

### Cara Integrasi ke Laravel:
```php
$response = Http::attach('file', file_get_contents($imagePath), 'tulisan.png')
    ->post('http://localhost:8000/predict', [
        'akademik' => json_encode(['matematika' => 88, 'ipa' => 85]),
        'talent'   => json_encode(['logika' => 8, 'teknologi' => 9]),
    ]);

$hasil = $response->json();
// $hasil['karakter']['nama']          → "Analitis & Ilmiah"
// $hasil['rekomendasi_jurusan'][0]    → {"rank":1, "jurusan":"Informatika", ...}
```

### Endpoint Utama:
- **POST /predict** — prediksi dari gambar + data siswa
- **GET /predict/demo** — contoh respons tanpa upload (untuk testing)
- **GET /health** — cek status server
- **GET /docs** — dokumentasi interaktif (Swagger UI)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------------
# CORS Middleware
# Izinkan request dari domain manapun (ubah origins di production)
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Ganti dengan domain Laravel kamu di production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Daftarkan Routes
# ------------------------------------------------------------------
app.include_router(health.router)
app.include_router(predict.router)

# ------------------------------------------------------------------
# Error handler global
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Terjadi error internal server.", "detail": str(exc)},
    )


# ------------------------------------------------------------------
# Jalankan langsung (opsional, biasanya pakai uvicorn)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
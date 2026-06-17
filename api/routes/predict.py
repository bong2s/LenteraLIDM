"""
=============================================================
ROUTE: /predict
=============================================================
TUJUAN:
  Endpoint utama yang diterima dari website Laravel.
  Menerima:
    - File gambar tulisan tangan (multipart/form-data)
    - Data akademik (JSON, opsional)
    - Data bakat (JSON, opsional)

  Mengembalikan:
    - Tipe karakter RIASEC
    - Deskripsi kepribadian siswa
    - Top-3 rekomendasi jurusan + alasan

CARA KIRIM DARI LARAVEL / JavaScript:
  const formData = new FormData();
  formData.append("file", fileInput);
  formData.append("akademik", JSON.stringify({matematika: 88, ipa: 85}));
  formData.append("talent", JSON.stringify({logika: 8, teknologi: 9}));

  const res = await fetch("/predict", { method: "POST", body: formData });
  const result = await res.json();
=============================================================
"""

import json
import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional

from api.schemas import PredictionResponse, AkademikInput, TalentInput, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediksi"],
    summary="Analisis Tulisan Tangan → Karakter + Rekomendasi Jurusan",
    responses={
        200: {"description": "Berhasil", "model": PredictionResponse},
        400: {"description": "Input tidak valid", "model": ErrorResponse},
        500: {"description": "Error server", "model": ErrorResponse},
    },
)
async def predict_handwriting(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Gambar tulisan tangan siswa (.png atau .jpg, maks 10MB)"
    ),
    akademik: Optional[str] = Form(
        None,
        description='JSON nilai akademik, contoh: {"matematika": 88, "ipa": 85}'
    ),
    talent: Optional[str] = Form(
        None,
        description='JSON skor bakat, contoh: {"logika": 8, "teknologi": 9}'
    ),
):
    """
    ## Analisis Tulisan Tangan Siswa

    Upload gambar tulisan tangan untuk mendapatkan:
    1. **Profil karakter** siswa (tipe RIASEC + deskripsi)
    2. **Skor RIASEC** (persentase tiap tipe kepribadian)
    3. **Top-3 rekomendasi jurusan** + alasan personal
    4. **Fitur tulisan** yang diekstrak dari gambar

    ### Input:
    - **file**: gambar tulisan tangan (.png / .jpg)
    - **akademik** *(opsional)*: nilai rapor sebagai JSON
    - **talent** *(opsional)*: skor bakat sebagai JSON

    ### Contoh penggunaan (JavaScript/fetch):
    ```javascript
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("akademik", JSON.stringify({ matematika: 88, ipa: 85 }));
    formData.append("talent", JSON.stringify({ logika: 8, teknologi: 9 }));

    const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
    });
    const hasil = await response.json();
    console.log(hasil.karakter.nama);            // "Analitis & Ilmiah"
    console.log(hasil.rekomendasi_jurusan[0]);   // { rank: 1, jurusan: "Informatika", ... }
    ```

    ### Contoh penggunaan (PHP / Laravel):
    ```php
    $response = Http::attach('file', file_get_contents($path), 'tulisan.png')
        ->post('http://localhost:8000/predict', [
            'akademik' => json_encode(['matematika' => 88, 'ipa' => 85]),
            'talent'   => json_encode(['logika' => 8, 'teknologi' => 9]),
        ]);
    $hasil = $response->json();
    ```
    """

    # ------------------------------------------------------------------
    # Validasi file
    # ------------------------------------------------------------------
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Format file tidak didukung: {file.content_type}. Gunakan PNG atau JPG."
        )

    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Ukuran file terlalu besar: {len(image_bytes)//1024}KB. Maks 10MB."
        )

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="File gambar terlalu kecil atau kosong.")

    # ------------------------------------------------------------------
    # Parse JSON opsional
    # ------------------------------------------------------------------
    akademik_dict = None
    talent_dict = None

    if akademik:
        try:
            raw = json.loads(akademik)
            akademik_dict = AkademikInput(**raw).to_dict()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON 'akademik' tidak valid.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data akademik tidak valid: {e}")

    if talent:
        try:
            raw = json.loads(talent)
            talent_dict = TalentInput(**raw).to_dict()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON 'talent' tidak valid.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data talent tidak valid: {e}")

    # ------------------------------------------------------------------
    # Jalankan prediksi
    # ------------------------------------------------------------------
    predictor = request.app.state.predictor

    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model ML belum siap. Jalankan training terlebih dahulu: python train_and_save.py"
        )

    try:
        result = predictor.predict(
            image_bytes=image_bytes,
            akademik=akademik_dict,
            talent=talent_dict,
        )
        logger.info(
            f"Prediksi berhasil: RIASEC={result['karakter']['tipe']}, "
            f"Jurusan={result['rekomendasi_jurusan'][0]['jurusan']}"
        )
        return result

    except Exception as e:
        logger.exception(f"Error saat prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")


@router.get(
    "/predict/demo",
    tags=["Prediksi"],
    summary="Demo prediksi tanpa upload gambar (untuk testing)",
)
async def predict_demo(request: Request):
    """
    Demo endpoint yang mengembalikan contoh respons tanpa membutuhkan gambar.
    Berguna untuk web dev testing integrasi sebelum model siap.
    """
    return {
        "status": "demo",
        "pesan": "Ini adalah contoh respons. Gunakan POST /predict untuk prediksi nyata.",
        "karakter": {
            "tipe": "Investigative",
            "nama": "Analitis & Ilmiah",
            "deskripsi": "Kamu adalah pemikir mendalam yang suka menganalisis, meneliti, dan memecahkan masalah kompleks.",
            "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis & sistematis", "Rasa ingin tahu tinggi"],
            "warna": "#2980B9",
        },
        "riasec_skor": {
            "Investigative": 0.45,
            "Conventional": 0.22,
            "Realistic": 0.13,
            "Artistic": 0.09,
            "Social": 0.07,
            "Enterprising": 0.04,
        },
        "rekomendasi_jurusan": [
            {"rank": 1, "jurusan": "Informatika", "match_score": 87.3, "alasan": "Kemampuan analitismu sangat cocok untuk pemrograman dan AI"},
            {"rank": 2, "jurusan": "Statistik", "match_score": 72.1, "alasan": "Kamu suka pola dan data – statistik adalah duniamu"},
            {"rank": 3, "jurusan": "Sistem Informasi", "match_score": 65.8, "alasan": "Kamu menyukai sistem yang teratur – SI adalah pilihan tepat"},
        ],
        "fitur_tulisan": {
            "letter_size_score": 4.2,
            "slant_angle": 5.1,
            "pressure_score": 6.3,
            "spacing_score": 4.8,
            "readability_score": 7.2,
            "neatness_score": 8.1,
            "connectivity_score": 3.4,
            "ornament_score": 1.2,
            "line_straightness": 8.5,
            "density_score": 5.0,
        },
        "feature_importance": [
            {"fitur": "conventional", "importance": 0.1832},
            {"fitur": "investigative", "importance": 0.1654},
            {"fitur": "neatness_score", "importance": 0.1102},
            {"fitur": "matematika", "importance": 0.0943},
            {"fitur": "logika_t", "importance": 0.0871},
        ],
    }
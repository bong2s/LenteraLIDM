"""
=============================================================
ROUTE: /predict  (Dataset Real v2)
=============================================================
Endpoint utama yang diterima dari website Laravel.

Menerima:
  - file     : gambar tulisan tangan (multipart/form-data)
  - akademik : JSON nilai 14 mata pelajaran (opsional)
  - talent   : JSON skor 8 kecerdasan Gardner (opsional)

Mengembalikan:
  - profil_karakter      : Big Five + RIASEC
  - analisis_akademik    : Rumpun Ilmu + nilai
  - rekomendasi_jurusan  : TOP-3 Program Studi
  - fitur_tulisan        : 10 fitur numerik tulisan
  - kelengkapan_data     : status field input

CONTOH LARAVEL:
  $response = Http::withOptions(['proxy' => ''])
      ->attach('file', file_get_contents($path), 'tulisan.jpg')
      ->post('http://localhost:8000/predict', [
          'akademik' => json_encode([
              'mat_s4' => 90, 'fis_s4' => 85, 'info_s4' => 95,
              'mat_s5' => 88, 'info_s5' => 92,
          ]),
          'talent' => json_encode([
              'logika_mat' => 16, 'spasial' => 14,
          ]),
      ]);
=============================================================
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    PredictResponse, AkademikInput, TalentInput,
    build_predict_response,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediksi"],
    summary="Analisis tulisan tangan → Profil karakter + Rekomendasi jurusan",
)
async def predict_handwriting(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Gambar tulisan tangan siswa (.jpg / .png, maks 10MB)",
    ),
    akademik: Optional[str] = Form(
        None,
        description=(
            "JSON nilai 14 mata pelajaran (opsional). "
            'Contoh: {"mat_s4":90,"fis_s4":85,"info_s4":95,'
            '"mat_s5":88,"fis_s5":82,"info_s5":92}'
        ),
    ),
    talent: Optional[str] = Form(
        None,
        description=(
            "JSON skor kecerdasan Gardner (opsional). "
            'Contoh: {"logika_mat":16,"spasial":14,"interpersonal":12}'
        ),
    ),
):
    """
    ## Analisis Tulisan Tangan Siswa

    Upload gambar tulisan tangan untuk mendapatkan:

    1. **Profil karakter RIASEC** — tipe kepribadian + deskripsi + kekuatan
    2. **Profil Big Five** — 5 dimensi kepribadian dari analisis tulisan
    3. **Analisis akademik** — Rumpun Ilmu yang cocok + nilai tertinggi
    4. **Top-3 rekomendasi Program Studi** — + alasan + estimasi IPK
    5. **10 fitur tulisan tangan** — hasil ekstraksi OpenCV

    ---

    ### Keys untuk field `akademik` (JSON):
    | Key | Keterangan |
    |-----|-----------|
    | mat_s4 / mat_s5 | Matematika Semester 4 / 5 |
    | fis_s4 / fis_s5 | Fisika Semester 4 / 5 |
    | kim_s4 / kim_s5 | Kimia Semester 4 / 5 |
    | bio_s4 / bio_s5 | Biologi Semester 4 / 5 |
    | bind_s4 / bind_s5 | Bahasa Indonesia Semester 4 / 5 |
    | bing_s4 / bing_s5 | Bahasa Inggris Semester 4 / 5 |
    | info_s4 / info_s5 | Informatika Semester 4 / 5 |

    ### Keys untuk field `talent` (JSON):
    | Key | Keterangan |
    |-----|-----------|
    | linguistik | Kecerdasan bahasa |
    | musikal | Kecerdasan musik |
    | kinestetik | Kecerdasan fisik/gerak |
    | logika_mat | Kecerdasan logika-matematika |
    | spasial | Kecerdasan ruang/visual |
    | interpersonal | Kecerdasan sosial |
    | intrapersonal | Kecerdasan refleksi diri |
    | naturalis | Kecerdasan alam |
    """

    # --- Validasi file ---
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Format file tidak didukung: {file.content_type}. Gunakan JPG atau PNG.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Ukuran file terlalu besar ({len(image_bytes)//1024}KB). Maks 10MB.",
        )

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="File gambar terlalu kecil atau kosong.")

    # --- Parse JSON opsional ---
    akademik_dict = None
    talent_dict   = None

    if akademik:
        try:
            akademik_dict = AkademikInput(**json.loads(akademik)).to_dict()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON 'akademik' tidak valid.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data akademik tidak valid: {e}")

    if talent:
        try:
            talent_dict = TalentInput(**json.loads(talent)).to_dict()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON 'talent' tidak valid.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data talent tidak valid: {e}")

    # --- Prediksi ---
    predictor = request.app.state.predictor

    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model ML belum siap. Jalankan dulu: "
                "cd ml-handwriting && python train_and_save.py"
            ),
        )

    try:
        raw = predictor.predict(
            image_bytes=image_bytes,
            academic_scores=akademik_dict,
            talent_scores=talent_dict,
        )
        response = build_predict_response(raw)
        logger.info(
            f"Prediksi OK — "
            f"BigFive={response.big_five.dominant}  "
            f"RIASEC={response.profil_karakter.dominant}  "
            f"Rumpun={response.analisis_akademik.rumpun_ilmu}  "
            f"Top1={response.rekomendasi_jurusan[0].program_studi if response.rekomendasi_jurusan else '-'}"
        )
        return response

    except Exception as e:
        logger.exception(f"Error prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")


@router.get(
    "/predict/demo",
    tags=["Prediksi"],
    summary="Contoh respons tanpa upload (untuk testing Laravel)",
)
async def predict_demo():
    """
    Kembalikan contoh respons lengkap tanpa upload gambar.
    Berguna untuk web dev testing integrasi sebelum model siap.
    """
    return {
        "status": "demo — gunakan POST /predict untuk prediksi nyata",
        "profil_karakter": {
            "dominant": "Investigative",
            "karakter": "Analitis & Ilmiah",
            "deskripsi": "Kamu pemikir mendalam yang suka menganalisis...",
            "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis", "Rasa ingin tahu tinggi"],
            "warna": "#2980B9",
            "skor": {
                "Investigative": 38.5, "Conventional": 20.1, "Realistic": 15.3,
                "Artistic": 12.0, "Social": 9.1, "Enterprising": 5.0,
            },
        },
        "big_five": {
            "dominant": "Conscientiousness",
            "skor": {
                "Openness": 5.1, "Conscientiousness": 7.8, "Extraversion": 4.2,
                "Agreeableness": 5.5, "Neuroticism": 3.9,
            },
            "probabilitas": {
                "Conscientiousness": 45.2, "Openness": 22.1, "Agreeableness": 18.0,
                "Neuroticism": 8.7, "Extraversion": 6.0,
            },
        },
        "analisis_akademik": {
            "rumpun_ilmu": "STEM",
            "rumpun_probabilitas": {
                "STEM": 62.5, "Bisnis Manajemen": 18.0, "Sosial Humaniora": 10.5,
                "Pendidikan": 6.0, "Seni Kreatif": 3.0,
            },
            "nilai_rata_rata": 88.5,
            "mata_pelajaran_kuat": [
                {"mata_pelajaran": "Informatika Smt 5", "nilai": 95.0},
                {"mata_pelajaran": "Matematika Smt 5", "nilai": 92.0},
                {"mata_pelajaran": "Matematika Smt 4", "nilai": 90.0},
            ],
        },
        "rekomendasi_jurusan": [
            {
                "program_studi": "S1 Teknik Informatika",
                "rumpun_ilmu": "STEM",
                "alasan": "Profil Analitis & Ilmiah sangat sesuai bidang STEM...",
                "skor_kesesuaian": 4.6,
                "prediksi_ipk": "3.3 – 3.7",
            },
            {
                "program_studi": "S1 Statistika",
                "rumpun_ilmu": "STEM",
                "alasan": "Kecerdasan logika-matematika tinggi cocok untuk Statistika.",
                "skor_kesesuaian": 4.1,
                "prediksi_ipk": "3.1 – 3.5",
            },
            {
                "program_studi": "S1 Sistem Informasi",
                "rumpun_ilmu": "STEM",
                "alasan": "Kombinasi teknis dan analitis ideal untuk Sistem Informasi.",
                "skor_kesesuaian": 3.9,
                "prediksi_ipk": "3.0 – 3.4",
            },
        ],
        "fitur_tulisan": {
            "letter_size": 4.2, "slant": 5.1, "pressure": 6.3,
            "spacing": 4.8, "readability": 7.2, "neatness": 8.1,
            "connectivity": 3.4, "ornament": 1.2, "baseline": 8.5, "density": 5.0,
        },
        "kelengkapan_data": {
            "akademik": "demo",
            "talent": "demo",
        },
    }

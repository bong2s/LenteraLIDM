"""
=============================================================
ROUTE: /predict  (Dataset Real v3.1 — RIASEC + Minat)
=============================================================
Endpoint utama yang diterima dari website Laravel.

Menerima (multipart/form-data):
  - file     : gambar tulisan tangan (jpg/png, maks 10MB)
  - akademik : JSON nilai 14 mata pelajaran (opsional)
  - minat    : JSON jawaban 24 soal RIASEC atau 6 skor (opsional)

CONTOH LARAVEL — kirim semua:
  $response = Http::attach('file', file_get_contents($path), 'tulisan.jpg')
      ->post('http://localhost:8000/predict', [
          'akademik' => json_encode([
              'mat_s4' => 90, 'fis_s4' => 85, 'info_s4' => 95,
              'mat_s5' => 88, 'info_s5' => 92,
          ]),
          'minat' => json_encode([
              'q_R1'=>2,'q_R2'=>3,'q_R3'=>2,'q_R4'=>1,
              'q_I1'=>4,'q_I2'=>3,'q_I3'=>4,'q_I4'=>3,
              'q_A1'=>1,'q_A2'=>2,'q_A3'=>1,'q_A4'=>0,
              'q_S1'=>2,'q_S2'=>1,'q_S3'=>2,'q_S4'=>1,
              'q_E1'=>3,'q_E2'=>2,'q_E3'=>3,'q_E4'=>2,
              'q_C1'=>4,'q_C2'=>3,'q_C3'=>4,'q_C4'=>3,
          ]),
      ]);

ATAU — kirim skor yang sudah dijumlah dari web (lebih ringkas):
  'minat' => json_encode([
      'score_R'=>8,'score_I'=>14,'score_A'=>4,
      'score_S'=>6,'score_E'=>10,'score_C'=>14,
  ])
=============================================================
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    PredictResponse, AkademikInput,
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
    summary="Analisis tulisan tangan → Karakter RIASEC + Minat + Rekomendasi jurusan",
)
async def predict_handwriting(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Gambar tulisan tangan (.jpg / .png, maks 10MB)",
    ),
    akademik: Optional[str] = Form(
        None,
        description=(
            "JSON nilai 14 mata pelajaran (opsional). "
            'Contoh: {"mat_s4":90,"fis_s4":85,"info_s4":95,"mat_s5":88,"info_s5":92}'
        ),
    ),
    minat: Optional[str] = Form(
        None,
        description=(
            "JSON jawaban kuesioner RIASEC (opsional). "
            "Format A — 24 soal: {'q_R1':3,'q_R2':2,...,'q_C4':3} | "
            "Format B — 6 skor: {'score_R':11,'score_I':14,...,'score_C':9}"
        ),
    ),
):
    """
    ## Analisis Tulisan Tangan Siswa (v3.1)

    Upload gambar tulisan tangan untuk mendapatkan:

    1. **Karakter RIASEC dari tulisan tangan** — tipe Holland langsung dari analisis grafologi
    2. **Karakter RIASEC dari kuesioner minat** — jika `minat` dikirim
    3. **Perbandingan tulisan vs minat** — sejalan atau berbeda? (jika `minat` dikirim)
    4. **Analisis akademik** — Rumpun Ilmu + mata pelajaran terkuat
    5. **Perbandingan tulisan vs akademik** — sejalan atau berbeda?
    6. **Rekomendasi jurusan** — 3 jurusan (semua konsisten) atau 5 jurusan (ada perbedaan)
    7. **10 fitur tulisan tangan** — hasil ekstraksi OpenCV

    ---

    ### Field `minat` — dua format yang diterima:

    **Format A** (24 jawaban mentah, tiap soal skala 0–4):
    ```json
    {"q_R1":2,"q_R2":3,"q_R3":2,"q_R4":1,
     "q_I1":4,"q_I2":3,"q_I3":4,"q_I4":3, ...}
    ```

    **Format B** (6 skor sudah dijumlah dari web, max 16 per tipe):
    ```json
    {"score_R":8,"score_I":14,"score_A":4,"score_S":6,"score_E":10,"score_C":14}
    ```
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

    # --- Parse akademik JSON (opsional) ---
    akademik_dict = None
    if akademik:
        try:
            akademik_dict = AkademikInput(**json.loads(akademik)).to_dict()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON 'akademik' tidak valid.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data akademik tidak valid: {e}")

    # --- Parse minat JSON (opsional) ---
    minat_dict = None
    if minat:
        try:
            minat_dict = MinatInput(**json.loads(minat)).to_dict()
            if not minat_dict:
                raise ValueError("Semua field minat kosong.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON 'minat' tidak valid.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data minat tidak valid: {e}")

    # --- Prediksi ---
    predictor = request.app.state.predictor

    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model ML belum siap. Jalankan dulu: python train_and_save.py",
        )

    try:
        raw = predictor.predict(
            image_bytes=image_bytes,
            academic_scores=akademik_dict,
            minat_scores=minat_dict,
        )
        response = build_predict_response(raw)

        minat_info = ""
        if response.riasec_minat:
            minat_info = f"  MinatRIASEC={response.riasec_minat.dominant}"
            if response.perbandingan_minat:
                minat_info += f"({response.perbandingan_minat.status})"

        logger.info(
            f"Prediksi OK — "
            f"RIASEC={response.riasec_karakter.dominant}  "
            f"Rumpun={response.analisis_akademik.rumpun_ilmu}  "
            f"Akademik={response.perbandingan_akademik.status}"
            f"{minat_info}  "
            f"Top1={response.rekomendasi_jurusan[0].program_studi if response.rekomendasi_jurusan else '-'}  "
            f"N={len(response.rekomendasi_jurusan)}"
        )
        return response

    except Exception as e:
        logger.exception(f"Error prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")


@router.get(
    "/predict/demo",
    tags=["Prediksi"],
    summary="Contoh respons lengkap tanpa upload (untuk testing Laravel)",
)
async def predict_demo():
    """
    Kembalikan contoh respons lengkap v3.1 (dengan minat) tanpa upload gambar.
    """
    return {
        "status": "demo — gunakan POST /predict untuk prediksi nyata",
        "riasec_karakter": {
            "dominant": "Investigative", "karakter": "Analitis & Ilmiah",
            "deskripsi": "Kamu pemikir mendalam yang suka menganalisis...",
            "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis & sistematis", "Rasa ingin tahu tinggi"],
            "warna": "#2980B9",
            "skor": {"Investigative": 38.5, "Conventional": 20.1, "Realistic": 15.3,
                     "Artistic": 12.0, "Social": 9.1, "Enterprising": 5.0},
        },
        "riasec_minat": {
            "dominant": "Investigative", "karakter": "Analitis & Ilmiah",
            "deskripsi": "Kamu pemikir mendalam yang suka menganalisis...",
            "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis & sistematis", "Rasa ingin tahu tinggi"],
            "warna": "#2980B9",
            "skor_raw":    {"Investigative": 14.0, "Conventional": 12.0, "Realistic": 8.0,
                            "Artistic": 4.0, "Social": 6.0, "Enterprising": 5.0},
            "skor_persen": {"Investigative": 28.6, "Conventional": 24.5, "Realistic": 16.3,
                            "Artistic": 8.2, "Social": 12.2, "Enterprising": 10.2},
        },
        "analisis_akademik": {
            "rumpun_ilmu": "STEM",
            "rumpun_probabilitas": {"STEM": 62.5, "Bisnis Manajemen": 18.0,
                                    "Sosial Humaniora": 10.5, "Pendidikan": 6.0, "Seni Kreatif": 3.0},
            "nilai_rata_rata": 88.5,
            "mata_pelajaran_kuat": [
                {"mata_pelajaran": "Informatika Smt 5", "nilai": 95.0},
                {"mata_pelajaran": "Matematika Smt 5",  "nilai": 92.0},
                {"mata_pelajaran": "Matematika Smt 4",  "nilai": 90.0},
            ],
        },
        "perbandingan_akademik": {
            "status": "SEJALAN",
            "penjelasan": "Karakter Analitis & Ilmiah sejalan dengan kecenderungan akademik di bidang STEM.",
        },
        "perbandingan_minat": {
            "status": "SEJALAN",
            "penjelasan": "Tulisan tangan dan kuesioner minat sama-sama menunjukkan karakter Investigative.",
        },
        "rekomendasi_jurusan": [
            {"program_studi": "S1 Teknik Informatika", "rumpun_ilmu": "STEM",
             "alasan": "Profil Analitis & Ilmiah sejalan dengan STEM...",
             "skor_kesesuaian": 4.6, "prediksi_ipk": "3.3 – 3.7"},
            {"program_studi": "S1 Statistika", "rumpun_ilmu": "STEM",
             "alasan": "Kemampuan logis dan analitis kuat cocok untuk Statistika.",
             "skor_kesesuaian": 4.1, "prediksi_ipk": "3.1 – 3.5"},
            {"program_studi": "S1 Sistem Informasi", "rumpun_ilmu": "STEM",
             "alasan": "Kombinasi teknis dan analitis ideal untuk Sistem Informasi.",
             "skor_kesesuaian": 3.9, "prediksi_ipk": "3.0 – 3.4"},
        ],
        "fitur_tulisan": {
            "letter_size": 4.2, "slant": 5.1, "pressure": 6.3,
            "spacing": 4.8, "readability": 7.2, "neatness": 8.1,
            "connectivity": 3.4, "ornament": 1.2, "baseline": 8.5, "density": 5.0,
        },
        "kelengkapan_data": {"akademik": "demo", "minat": "demo"},
    }
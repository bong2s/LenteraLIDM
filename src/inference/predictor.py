"""
=============================================================
MODUL: predictor.py
=============================================================
TUJUAN:
  Kelas utama yang dipanggil oleh FastAPI untuk prediksi lengkap.

ALUR:
  1. Terima gambar + data siswa (nilai boleh tidak lengkap)
  2. Ekstrak 10 fitur gambar via OpenCV
  3. Petakan fitur gambar ke nama kolom Excel (agar cocok dgn model)
  4. Hitung kecenderungan RIASEC dari tulisan (grafologi)
  5. Gabungkan dengan fitur akademik & bakat
  6. Prediksi RIASEC + rekomendasikan Top-3 jurusan
  7. Kembalikan hasil terstruktur

KENAPA ADA 2 SET NAMA FITUR GAMBAR?
  Saat training, model melihat DUA sumber fitur gambar:
    a. Kolom dari Dataset_Tulisan.xlsx : letter_size, slant, pressure, ...
    b. Output image_processor.py       : letter_size_score, slant_angle, ...
  Saat prediksi, kita harus menyediakan KEDUANYA agar model tidak
  mendapat nilai 0 (yang menyebabkan semua siswa hasilnya sama).
=============================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.image_processor import HandwritingFeatureExtractor
from src.models.riasec_classifier import RIASECClassifier
from src.models.major_recommender import MajorRecommender
from src.preprocessing.data_loader import RIASEC_DESCRIPTIONS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Nilai default untuk fitur akademik dan bakat yang tidak diisi
# ------------------------------------------------------------------

DEFAULT_AKADEMIK: Dict[str, float] = {
    "agama": 75, "pancasila": 75, "bahasa_indonesia": 75,
    "matematika": 75, "ipa": 75, "ips": 75, "bahasa_inggris": 75,
    "pjok": 75, "informatika": 75, "seni_budaya": 75,
    "logika": 75, "kreativitas": 75, "komunikasi": 75,
    "kepemimpinan": 75, "problem_solving": 75, "teamwork": 75,
    "literasi": 75, "numerasi": 75,
}

DEFAULT_TALENT: Dict[str, float] = {
    "komunikasi_t": 5, "kepemimpinan_t": 5, "kreativitas_t": 5,
    "logika_t": 5, "teknologi": 5, "riset": 5, "seni": 5,
    "olahraga": 5, "organisasi": 5, "kewirausahaan": 5,
    "kerja_tim": 5, "problem_solving_t": 5,
}

# Pemetaan nama fitur image_processor → nama kolom Excel di Dataset_Tulisan.xlsx
# Model dilatih dengan KEDUA nama ini, jadi keduanya harus ada saat prediksi
IMG_TO_EXCEL_MAP = {
    "letter_size_score": "letter_size",
    "slant_angle":       "slant",
    "pressure_score":    "pressure",
    "spacing_score":     "spacing",
    "readability_score": "readability",
    "neatness_score":    "neatness",
    "ornament_score":    "ornament",
    "connectivity_score":"connectivity",
    "line_straightness": "baseline",
}


class HandwritingPredictor:
    """
    Kelas prediksi end-to-end untuk tulisan tangan siswa.

    Catatan penting:
      - Model dilatih dengan 55 fitur dari 3 sumber berbeda.
      - Fitur gambar hadir dalam DUA nama berbeda (Excel + image_processor).
      - Kecenderungan RIASEC (realistic, investigative, dll.) dihitung
        otomatis dari fitur gambar menggunakan prinsip grafologi.
    """

    def __init__(self):
        self.image_extractor = HandwritingFeatureExtractor()
        self.riasec_clf  = RIASECClassifier()
        self.major_rec   = MajorRecommender()
        self.feature_meta: Dict = {}
        self._models_loaded = False

    # ------------------------------------------------------------------
    # MUAT MODEL
    # ------------------------------------------------------------------
    def load_models(self, model_dir: str = "models") -> bool:
        try:
            self.riasec_clf.load(os.path.join(model_dir, "riasec_model.pkl"))
            self.major_rec.load(os.path.join(model_dir,  "major_model.pkl"))
            with open(os.path.join(model_dir, "feature_meta.json")) as f:
                self.feature_meta = json.load(f)
            self._models_loaded = True
            logger.info("Semua model berhasil dimuat.")
            return True
        except Exception as e:
            logger.error(f"Gagal memuat model: {e}")
            self._models_loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._models_loaded

    # ------------------------------------------------------------------
    # PREDIKSI UTAMA
    # ------------------------------------------------------------------
    def predict(
                self,
        image_bytes: bytes,
        akademik: Optional[Dict[str, float]] = None,
        talent:   Optional[Dict[str, float]] = None,
    ) -> Dict:
        if not self._models_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load_models() terlebih dahulu.")

        # --- Step 1: Ekstrak 10 fitur dari gambar ---
        logger.info("Mengekstrak fitur tulisan tangan...")
        img_features = self.image_extractor.extract_from_bytes(image_bytes)

        # --- Step 2: Petakan ke nama kolom Excel agar cocok dengan model ---
        # Model dilatih dengan kedua nama (Excel + image_processor), jadi kita isi keduanya
        excel_features: Dict[str, float] = {}
        for img_key, excel_key in IMG_TO_EXCEL_MAP.items():
            excel_features[excel_key] = img_features.get(img_key, 5.0)

        # --- Step 3: Hitung kecenderungan RIASEC dari tulisan (grafologi) ---
        riasec_tendency = self._compute_riasec_tendency(img_features)

        # --- Step 4: Isi akademik & bakat (default jika tidak diisi) ---
        akademik_provided = set(akademik.keys()) if akademik else set()
        talent_provided   = set(talent.keys())   if talent   else set()

        akademik_final = dict(DEFAULT_AKADEMIK)
        if akademik:
            akademik_final.update(akademik)

        talent_input = talent or {}
        rename_map = {
            "komunikasi": "komunikasi_t", "kepemimpinan": "kepemimpinan_t",
            "kreativitas": "kreativitas_t", "logika": "logika_t",
            "problem_solving": "problem_solving_t",
        }
        talent_renamed = {rename_map.get(k, k): v for k, v in talent_input.items()}
        talent_final = dict(DEFAULT_TALENT)
        talent_final.update(talent_renamed)

        # --- Step 5: Bangun feature dict lengkap (55 fitur seperti saat training) ---
        feature_dict: Dict[str, float] = {}
        feature_dict.update(akademik_final)          # 18 fitur akademik
        feature_dict.update(talent_final)            # 12 fitur bakat
        feature_dict.update(excel_features)          # 9 fitur gambar (nama Excel)
        feature_dict.update(riasec_tendency)         # 6 skor RIASEC dari grafologi
        feature_dict.update(img_features)            # 10 fitur gambar (nama image_processor)
        # Total = 18 + 12 + 9 + 6 + 10 = 55 fitur ✓

        X = pd.DataFrame([feature_dict])

        # --- Step 6: Prediksi RIASEC ---
        riasec_type  = self.riasec_clf.predict(X)
        riasec_proba = self.riasec_clf.predict_proba(X)

        # --- Step 7: Rekomendasikan Top-3 jurusan ---
        top3_jurusan = self.major_rec.recommend_top3(
            X,
            riasec_type=riasec_type,
            riasec_proba=riasec_proba,
            akademik_scores=akademik_final,
        )

        # --- Step 8: Susun kelengkapan data ---
        kelengkapan = self._buat_info_kelengkapan(akademik_provided, talent_provided)

        # --- Step 9: Susun hasil ---
        riasec_info = RIASEC_DESCRIPTIONS.get(riasec_type, {})
        result = {
            "karakter": {
                "tipe":      riasec_type,
                "nama":      riasec_info.get("karakter", riasec_type),
                "deskripsi": riasec_info.get("deskripsi", ""),
                "kekuatan":  riasec_info.get("kekuatan", []),
                "warna":     riasec_info.get("warna", "#666666"),
            },
            "riasec_skor":         riasec_proba,
            "rekomendasi_jurusan": top3_jurusan,
            "fitur_tulisan":       img_features,
            "fitur_grafologi":     riasec_tendency,
            "feature_importance":  self.riasec_clf.get_feature_importance(top_n=5),
            "kelengkapan_data":    kelengkapan,
        }

        logger.info(
            f"Prediksi → RIASEC={riasec_type} | "
            f"Jurusan={[j['jurusan'] for j in top3_jurusan]} | "
            f"Grafologi={riasec_tendency}"
        )
        return result

    # ------------------------------------------------------------------
    # GRAFOLOGI: Hitung RIASEC tendency dari fitur tulisan tangan
    # ------------------------------------------------------------------
    def _compute_riasec_tendency(self, img: Dict[str, float]) -> Dict[str, float]:
        """
        Hitung kecenderungan RIASEC dari fitur tulisan tangan.
        Berdasarkan prinsip grafologi (ilmu analisis tulisan tangan).

        Skala output: 1-10 (sama dengan Dataset_Tulisan.xlsx)

        Referensi grafologi:
          Realistic     → tekanan kuat, huruf besar, tulisan padat
          Investigative → keterbacaan tinggi, rapi, garis lurus
          Artistic      → banyak ornamen, tulisan miring, kreativitas visual
          Social        → konektivitas tinggi (cursive), miring kanan, jarak lebar
          Enterprising  → huruf besar, tekanan kuat, miring kanan (dominan)
          Conventional  → sangat rapi, baseline konsisten, huruf seragam
        """
        sz  = img.get("letter_size_score",  5.0)   # ukuran huruf
        sl  = img.get("slant_angle",        5.0)   # kemiringan
        pr  = img.get("pressure_score",     5.0)   # tekanan pena
        sp  = img.get("spacing_score",      5.0)   # jarak
        rd  = img.get("readability_score",  5.0)   # keterbacaan
        nt  = img.get("neatness_score",     5.0)   # kerapian
        cn  = img.get("connectivity_score", 5.0)   # konektivitas (cursive)
        or_ = img.get("ornament_score",     3.0)   # ornamen/hiasan
        bl  = img.get("line_straightness",  7.0)   # kelurusan baris
        dn  = img.get("density_score",      5.0)   # kepadatan

        def clamp(val: float) -> float:
            return float(max(1.0, min(10.0, val)))

        # Realistic: teknis, fisik, tegas
        # → tekanan kuat + huruf besar + tinta padat + jarak rapat
        realistic = clamp(
            pr  * 0.35 +
            sz  * 0.30 +
            dn  * 0.20 +
            (10 - sp) * 0.15
        )

        # Investigative: analitis, teliti, sistematis
        # → sangat terbaca + rapi + garis lurus + sedikit ornamen
        investigative = clamp(
            rd  * 0.35 +
            nt  * 0.30 +
            bl  * 0.25 +
            (10 - or_) * 0.10
        )

        # Artistic: kreatif, ekspresif, imajinatif
        # → banyak ornamen + sedikit rapi + miring + tersambung
        artistic = clamp(
            or_ * 0.40 +
            (10 - nt) * 0.25 +
            cn  * 0.20 +
            sl  * 0.15
        )

        # Social: hangat, komunikatif, empati
        # → tulisan sambung (cursive) + sedikit miring kanan + jarak lebar
        social = clamp(
            cn  * 0.40 +
            sl  * 0.25 +
            sp  * 0.20 +
            (10 - pr) * 0.15
        )

        # Enterprising: dominan, percaya diri, pemimpin
        # → huruf besar + tekanan kuat + miring kanan
        enterprising = clamp(
            sz  * 0.35 +
            pr  * 0.30 +
            sl  * 0.25 +
            dn  * 0.10
        )

        # Conventional: teratur, detail, disiplin
        # → sangat rapi + garis lurus + terbaca + sedikit ornamen
        conventional = clamp(
            nt  * 0.40 +
            bl  * 0.30 +
            (10 - or_) * 0.20 +
            rd  * 0.10
        )

        return {
            "realistic":     round(realistic,     2),
            "investigative": round(investigative, 2),
            "artistic":      round(artistic,      2),
            "social":        round(social,        2),
            "enterprising":  round(enterprising,  2),
            "conventional":  round(conventional,  2),
        }

    # ------------------------------------------------------------------
    # Helper: Info kelengkapan data siswa
    # ------------------------------------------------------------------
    def _buat_info_kelengkapan(
        self,
        akademik_provided: Set[str],
        talent_provided:   Set[str],
    ) -> Dict:
        semua_akademik = set(DEFAULT_AKADEMIK.keys())
        semua_talent   = set(DEFAULT_TALENT.keys())

        akademik_kosong = semua_akademik - akademik_provided
        talent_kosong   = {k.replace("_t", "") for k in semua_talent - talent_provided}
        talent_diisi_display = {k.replace("_t", "") for k in talent_provided}

        persen_akademik = round(len(akademik_provided) / len(semua_akademik) * 100)
        persen_talent   = round(len(talent_provided)   / len(semua_talent)   * 100)

        return {
            "gambar": {
                "status": "diisi",
                "keterangan": "Gambar tulisan tangan berhasil dianalisis",
            },
            "akademik": {
                "diisi":          sorted(akademik_provided),
                "diisi_otomatis": sorted(akademik_kosong),
                "default_value":  75,
                "persen_lengkap": persen_akademik,
                "keterangan": (
                    f"{len(akademik_provided)} dari {len(semua_akademik)} mata pelajaran diisi. "
                    f"{len(akademik_kosong)} diisi otomatis nilai 75."
                ) if akademik_kosong else "Semua mata pelajaran terisi lengkap.",
            },
            "bakat": {
                "diisi":          sorted(talent_diisi_display),
                "diisi_otomatis": sorted(talent_kosong),
                "default_value":  5,
                "persen_lengkap": persen_talent,
                "keterangan": (
                    f"{len(talent_provided)} dari {len(semua_talent)} bakat diisi. "
                    f"{len(talent_kosong)} diisi otomatis nilai 5."
                ) if talent_kosong else "Semua bakat terisi lengkap.",
            },
            "catatan": (
                "Prediksi menggunakan analisis tulisan tangan (grafologi) sebagai dasar utama. "
                "Semakin lengkap data yang diisi, semakin personal rekomendasinya."
            ),
        }

    # ------------------------------------------------------------------
    # PREDIKSI DARI FILE (untuk testing lokal)
    # ------------------------------------------------------------------
    def predict_from_file(
        self,
        image_path: str,
        akademik: Optional[Dict] = None,
        talent:   Optional[Dict] = None,
    ) -> Dict:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.predict(image_bytes, akademik=akademik, talent=talent)

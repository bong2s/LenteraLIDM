"""
=============================================================
MODUL: predictor.py  (Dataset Real v3 — RIASEC Direct)
=============================================================
Kelas utama yang dipanggil FastAPI untuk prediksi lengkap.

ALUR PREDIKSI:
  1. Siapkan nilai akademik (default semua 0.0)
  2. Ekstrak 10 fitur gambar via OpenCV
  3. Prediksi RIASEC langsung dari fitur gambar
  4. Prediksi Rumpun Ilmu dari nilai akademik
  5. Cek apakah RIASEC sejalan dengan Rumpun Ilmu
  6. Hitung rata-rata nilai (hanya pelajaran yang diambil, nilai > 0)
  7. Rekomendasikan 3 jurusan (sejalan) atau 5 jurusan (berbeda)

OUTPUT API:
  - riasec_karakter   : tipe RIASEC + deskripsi + kekuatan
  - analisis_akademik : Rumpun Ilmu + nilai rata-rata + pelajaran kuat
  - perbandingan      : status SEJALAN/BERBEDA + penjelasan
  - rekomendasi_jurusan: TOP-3 atau TOP-5 Program Studi
  - fitur_tulisan     : 10 fitur numerik tulisan tangan
  - kelengkapan_data  : field mana yang diisi / default
=============================================================
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.image_processor import HandwritingFeatureExtractor
from src.preprocessing.data_loader import (
    RIASEC_TYPES, RIASEC_DESCRIPTIONS, RIASEC_TO_RUMPUN,
)
from src.models.riasec_classifier import RiasecClassifier, RumpunClassifier
from src.models.major_recommender import MajorRecommender

logger = logging.getLogger(__name__)


class Predictor:
    """
    Orkestrator prediksi lengkap analisis tulisan tangan.

    Cara pakai:
        predictor = Predictor()
        predictor.load_models("models/")
        result = predictor.predict(
            image_bytes=<bytes>,
            academic_scores={...},   # opsional
        )
    """

    def __init__(self):
        self.extractor  = HandwritingFeatureExtractor()
        self.riasec_clf: Optional[RiasecClassifier] = None
        self.rumpun_clf: Optional[RumpunClassifier] = None
        self.major_rec:  Optional[MajorRecommender] = None
        self._models_loaded = False

    def load_models(self, model_dir: str = "models") -> None:
        """Muat semua model dari folder model_dir."""
        riasec_path = os.path.join(model_dir, "riasec_model.pkl")
        rumpun_path = os.path.join(model_dir, "rumpun_model.pkl")
        major_path  = os.path.join(model_dir, "major_model.pkl")

        ok = True
        if os.path.exists(riasec_path):
            self.riasec_clf = RiasecClassifier.load(riasec_path)
        else:
            logger.warning(f"RIASEC model tidak ditemukan: {riasec_path}")
            ok = False

        if os.path.exists(rumpun_path):
            self.rumpun_clf = RumpunClassifier.load(rumpun_path)
        else:
            logger.warning(f"Rumpun model tidak ditemukan: {rumpun_path}")

        if os.path.exists(major_path):
            self.major_rec = MajorRecommender.load(major_path)
        else:
            logger.warning(f"Major model tidak ditemukan: {major_path}")
            self.major_rec = MajorRecommender()

        self._models_loaded = ok
        status = "✅ siap" if ok else "⚠️ sebagian (jalankan train_and_save.py)"
        logger.info(f"Model status: {status}")

    # ------------------------------------------------------------------
    # PREDIKSI UTAMA
    # ------------------------------------------------------------------
    def predict(
        self,
        image_bytes: bytes,
        academic_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Prediksi lengkap dari gambar + nilai akademik.

        Args:
            image_bytes    : bytes gambar tulisan tangan (jpg/png)
            academic_scores: nilai akademik 2 semester (opsional)
                             Keys: mat_s4, fis_s4, kim_s4, bio_s4,
                                   bind_s4, bing_s4, info_s4,
                                   mat_s5, fis_s5, kim_s5, bio_s5,
                                   bind_s5, bing_s5, info_s5
                             Nilai 0 = tidak ambil pelajaran (TETAP 0)

        Returns:
            dict lengkap hasil analisis (lihat schemas.py)
        """
        kelengkapan = {}

        # --- STEP 1: Siapkan nilai akademik (default 0.0) ---
        ac, ac_defaults = self._prepare_academic(academic_scores)
        kelengkapan["akademik"] = "lengkap" if not ac_defaults else f"default: {ac_defaults}"

        # --- STEP 2: Ekstrak 10 fitur gambar ---
        try:
            raw_features = self.extractor.extract_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Gagal proses gambar: {e}")
            raw_features = self.extractor._defaults()

        # --- STEP 3: Prediksi RIASEC langsung (bukan BigFive dulu) ---
        riasec_proba: Dict[str, float] = {}
        if self.riasec_clf is not None:
            try:
                riasec_dominant = self.riasec_clf.predict(raw_features)
                riasec_proba    = self.riasec_clf.predict_proba(raw_features)
            except Exception as e:
                logger.warning(f"RIASEC model error, pakai fallback: {e}")
                riasec_dominant = "Investigative"
        else:
            riasec_dominant = "Investigative"

        # --- STEP 4: Prediksi Rumpun Ilmu dari nilai akademik ---
        rumpun_proba: Dict[str, float] = {}
        if self.rumpun_clf is not None:
            try:
                rumpun_dominant = self.rumpun_clf.predict(ac)
                rumpun_proba    = self.rumpun_clf.predict_proba(ac)
            except Exception as e:
                logger.warning(f"Rumpun model error, pakai heuristik: {e}")
                rumpun_clf_fallback = RumpunClassifier()
                rumpun_dominant = rumpun_clf_fallback.heuristic_predict(ac)
        else:
            rumpun_clf_fallback = RumpunClassifier()
            rumpun_dominant = rumpun_clf_fallback.heuristic_predict(ac)

        # --- STEP 5: Perbandingan RIASEC vs Rumpun Ilmu ---
        sejalan    = self._check_sejalan(riasec_dominant, rumpun_dominant)
        penjelasan = self._generate_comparison_text(sejalan, riasec_dominant, rumpun_dominant)

        # --- STEP 6: Rata-rata nilai (hanya pelajaran yang diambil, nilai > 0) ---
        nilai_vals = [v for v in ac.values() if v > 0]
        avg_nilai  = float(np.mean(nilai_vals)) if nilai_vals else 0.0

        # --- STEP 7: Rekomendasi jurusan (3 jika sejalan, 5 jika berbeda) ---
        if self.major_rec is None:
            self.major_rec = MajorRecommender()

        top_n = 3 if sejalan else 5
        rekomendasi = self.major_rec.recommend(
            riasec_dominant=riasec_dominant,
            rumpun_ilmu=rumpun_dominant,
            riasec_proba=riasec_proba,
            avg_nilai=avg_nilai,
            top_n=top_n,
            sejalan=sejalan,
        )

        # --- STEP 8: Top mata pelajaran ---
        top_subjects: List[Dict] = []
        if self.rumpun_clf is not None:
            top_subjects = self.rumpun_clf.get_top_subjects(ac, top_n=3)

        # --- STEP 9: Susun output ---
        riasec_desc = RIASEC_DESCRIPTIONS.get(riasec_dominant, {})

        return {
            "riasec_karakter": {
                "dominant":   riasec_dominant,
                "karakter":   riasec_desc.get("karakter", ""),
                "deskripsi":  riasec_desc.get("deskripsi", ""),
                "kekuatan":   riasec_desc.get("kekuatan", []),
                "warna":      riasec_desc.get("warna", "#3498DB"),
                "skor":       {k: round(v * 100, 1) for k, v in riasec_proba.items()},
            },
            "analisis_akademik": {
                "rumpun_ilmu":         rumpun_dominant,
                "rumpun_probabilitas": {k: round(v * 100, 1) for k, v in rumpun_proba.items()},
                "nilai_rata_rata":     round(avg_nilai, 1),
                "mata_pelajaran_kuat": top_subjects,
            },
            "perbandingan": {
                "status":     "SEJALAN" if sejalan else "BERBEDA",
                "penjelasan": penjelasan,
            },
            "rekomendasi_jurusan": rekomendasi,
            "fitur_tulisan": {
                "letter_size":  round(raw_features.get("letter_size",  5.0), 2),
                "slant":        round(raw_features.get("slant",        5.0), 2),
                "pressure":     round(raw_features.get("pressure",     5.0), 2),
                "spacing":      round(raw_features.get("spacing",      5.0), 2),
                "readability":  round(raw_features.get("readability",  5.0), 2),
                "neatness":     round(raw_features.get("neatness",     5.0), 2),
                "connectivity": round(raw_features.get("connectivity", 5.0), 2),
                "ornament":     round(raw_features.get("ornament",     3.0), 2),
                "baseline":     round(raw_features.get("baseline",     7.0), 2),
                "density":      round(raw_features.get("density",      5.0), 2),
            },
            "kelengkapan_data": kelengkapan,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_academic(
        self, ac: Optional[Dict[str, float]]
    ) -> tuple:
        """
        Siapkan nilai akademik dengan default 0.0 untuk yang tidak dikirim.
        Nilai 0 yang eksplisit dari client TETAP 0 (artinya tidak ambil pelajaran).
        """
        all_keys = [
            "mat_s4", "fis_s4", "kim_s4", "bio_s4", "bind_s4", "bing_s4", "info_s4",
            "mat_s5", "fis_s5", "kim_s5", "bio_s5", "bind_s5", "bing_s5", "info_s5",
        ]
        result: Dict[str, float] = {k: 0.0 for k in all_keys}
        defaults_applied: List[str] = []

        if ac:
            for k in all_keys:
                if k in ac and ac[k] is not None:
                    result[k] = float(ac[k])   # 0 tetap 0
                else:
                    defaults_applied.append(k)
        else:
            defaults_applied = all_keys[:]

        return result, defaults_applied

    def _check_sejalan(self, riasec_dominant: str, rumpun_ilmu: str) -> bool:
        """Cek apakah tipe RIASEC sejalan dengan Rumpun Ilmu dari nilai akademik."""
        rumpun_cocok = RIASEC_TO_RUMPUN.get(riasec_dominant, [])
        return rumpun_ilmu in rumpun_cocok

    def _generate_comparison_text(
        self, sejalan: bool, riasec: str, rumpun: str
    ) -> str:
        """Buat teks penjelasan perbandingan RIASEC vs Rumpun Ilmu."""
        karakter = RIASEC_DESCRIPTIONS.get(riasec, {}).get("karakter", riasec)
        if sejalan:
            return (
                f"Karakter {karakter} dari analisis tulisan tangan sejalan dengan "
                f"kecenderungan akademik di bidang {rumpun}. Ini menunjukkan keselarasan "
                f"antara kepribadianmu dan kemampuan akademik yang kamu miliki."
            )
        else:
            return (
                f"Karakter {karakter} dari analisis tulisan tangan berbeda dengan "
                f"kecenderungan akademik di bidang {rumpun}. Kondisi ini wajar — "
                f"banyak siswa berhasil di bidang yang berbeda dari tipe dominan mereka. "
                f"Rekomendasi berikut mempertimbangkan kedua sisi potensimu."
            )

    @property
    def is_ready(self) -> bool:
        return self._models_loaded
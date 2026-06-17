"""
=============================================================
MODUL: predictor.py
=============================================================
TUJUAN:
  Kelas utama yang dipanggil oleh FastAPI untuk melakukan
  prediksi lengkap:
  1. Terima gambar + data siswa
  2. Ekstrak fitur gambar
  3. Gabungkan dengan fitur akademik & bakat
  4. Prediksi RIASEC
  5. Rekomendasikan Top-3 jurusan
  6. Kembalikan hasil terstruktur

KELAS INI ADALAH JEMBATAN antara model ML dan API.
=============================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.image_processor import HandwritingFeatureExtractor
from src.models.riasec_classifier import RIASECClassifier
from src.models.major_recommender import MajorRecommender
from src.preprocessing.data_loader import RIASEC_DESCRIPTIONS

logger = logging.getLogger(__name__)


class HandwritingPredictor:
    """
    Kelas prediksi end-to-end untuk tulisan tangan siswa.

    Cara pakai:
        predictor = HandwritingPredictor()
        predictor.load_models("models/")
        result = predictor.predict(image_bytes, akademik={...}, talent={...})
    """

    def __init__(self):
        self.image_extractor = HandwritingFeatureExtractor()
        self.riasec_clf = RIASECClassifier()
        self.major_rec = MajorRecommender()
        self.feature_meta: Dict = {}
        self._models_loaded = False

    # ------------------------------------------------------------------
    # MUAT MODEL
    # ------------------------------------------------------------------
    def load_models(self, model_dir: str = "models") -> bool:
        """
        Muat semua model dari folder.

        Args:
            model_dir: path ke folder berisi .pkl dan .json

        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            riasec_path = os.path.join(model_dir, "riasec_model.pkl")
            major_path = os.path.join(model_dir, "major_model.pkl")
            meta_path = os.path.join(model_dir, "feature_meta.json")

            self.riasec_clf.load(riasec_path)
            self.major_rec.load(major_path)

            with open(meta_path) as f:
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
        talent: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Prediksi lengkap dari gambar tulisan tangan.

        Args:
            image_bytes: isi file gambar dalam bytes
            akademik   : dict nilai akademik (boleh None)
                         contoh: {"matematika": 88, "ipa": 85, ...}
            talent     : dict skor bakat (boleh None)
                         contoh: {"logika": 8, "kreativitas": 7, ...}

        Returns:
            dict hasil prediksi lengkap:
            {
                "karakter": { "tipe": "Investigative", "nama": "Analitis & Ilmiah", ... },
                "riasec_skor": { "Investigative": 0.45, "Conventional": 0.20, ... },
                "rekomendasi_jurusan": [ {"rank": 1, "jurusan": "Informatika", ...}, ... ],
                "fitur_tulisan": { "letter_size_score": 5.2, ... },
                "feature_importance": [ {"fitur": "matematika", "importance": 0.12}, ... ]
            }
        """
        if not self._models_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load_models() terlebih dahulu.")

        # --- Step 1: Ekstrak fitur gambar ---
        logger.info("Mengekstrak fitur tulisan tangan dari gambar...")
        img_features = self.image_extractor.extract_from_bytes(image_bytes)

        # --- Step 2: Bangun DataFrame fitur gabungan ---
        feature_dict = {}
        feature_dict.update(img_features)

        if akademik:
            feature_dict.update(akademik)
        if talent:
            # Rename agar konsisten dengan training
            talent_renamed = {
                k + "_t" if k in ["komunikasi", "kepemimpinan", "kreativitas", "logika", "problem_solving"] else k: v
                for k, v in talent.items()
            }
            feature_dict.update(talent_renamed)

        X = pd.DataFrame([feature_dict])

        # --- Step 3: Prediksi RIASEC ---
        riasec_type = self.riasec_clf.predict(X)
        riasec_proba = self.riasec_clf.predict_proba(X)

        # --- Step 4: Rekomendasikan Jurusan ---
        top3_jurusan = self.major_rec.recommend_top3(
            X,
            riasec_type=riasec_type,
            riasec_proba=riasec_proba,
            akademik_scores=akademik,
        )

        # --- Step 5: Ambil deskripsi karakter RIASEC ---
        riasec_info = RIASEC_DESCRIPTIONS.get(riasec_type, {})

        # --- Step 6: Susun hasil ---
        result = {
            "karakter": {
                "tipe": riasec_type,
                "nama": riasec_info.get("karakter", riasec_type),
                "deskripsi": riasec_info.get("deskripsi", ""),
                "kekuatan": riasec_info.get("kekuatan", []),
                "warna": riasec_info.get("warna", "#666666"),
            },
            "riasec_skor": riasec_proba,
            "rekomendasi_jurusan": top3_jurusan,
            "fitur_tulisan": img_features,
            "feature_importance": self.riasec_clf.get_feature_importance(top_n=5),
        }

        logger.info(f"Prediksi selesai: RIASEC={riasec_type}, Jurusan={[j['jurusan'] for j in top3_jurusan]}")
        return result

    # ------------------------------------------------------------------
    # PREDIKSI DARI FILE (untuk testing di luar API)
    # ------------------------------------------------------------------
    def predict_from_file(
        self,
        image_path: str,
        akademik: Optional[Dict] = None,
        talent: Optional[Dict] = None,
    ) -> Dict:
        """
        Versi predict() yang menerima path file (untuk testing).
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.predict(image_bytes, akademik=akademik, talent=talent)
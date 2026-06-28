"""
=============================================================
MODUL: predictor.py  (Dataset Real v2)
=============================================================
Kelas utama yang dipanggil FastAPI untuk prediksi lengkap.

ALUR PREDIKSI:
  1. Terima bytes gambar + data siswa (nilai boleh tidak lengkap)
  2. Ekstrak 10 fitur gambar via OpenCV
  3. Hitung Big Five scores dari fitur gambar (grafologi)
  4. Prediksi Big Five dominant → mapping ke RIASEC
  5. Prediksi Rumpun Ilmu dari nilai akademik
  6. Hitung RIASEC dari kecerdasan Gardner (opsional)
  7. Fusi: gabungkan sinyal RIASEC dari gambar + talent
  8. Rekomendasikan TOP-3 Program Studi

OUTPUT API:
  - profil_karakter: Big Five + RIASEC dominant + deskripsi
  - rekomendasi_jurusan: TOP-3 Program Studi + alasan + skor
  - fitur_tulisan: 10 fitur numerik tulisan tangan
  - kelengkapan_data: field mana yang diisi / default
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
    RIASEC_TYPES, RIASEC_DESCRIPTIONS,
    BIG_FIVE_TYPES, BIGFIVE_TO_RIASEC,
    compute_riasec_from_gardner, GARDNER_TO_RIASEC_WEIGHTS,
)
from src.models.riasec_classifier import BigFiveClassifier, RumpunClassifier
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
            talent_scores={...},     # opsional
        )
    """

    def __init__(self):
        self.extractor  = HandwritingFeatureExtractor()
        self.bigfive_clf: Optional[BigFiveClassifier] = None
        self.rumpun_clf:  Optional[RumpunClassifier]  = None
        self.major_rec:   Optional[MajorRecommender]  = None
        self._models_loaded = False

    def load_models(self, model_dir: str = "models") -> None:
        """Muat semua model dari folder model_dir."""
        bigfive_path = os.path.join(model_dir, "bigfive_model.pkl")
        rumpun_path  = os.path.join(model_dir, "rumpun_model.pkl")
        major_path   = os.path.join(model_dir, "major_model.pkl")

        ok = True
        if os.path.exists(bigfive_path):
            self.bigfive_clf = BigFiveClassifier.load(bigfive_path)
        else:
            logger.warning(f"BigFive model tidak ditemukan: {bigfive_path}")
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
        talent_scores:   Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Prediksi lengkap dari gambar + data siswa.

        Args:
            image_bytes    : bytes gambar tulisan tangan (jpg/png)
            academic_scores: nilai akademik 2 semester (opsional)
                             Keys: mat_s4, fis_s4, kim_s4, bio_s4,
                                   bind_s4, bing_s4, info_s4,
                                   mat_s5, fis_s5, kim_s5, bio_s5,
                                   bind_s5, bing_s5, info_s5
            talent_scores  : kecerdasan Gardner (opsional)
                             Keys: linguistik, musikal, kinestetik,
                                   logika_mat, spasial, interpersonal,
                                   intrapersonal, naturalis

        Returns:
            dict lengkap hasil analisis (lihat schemas.py)
        """
        kelengkapan = {}

        # --- STEP 1: Nilai akademik ---
        ac, ac_defaults = self._prepare_academic(academic_scores)
        kelengkapan["akademik"] = "lengkap" if not ac_defaults else f"default: {ac_defaults}"

        # --- STEP 2: Talent Gardner ---
        talent, talent_defaults = self._prepare_talent(talent_scores)
        kelengkapan["talent"] = "lengkap" if not talent_defaults else f"default: {talent_defaults}"

        # --- STEP 3: Ekstrak fitur gambar ---
        try:
            img_result = self.extractor.extract_full_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Gagal proses gambar: {e}")
            img_result = self.extractor._full_defaults()

        raw_features   = img_result["raw_features"]
        bigfive_scores = img_result["big_five_scores"]     # {'Openness': 7.2, ...}
        bigfive_dominant = img_result["big_five_dominant"] # 'Openness'

        # --- STEP 4: BigFive dari model (override grafologi jika model ada) ---
        bigfive_proba = {}
        if self.bigfive_clf is not None:
            try:
                bigfive_dominant = self.bigfive_clf.predict(raw_features)
                bigfive_proba    = self.bigfive_clf.predict_proba(raw_features)
                # Perbarui bigfive_scores agar konsisten dengan prediksi model:
                # Kalikan skor grafologi (0-10) dengan probabilitas model sebagai bobot
                total_proba = sum(bigfive_proba.values()) or 1.0
                for trait in bigfive_scores:
                    model_weight = bigfive_proba.get(trait, 0.0) / total_proba
                    grafologi_raw = bigfive_scores[trait]
                    # Blend: 50% dari model weight (dikali 10 supaya skala sama) + 50% grafologi
                    bigfive_scores[trait] = round(
                        0.5 * model_weight * 10 + 0.5 * grafologi_raw, 2
                    )
                # Pastikan dominant dari model tercermin di scores (dinaikkan)
                bigfive_scores[bigfive_dominant] = max(
                    bigfive_scores[bigfive_dominant],
                    max(bigfive_scores.values()),
                )
            except Exception as e:
                logger.warning(f"BigFive model error, pakai grafologi: {e}")

        # --- STEP 5: RIASEC dari Big Five (sekarang mencerminkan model) ---
        riasec_from_image = self._bigfive_to_riasec(bigfive_scores)

        # --- STEP 6: RIASEC dari Gardner ---
        riasec_from_talent: Dict[str, float] = {}
        if any(v > 0 for v in talent.values()):
            riasec_from_talent = compute_riasec_from_gardner(talent)

        # --- STEP 7: Fusi RIASEC (gambar 60% + talent 40%) ---
        riasec_fused = self._fuse_riasec(riasec_from_image, riasec_from_talent)
        riasec_dominant = max(riasec_fused, key=riasec_fused.get)

        # --- STEP 8: Rumpun Ilmu dari nilai akademik ---
        rumpun_proba = {}
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

        # --- STEP 9: Rata-rata nilai akademik ---
        nilai_vals = [v for v in ac.values() if v > 0]
        avg_nilai = float(np.mean(nilai_vals)) if nilai_vals else 75.0

        # --- STEP 10: Rekomendasi TOP-3 ---
        if self.major_rec is None:
            self.major_rec = MajorRecommender()

        top3 = self.major_rec.recommend(
            riasec_dominant=riasec_dominant,
            rumpun_ilmu=rumpun_dominant,
            riasec_proba=riasec_fused,
            avg_nilai=avg_nilai,
            top_n=3,
        )

        # --- STEP 11: Top mata pelajaran ---
        top_subjects = []
        if self.rumpun_clf is not None:
            top_subjects = self.rumpun_clf.get_top_subjects(ac, top_n=3)

        # --- STEP 12: Susun output ---
        riasec_desc = RIASEC_DESCRIPTIONS.get(riasec_dominant, {})

        return {
            "profil_karakter": {
                "riasec_dominant":   riasec_dominant,
                "riasec_karakter":   riasec_desc.get("karakter", ""),
                "riasec_deskripsi":  riasec_desc.get("deskripsi", ""),
                "riasec_kekuatan":   riasec_desc.get("kekuatan", []),
                "riasec_warna":      riasec_desc.get("warna", "#3498DB"),
                "riasec_skor":       {k: round(v * 100, 1) for k, v in riasec_fused.items()},
                "big_five_dominant": bigfive_dominant,
                "big_five_skor":     bigfive_scores,
                "big_five_proba":    {k: round(v * 100, 1) for k, v in bigfive_proba.items()},
            },
            "analisis_akademik": {
                "rumpun_ilmu":       rumpun_dominant,
                "rumpun_proba":      {k: round(v * 100, 1) for k, v in rumpun_proba.items()},
                "nilai_rata_rata":   round(avg_nilai, 1),
                "mata_pelajaran_kuat": top_subjects,
            },
            "rekomendasi_jurusan": top3,
            "fitur_tulisan": {
                "letter_size":   round(raw_features.get("letter_size",   5.0), 2),
                "slant":         round(raw_features.get("slant",         5.0), 2),
                "pressure":      round(raw_features.get("pressure",      5.0), 2),
                "spacing":       round(raw_features.get("spacing",       5.0), 2),
                "readability":   round(raw_features.get("readability",   5.0), 2),
                "neatness":      round(raw_features.get("neatness",      5.0), 2),
                "connectivity":  round(raw_features.get("connectivity",  5.0), 2),
                "ornament":      round(raw_features.get("ornament",      3.0), 2),
                "baseline":      round(raw_features.get("baseline",      7.0), 2),
                "density":       round(raw_features.get("density",       5.0), 2),
            },
            "kelengkapan_data": kelengkapan,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_academic(
        self, ac: Optional[Dict[str, float]]
    ) -> tuple[Dict[str, float], List[str]]:
        """Siapkan nilai akademik dengan default 75 untuk yang kosong."""
        all_keys = [
            "mat_s4", "fis_s4", "kim_s4", "bio_s4", "bind_s4", "bing_s4", "info_s4",
            "mat_s5", "fis_s5", "kim_s5", "bio_s5", "bind_s5", "bing_s5", "info_s5",
        ]
        result = {k: 75.0 for k in all_keys}
        defaults_applied = []
        if ac:
            for k in all_keys:
                if k in ac and ac[k] is not None and ac[k] > 0:
                    result[k] = float(ac[k])
                else:
                    defaults_applied.append(k)
        else:
            defaults_applied = all_keys[:]
        return result, defaults_applied

    def _prepare_talent(
        self, talent: Optional[Dict[str, float]]
    ) -> tuple[Dict[str, float], List[str]]:
        """Siapkan skor Gardner dengan default 10 untuk yang kosong."""
        all_keys = [
            "linguistik", "musikal", "kinestetik", "logika_mat",
            "spasial", "interpersonal", "intrapersonal", "naturalis",
        ]
        result = {k: 0.0 for k in all_keys}
        defaults = []
        if talent:
            for k in all_keys:
                if k in talent and talent[k] is not None:
                    result[k] = float(talent[k])
                else:
                    defaults.append(k)
        else:
            defaults = all_keys[:]
        return result, defaults

    def _bigfive_to_riasec(self, bigfive_scores: Dict[str, float]) -> Dict[str, float]:
        """Konversi skor Big Five ke distribusi RIASEC."""
        riasec_raw = {t: 0.0 for t in RIASEC_TYPES}
        total_bf = sum(bigfive_scores.values()) or 1.0

        for trait, score in bigfive_scores.items():
            w = score / total_bf
            if trait not in BIGFIVE_TO_RIASEC:
                continue
            for riasec_type, weight in BIGFIVE_TO_RIASEC[trait].items():
                riasec_raw[riasec_type] += w * weight

        total = sum(riasec_raw.values()) or 1.0
        return {k: round(v / total, 4) for k, v in riasec_raw.items()}

    def _fuse_riasec(
        self,
        from_image: Dict[str, float],
        from_talent: Dict[str, float],
        image_weight: float = 0.60,
    ) -> Dict[str, float]:
        """Fusi RIASEC dari gambar (60%) dan talent (40%)."""
        if not from_talent or all(v == 0 for v in from_talent.values()):
            return from_image

        all_keys = set(list(from_image.keys()) + list(from_talent.keys()))
        fused = {}
        for k in all_keys:
            fused[k] = (
                image_weight * from_image.get(k, 0.0) +
                (1 - image_weight) * from_talent.get(k, 0.0)
            )
        total = sum(fused.values()) or 1.0
        return {k: round(v / total, 4) for k, v in sorted(fused.items(), key=lambda x: -x[1])}

    @property
    def is_ready(self) -> bool:
        return self._models_loaded

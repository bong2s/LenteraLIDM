"""
=============================================================
MODUL: predictor.py  (Dataset Real v3.1 — RIASEC + Minat)
=============================================================
Kelas utama yang dipanggil FastAPI untuk prediksi lengkap.

ALUR PREDIKSI:
  1. Siapkan nilai akademik (default semua 0.0)
  2. Hitung RIASEC minat dari kuesioner (jika dikirim)
  3. Ekstrak 10 fitur gambar via OpenCV
  4. Prediksi RIASEC langsung dari fitur gambar
  5. Prediksi Rumpun Ilmu dari nilai akademik
  6. Perbandingan RIASEC tulisan vs Rumpun Ilmu → sejalan/berbeda
  7. Perbandingan RIASEC tulisan vs RIASEC minat (jika ada)
  8. Hitung rata-rata nilai (hanya pelajaran yang diambil, nilai > 0)
  9. Tentukan top_n: 3 jika semua konsisten, 5 jika ada perbedaan
 10. Rekomendasikan jurusan

OUTPUT API:
  - riasec_karakter       : RIASEC dari tulisan + deskripsi + kekuatan
  - riasec_minat          : RIASEC dari kuesioner (None jika tidak dikirim)
  - analisis_akademik     : Rumpun Ilmu + nilai rata-rata + pelajaran kuat
  - perbandingan_akademik : status SEJALAN/BERBEDA (tulisan vs akademik)
  - perbandingan_minat    : status SEJALAN/BERBEDA (tulisan vs minat, jika ada)
  - rekomendasi_jurusan   : TOP-3 atau TOP-5 Program Studi
  - fitur_tulisan         : 10 fitur numerik tulisan tangan
  - kelengkapan_data      : field mana yang diisi / default
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

# Kode huruf → nama RIASEC
_CODE_TO_NAME = {
    "R": "Realistic", "I": "Investigative", "A": "Artistic",
    "S": "Social",    "E": "Enterprising",  "C": "Conventional",
}


class Predictor:
    """
    Orkestrator prediksi lengkap analisis tulisan tangan + kuesioner minat.

    Cara pakai:
        predictor = Predictor()
        predictor.load_models("models/")
        result = predictor.predict(
            image_bytes=<bytes>,
            academic_scores={...},   # opsional
            minat_scores={...},      # opsional — jawaban 24 soal atau 6 skor
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
        image_bytes:    bytes,
        academic_scores: Optional[Dict[str, float]] = None,
        minat_scores:    Optional[Dict[str, Any]]   = None,
    ) -> Dict[str, Any]:
        """
        Prediksi lengkap dari gambar + nilai akademik + kuesioner minat.

        Args:
            image_bytes     : bytes gambar tulisan tangan (jpg/png)
            academic_scores : nilai akademik 14 mapel (opsional)
            minat_scores    : jawaban kuesioner RIASEC (opsional)
                Format A — 24 jawaban: {"q_R1":3,"q_R2":2,...,"q_C4":3}
                Format B — 6 skor:     {"score_R":11,"score_I":12,...}

        Returns:
            dict lengkap hasil analisis (lihat schemas.py)
        """
        kelengkapan: Dict[str, str] = {}

        # --- STEP 1: Siapkan nilai akademik ---
        ac, ac_defaults = self._prepare_academic(academic_scores)
        kelengkapan["akademik"] = "lengkap" if not ac_defaults else "sebagian (sisanya default 0)"

        # --- STEP 2: Hitung RIASEC minat dari kuesioner (jika ada) ---
        minat_result: Optional[Dict] = None
        if minat_scores:
            try:
                minat_result = self._compute_minat(minat_scores)
                kelengkapan["minat"] = "lengkap"
            except Exception as e:
                logger.warning(f"Gagal proses data minat: {e}")
                kelengkapan["minat"] = "error"

        # --- STEP 3: Ekstrak 10 fitur gambar ---
        try:
            raw_features = self.extractor.extract_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Gagal proses gambar: {e}")
            raw_features = self.extractor._defaults()

        # --- STEP 4: Prediksi RIASEC dari tulisan tangan ---
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

        # --- STEP 5: Prediksi Rumpun Ilmu dari nilai akademik ---
        rumpun_proba: Dict[str, float] = {}
        if self.rumpun_clf is not None:
            try:
                rumpun_dominant = self.rumpun_clf.predict(ac)
                rumpun_proba    = self.rumpun_clf.predict_proba(ac)
            except Exception as e:
                logger.warning(f"Rumpun model error, pakai heuristik: {e}")
                rumpun_dominant = RumpunClassifier().heuristic_predict(ac)
        else:
            rumpun_dominant = RumpunClassifier().heuristic_predict(ac)

        # --- STEP 6: Perbandingan tulisan vs akademik ---
        sejalan_akademik = self._check_sejalan(riasec_dominant, rumpun_dominant)
        perban_akademik  = self._comparison_text_akademik(
            sejalan_akademik, riasec_dominant, rumpun_dominant
        )

        # --- STEP 7: Perbandingan tulisan vs minat (jika ada) ---
        perban_minat: Optional[Dict] = None
        sejalan_minat = True   # default True jika tidak ada minat
        if minat_result:
            sejalan_minat = (riasec_dominant == minat_result["dominant"])
            perban_minat  = self._comparison_text_minat(
                sejalan_minat, riasec_dominant, minat_result["dominant"]
            )

        # --- STEP 8: Rata-rata nilai (hanya pelajaran yang diambil) ---
        nilai_vals = [v for v in ac.values() if v > 0]
        avg_nilai  = float(np.mean(nilai_vals)) if nilai_vals else 0.0

        # --- STEP 9: Tentukan top_n ---
        # TOP-3 hanya jika tulisan SEJALAN akademik DAN (tidak ada minat ATAU minat juga sejalan)
        # TOP-5 jika ada ketidakcocokan apapun
        semua_konsisten = sejalan_akademik and sejalan_minat
        top_n = 3 if semua_konsisten else 5

        # --- STEP 10: Rekomendasi jurusan ---
        if self.major_rec is None:
            self.major_rec = MajorRecommender()

        rekomendasi = self.major_rec.recommend(
            riasec_dominant = riasec_dominant,
            rumpun_ilmu     = rumpun_dominant,
            riasec_proba    = riasec_proba,
            avg_nilai       = avg_nilai,
            top_n           = top_n,
            sejalan         = sejalan_akademik,
        )

        # --- Top mata pelajaran ---
        top_subjects: List[Dict] = []
        if self.rumpun_clf is not None:
            top_subjects = self.rumpun_clf.get_top_subjects(ac, top_n=3)

        # --- Susun output ---
        riasec_desc = RIASEC_DESCRIPTIONS.get(riasec_dominant, {})

        result: Dict[str, Any] = {
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
            "perbandingan_akademik": {
                "status":     "SEJALAN" if sejalan_akademik else "BERBEDA",
                "penjelasan": perban_akademik,
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

        # Tambahkan minat jika ada
        if minat_result:
            result["riasec_minat"]      = minat_result
            result["perbandingan_minat"] = perban_minat

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_academic(
        self, ac: Optional[Dict[str, float]]
    ) -> tuple:
        """
        Siapkan nilai akademik dengan default 0.0 untuk yang tidak dikirim.
        Nilai 0 eksplisit dari client TETAP 0 (artinya tidak ambil pelajaran).
        """
        all_keys = [
            "mat_s4", "fis_s4", "kim_s4", "bio_s4", "bind_s4", "bing_s4", "info_s4",
        ]
        result: Dict[str, float] = {k: 0.0 for k in all_keys}
        defaults_applied: List[str] = []

        if ac:
            for k in all_keys:
                if k in ac and ac[k] is not None:
                    result[k] = float(ac[k])
                else:
                    defaults_applied.append(k)
        else:
            defaults_applied = all_keys[:]

        return result, defaults_applied

    def _compute_minat(self, minat: Dict) -> Dict:
        """
        Hitung skor RIASEC dari jawaban kuesioner.

        Terima salah satu format:
          A. 24 jawaban raw: q_R1..q_C4 (0-4 per soal, max 16 per tipe)
          B. 6 skor langsung: score_R..score_C

        Returns dict berisi dominant, karakter, skor_raw, skor_persen.
        """
        codes  = ["R", "I", "A", "S", "E", "C"]
        scores: Dict[str, float] = {}

        # Deteksi format — prioritaskan format A (raw questions)
        has_raw = any(f"q_{c}1" in minat for c in codes)
        if has_raw:
            for code in codes:
                total = sum(float(minat.get(f"q_{code}{i}", 0) or 0) for i in range(1, 5))
                scores[code] = total
        else:
            for code in codes:
                scores[code] = float(minat.get(f"score_{code}", 0) or 0)

        # Dominant = kode dengan skor tertinggi
        dominant_code = max(scores, key=lambda c: scores[c])
        dominant_name = _CODE_TO_NAME[dominant_code]

        # Normalisasi ke persen
        total = sum(scores.values()) or 1.0
        skor_raw    = {_CODE_TO_NAME[c]: round(scores[c], 1)              for c in codes}
        skor_persen = {_CODE_TO_NAME[c]: round(scores[c] / total * 100, 1) for c in codes}

        desc = RIASEC_DESCRIPTIONS.get(dominant_name, {})
        return {
            "dominant":    dominant_name,
            "karakter":    desc.get("karakter", ""),
            "deskripsi":   desc.get("deskripsi", ""),
            "kekuatan":    desc.get("kekuatan", []),
            "warna":       desc.get("warna", "#3498DB"),
            "skor_raw":    skor_raw,
            "skor_persen": skor_persen,
        }

    def _check_sejalan(self, riasec_dominant: str, rumpun_ilmu: str) -> bool:
        """Cek apakah RIASEC tulisan sejalan dengan Rumpun Ilmu akademik."""
        return rumpun_ilmu in RIASEC_TO_RUMPUN.get(riasec_dominant, [])

    def _comparison_text_akademik(
        self, sejalan: bool, riasec: str, rumpun: str
    ) -> str:
        """Teks perbandingan RIASEC tulisan vs Rumpun Ilmu."""
        karakter = RIASEC_DESCRIPTIONS.get(riasec, {}).get("karakter", riasec)
        if sejalan:
            return (
                f"Karakter {karakter} dari analisis tulisan tangan sejalan dengan "
                f"kecenderungan akademik di bidang {rumpun}. Ini menunjukkan keselarasan "
                f"antara kepribadianmu dan kemampuan akademik yang kamu miliki."
            )
        return (
            f"Karakter {karakter} dari analisis tulisan tangan berbeda dengan "
            f"kecenderungan akademik di bidang {rumpun}. Kondisi ini wajar — "
            f"banyak siswa berhasil di bidang yang berbeda dari tipe dominan mereka. "
            f"Rekomendasi berikut mempertimbangkan kedua sisi potensimu."
        )

    def _comparison_text_minat(
        self, sejalan: bool, riasec_tulisan: str, riasec_minat: str
    ) -> Dict:
        """Teks perbandingan RIASEC tulisan vs RIASEC kuesioner minat."""
        k_tulisan = RIASEC_DESCRIPTIONS.get(riasec_tulisan, {}).get("karakter", riasec_tulisan)
        k_minat   = RIASEC_DESCRIPTIONS.get(riasec_minat,   {}).get("karakter", riasec_minat)
        if sejalan:
            penjelasan = (
                f"Tulisan tanganmu menunjukkan karakter {k_tulisan}, dan kuesioner minat "
                f"juga mengkonfirmasi hal yang sama. Ini pertanda kuat bahwa profil "
                f"{riasec_tulisan} adalah karakter dominanmu yang sesungguhnya."
            )
        else:
            penjelasan = (
                f"Tulisan tanganmu menunjukkan karakter {k_tulisan}, namun kuesioner minat "
                f"menunjukkan kecenderungan {k_minat} ({riasec_minat}). Perbedaan ini wajar — "
                f"tulisan tangan mencerminkan kepribadian bawah sadar, sedangkan kuesioner "
                f"mencerminkan minat yang kamu sadari. Rekomendasi mempertimbangkan keduanya."
            )
        return {"status": "SEJALAN" if sejalan else "BERBEDA", "penjelasan": penjelasan}

    @property
    def is_ready(self) -> bool:
        return self._models_loaded
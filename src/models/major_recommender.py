"""
=============================================================
MODUL: major_recommender.py  (Dataset Real v2)
=============================================================
Merekomendasikan TOP-3 Program Studi dari:
  - RIASEC dominant (dari analisis tulisan tangan)
  - Rumpun Ilmu (dari nilai akademik)
  - Skor kecerdasan Gardner (opsional)
  - Rata-rata nilai akademik (sebagai tiebreaker)

Dataset sumber: Dataset_AkademikN.xlsx (140 siswa, 82 Program Studi)
Strategi:
  1. Filter Program Studi berdasarkan Rumpun Ilmu yang diprediksi
  2. Skor tambahan dari RIASEC → Rumpun Ilmu affinity
  3. Tiebreaker: rata-rata Tingkat Kesesuaian di dataset asli
=============================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple

from src.preprocessing.data_loader import (
    RIASEC_TO_RUMPUN, RIASEC_DESCRIPTIONS,
    RUMPUN_NORM, BIG_FIVE_TYPES,
)

logger = logging.getLogger(__name__)


# Daftar Program Studi per Rumpun Ilmu
# (dibangun otomatis dari dataset, tapi ada hardcode fallback)
DEFAULT_PRODI_RUMPUN: Dict[str, List[str]] = {
    "STEM": [
        "S1 Teknik Informatika", "D4 Teknik Informatika", "S1 Sistem Informasi",
        "S1 Matematika", "S1 Statistika", "S1 Kimia", "S1 Biologi",
        "S1 Teknik Sipil", "S1 Teknik Industri", "S1 Teknik Lingkungan",
        "S1 Teknik Biomedis", "S1 Teknik Geologi", "S1 Ilmu Keolahragaan",
        "S1 Ilmu Kesehatan Masyarakat", "S1 Kesehatan Masyarakat",
        "S1 Farmasi", "D3 Farmasi", "D3 Asuransi Kesehatan",
        "S1 Agroekoteknologi", "S1 Perencanaan Wilayah dan Kota",
        "S1 Pend Teknik Elektro",
    ],
    "Sosial Humaniora": [
        "S1 Ilmu Komunikasi", "S1 Psikologi", "S1 Sosiologi",
        "S1 Antropologi", "S1 Hukum", "S1 Ilmu Sejarah",
        "S1 Sastra Inggris", "S1 Hubungan Internasional",
        "S1 Pendidikan Bahasa Indonesia", "S1 Pendidikan Bahasa Inggris",
        "S1 Pendidikan Bahasa Arab",
    ],
    "Bisnis Manajemen": [
        "S1 Manajemen", "S1 Akuntansi", "S1 Bisnis Digital",
        "S1 Administrasi Bisnis", "S1 Manajemen Bisnis",
        "D4 Perbankan", "D3 Kesekretariatan",
    ],
    "Pendidikan": [
        "S1 Pendidikan Guru Sekolah Dasar", "S1 Bimbingan dan Konseling",
        "S1 Manajemen Pendidikan", "S1 Teknologi Pendidikan",
        "S1 Pendidikan IPA", "S1 Pendidikan Ekonomi",
        "S1 Pendidikan Sejarah", "S1 Pendidikan Olahraga",
        "S1 Pendidikan Jasmani", "S1 Pendidikan Agama Islam",
        "S1 Pendidikan Guru Pendidikan Anak Usia Dini",
    ],
    "Seni Kreatif": [
        "S1 Desain Komunikasi Visual", "S1 Seni Rupa Murni",
        "S1 Sendratasik", "D4 Seni Kuliner",
    ],
}

# RIASEC → Rumpun yang paling cocok dengan bobot (berurutan dari terbaik)
RIASEC_RUMPUN_AFFINITY: Dict[str, List[Tuple[str, float]]] = {
    "Realistic":      [("STEM", 0.9), ("Bisnis Manajemen", 0.4), ("Sosial Humaniora", 0.2)],
    "Investigative":  [("STEM", 0.9), ("Sosial Humaniora", 0.5), ("Pendidikan", 0.3)],
    "Artistic":       [("Seni Kreatif", 0.9), ("Sosial Humaniora", 0.5), ("Pendidikan", 0.3)],
    "Social":         [("Pendidikan", 0.9), ("Sosial Humaniora", 0.7), ("Bisnis Manajemen", 0.4)],
    "Enterprising":   [("Bisnis Manajemen", 0.9), ("Sosial Humaniora", 0.5), ("STEM", 0.3)],
    "Conventional":   [("Bisnis Manajemen", 0.9), ("STEM", 0.6), ("Pendidikan", 0.3)],
}


class MajorRecommender:
    """
    Merekomendasikan TOP-3 Program Studi berdasarkan kombinasi:
      - RIASEC dominant
      - Rumpun Ilmu dari nilai akademik
      - Profil kecerdasan Gardner (opsional)
      - Rata-rata nilai akademik (tiebreaker)
    """

    def __init__(self):
        # Mapping: rumpun → list program studi + metadata
        self._prodi_data: Dict[str, List[Dict]] = {}
        # Mapping: program studi → rumpun
        self._prodi_to_rumpun: Dict[str, str] = {}
        # Rata-rata tingkat kesesuaian per program studi (dari dataset)
        self._prodi_avg_score: Dict[str, float] = {}

    @property
    def n_programs(self) -> int:
        return sum(len(v) for v in self._prodi_data.values())

    def build_from_dataset(self, df: pd.DataFrame) -> None:
        """
        Bangun index Program Studi dari DataFrame akademik.

        Args:
            df: DataFrame hasil DatasetLoader.load_akademik()
                Kolom: rumpun_ilmu, program_studi, tingkat_kesesuaian
        """
        # Rata-rata tingkat kesesuaian per program studi
        avg_score = (
            df.groupby("program_studi")["tingkat_kesesuaian"]
            .mean()
            .round(2)
            .to_dict()
        )
        self._prodi_avg_score = avg_score

        # Group program studi per rumpun
        grouped = df.groupby("rumpun_ilmu")["program_studi"].unique()

        self._prodi_data = {}
        self._prodi_to_rumpun = {}

        for rumpun, prodis in grouped.items():
            rumpun_key = rumpun.strip()
            self._prodi_data[rumpun_key] = []
            for prodi in sorted(set(prodis)):
                prodi = prodi.strip()
                score = avg_score.get(prodi, 3.0)
                self._prodi_data[rumpun_key].append({
                    "program_studi": prodi,
                    "avg_kesesuaian": score,
                })
                self._prodi_to_rumpun[prodi] = rumpun_key

        # Tambahkan program dari hardcode default yang tidak ada di dataset
        for rumpun, prodis in DEFAULT_PRODI_RUMPUN.items():
            if rumpun not in self._prodi_data:
                self._prodi_data[rumpun] = []
            existing = {d["program_studi"] for d in self._prodi_data[rumpun]}
            for prodi in prodis:
                if prodi not in existing:
                    self._prodi_data[rumpun].append({
                        "program_studi": prodi,
                        "avg_kesesuaian": 3.5,
                    })
                    self._prodi_to_rumpun[prodi] = rumpun

        logger.info(f"MajorRecommender: {self.n_programs} program studi dari "
                    f"{len(self._prodi_data)} rumpun")

    def recommend(
        self,
        riasec_dominant: str,
        rumpun_ilmu: str,
        riasec_proba: Optional[Dict[str, float]] = None,
        avg_nilai: float = 80.0,
        top_n: int = 3,
    ) -> List[Dict]:
        """
        Rekomendasikan TOP-N Program Studi.

        Args:
            riasec_dominant: tipe RIASEC utama siswa
            rumpun_ilmu    : prediksi Rumpun Ilmu dari nilai akademik
            riasec_proba   : dict probabilitas RIASEC (opsional)
            avg_nilai      : rata-rata nilai akademik (70-100)
            top_n          : jumlah rekomendasi

        Returns:
            List of dict dengan key:
              program_studi, rumpun_ilmu, alasan, skor_kesesuaian,
              prediksi_nilai (estimasi IPK awal)
        """
        if not self._prodi_data:
            return self._fallback_recommend(riasec_dominant, top_n)

        # Kumpulkan semua program studi dengan skor
        scored: List[Tuple[str, str, float]] = []  # (prodi, rumpun, skor)

        for rumpun, prodis in self._prodi_data.items():
            # Bobot affinitas RIASEC → Rumpun
            riasec_affinity = self._riasec_rumpun_score(riasec_dominant, rumpun)
            # Bonus jika rumpun cocok dengan prediksi akademik
            rumpun_match_bonus = 0.3 if rumpun == rumpun_ilmu else 0.0

            for prodi_meta in prodis:
                prodi = prodi_meta["program_studi"]
                base_score = prodi_meta.get("avg_kesesuaian", 3.0) / 5.0  # norm ke 0-1

                # Skor akhir
                final_score = (
                    0.40 * riasec_affinity      +   # kecocokan RIASEC
                    0.30 * rumpun_match_bonus   +   # kecocokan nilai akademik
                    0.20 * base_score           +   # tingkat kesesuaian dataset
                    0.10 * (avg_nilai / 100.0)      # pengaruh nilai rata-rata
                )

                # Bonus tambahan dari probabilitas RIASEC jika ada
                if riasec_proba:
                    for r_type, r_score in riasec_proba.items():
                        affinity = self._riasec_rumpun_score(r_type, rumpun)
                        final_score += 0.05 * r_score * affinity

                scored.append((prodi, rumpun, final_score))

        # Sort dan ambil top_n tanpa duplikat per rumpun
        scored.sort(key=lambda x: -x[2])

        results = []
        seen_rumpun = {}
        for prodi, rumpun, skor in scored:
            if len(results) >= top_n:
                break
            seen_count = seen_rumpun.get(rumpun, 0)
            if seen_count >= 2:
                continue
            seen_rumpun[rumpun] = seen_count + 1
            results.append({
                "program_studi":  prodi,
                "rumpun_ilmu":    rumpun,
                "alasan":         self._generate_reason(prodi, rumpun, riasec_dominant, avg_nilai),
                "skor_kesesuaian": round(min(skor * 5, 5.0), 2),  # kembalikan ke skala 1-5
                "prediksi_ipk":   self._estimate_ipk(avg_nilai, skor),
            })

        return results[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _riasec_rumpun_score(self, riasec: str, rumpun: str) -> float:
        """Skor affinitas RIASEC → Rumpun Ilmu (0.0 – 1.0)."""
        affinities = RIASEC_RUMPUN_AFFINITY.get(riasec, [])
        for rumpun_name, weight in affinities:
            if rumpun_name == rumpun:
                return weight
        return 0.1

    def _generate_reason(
        self, prodi: str, rumpun: str, riasec: str, avg_nilai: float
    ) -> str:
        desc = RIASEC_DESCRIPTIONS.get(riasec, {})
        karakter = desc.get("karakter", riasec)
        nilai_ket = "sangat baik" if avg_nilai >= 90 else "baik" if avg_nilai >= 80 else "cukup"
        return (
            f"Profil {karakter} sangat sesuai dengan bidang {rumpun}. "
            f"Dengan nilai akademik {nilai_ket} ({avg_nilai:.0f}), "
            f"{prodi} menjadi pilihan yang kuat untuk pengembangan kariermu."
        )

    def _estimate_ipk(self, avg_nilai: float, skor: float) -> str:
        """Estimasi IPK awal berdasarkan nilai rata-rata dan skor kesesuaian."""
        base = 2.0 + (avg_nilai - 70) / 30 * 1.5 + skor * 0.3
        base = max(2.5, min(4.0, base))
        lo = max(2.5, base - 0.25)
        hi = min(4.0, base + 0.15)
        return f"{lo:.1f} – {hi:.1f}"

    def _fallback_recommend(self, riasec: str, top_n: int) -> List[Dict]:
        """Fallback jika model belum dibangun dari dataset."""
        fallback_map = {
            "Realistic":      ["S1 Teknik Informatika", "S1 Teknik Sipil", "S1 Ilmu Keolahragaan"],
            "Investigative":  ["S1 Matematika", "S1 Kimia", "S1 Psikologi"],
            "Artistic":       ["S1 Desain Komunikasi Visual", "S1 Seni Rupa Murni", "S1 Ilmu Komunikasi"],
            "Social":         ["S1 Pendidikan Guru Sekolah Dasar", "S1 Psikologi", "S1 Bimbingan dan Konseling"],
            "Enterprising":   ["S1 Manajemen", "S1 Administrasi Bisnis", "S1 Hubungan Internasional"],
            "Conventional":   ["S1 Akuntansi", "S1 Statistika", "S1 Sistem Informasi"],
        }
        prodis = fallback_map.get(riasec, ["S1 Manajemen", "S1 Psikologi", "S1 Ilmu Komunikasi"])
        return [
            {
                "program_studi": p,
                "rumpun_ilmu":   self._prodi_to_rumpun.get(p, "Umum"),
                "alasan":        f"Rekomendasi berdasarkan profil RIASEC {riasec}.",
                "skor_kesesuaian": 3.5,
                "prediksi_ipk":  "3.0 – 3.5",
            }
            for p in prodis[:top_n]
        ]

    # ------------------------------------------------------------------
    # Simpan / Muat
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "prodi_data":       self._prodi_data,
            "prodi_to_rumpun":  self._prodi_to_rumpun,
            "prodi_avg_score":  self._prodi_avg_score,
        }, path, compress=3)
        logger.info(f"MajorRecommender disimpan: {path}")

    @classmethod
    def load(cls, path: str) -> "MajorRecommender":
        obj = cls()
        data = joblib.load(path)
        obj._prodi_data      = data["prodi_data"]
        obj._prodi_to_rumpun = data["prodi_to_rumpun"]
        obj._prodi_avg_score = data.get("prodi_avg_score", {})
        logger.info(f"MajorRecommender dimuat: {path} ({obj.n_programs} program)")
        return obj

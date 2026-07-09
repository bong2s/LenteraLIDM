"""
=============================================================
MODUL: data_loader.py  (Dataset Real v3 — RIASEC Direct)
=============================================================
Dataset yang digunakan:
  1. Dataset_TulisanN.csv  — 235 baris fitur tulisan tangan (pra-ekstraksi)
     Target: riasec_primary (kode huruf R/I/A/S/E/C)
  2. Dataset_AkademikN.xlsx — 140 siswa, 14 nilai pelajaran (2 semester),
     digabung menjadi 7 nilai (rata-rata Smt 4 & 5) sebagai fitur model.
     Target: Rumpun Ilmu (5 kategori) + Program Studi

Arsitektur ML:
  CSV fitur tulisan → RiasecClassifier (10 fitur → RIASEC langsung)
  Akademik          → RumpunClassifier (7 nilai rata-rata → Rumpun Ilmu)
  Perbandingan RIASEC vs Rumpun → sejalan / berbeda
=============================================================
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# RIASEC Types (Holland)
# ------------------------------------------------------------------
RIASEC_TYPES = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]

# Kode huruf → nama lengkap RIASEC (dari kolom riasec_primary di CSV)
RIASEC_CODE_TO_NAME: Dict[str, str] = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional",
}


# ------------------------------------------------------------------
# Rumpun Ilmu & Program Studi
# ------------------------------------------------------------------
RUMPUN_ILMU = [
    "STEM",
    "Sosial Humaniora",
    "Bisnis Manajemen",
    "Pendidikan",
    "Seni Kreatif",
]

RUMPUN_NORM: Dict[str, str] = {
    "STEM (Science, Technology, Engineering, Mathematics)": "STEM",
    "Sosial Humaniora":          "Sosial Humaniora",
    "Bisnis dan Manajemen":      "Bisnis Manajemen",
    "Pendidikan":                "Pendidikan",
    "Seni dan Industri Kreatif": "Seni Kreatif",
}

# RIASEC → Rumpun Ilmu yang sejalan (untuk logika perbandingan)
RIASEC_TO_RUMPUN: Dict[str, List[str]] = {
    "Realistic":     ["STEM"],
    "Investigative": ["STEM", "Sosial Humaniora"],
    "Artistic":      ["Seni Kreatif", "Sosial Humaniora"],
    "Social":        ["Pendidikan", "Sosial Humaniora"],
    "Enterprising":  ["Bisnis Manajemen"],
    "Conventional":  ["Bisnis Manajemen", "STEM"],
}

# Deskripsi karakter RIASEC (ditampilkan ke siswa)
RIASEC_DESCRIPTIONS: Dict[str, Dict] = {
    "Realistic": {
        "karakter": "Praktis & Teknis",
        "deskripsi": (
            "Kamu tipe orang yang suka bekerja dengan tangan dan benda nyata. "
            "Terampil secara teknis, berorientasi hasil, dan lebih suka langsung "
            "action daripada banyak teori."
        ),
        "kekuatan": ["Keterampilan teknis tinggi", "Berorientasi hasil", "Reliabel & konsisten"],
        "warna": "#E67E22",
    },
    "Investigative": {
        "karakter": "Analitis & Ilmiah",
        "deskripsi": (
            "Kamu pemikir mendalam yang suka menganalisis, meneliti, dan memecahkan "
            "masalah kompleks. Nyaman dengan data, logika, dan eksperimen."
        ),
        "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis & sistematis", "Rasa ingin tahu tinggi"],
        "warna": "#2980B9",
    },
    "Artistic": {
        "karakter": "Kreatif & Ekspresif",
        "deskripsi": (
            "Kamu jiwa kreatif yang penuh imajinasi. Suka mengekspresikan diri "
            "dan punya sensitivitas estetika yang tinggi."
        ),
        "kekuatan": ["Kreativitas & inovasi tinggi", "Kepekaan estetika", "Berpikir out-of-the-box"],
        "warna": "#8E44AD",
    },
    "Social": {
        "karakter": "Empatik & Komunikatif",
        "deskripsi": (
            "Kamu hangat dan peduli pada orang lain. Pandai berkomunikasi, punya "
            "empati tinggi, dan senang membantu atau membimbing."
        ),
        "kekuatan": ["Komunikasi interpersonal", "Empati & kepedulian", "Kemampuan mengajar/membimbing"],
        "warna": "#27AE60",
    },
    "Enterprising": {
        "karakter": "Pemimpin & Wirausaha",
        "deskripsi": (
            "Kamu tipe pemimpin alami yang ambisius dan percaya diri. Suka "
            "memengaruhi orang, mengambil keputusan, dan mengejar target."
        ),
        "kekuatan": ["Jiwa kepemimpinan", "Persuasi & negosiasi", "Orientasi pencapaian"],
        "warna": "#C0392B",
    },
    "Conventional": {
        "karakter": "Teratur & Detail",
        "deskripsi": (
            "Kamu terorganisir, cermat, dan suka bekerja dengan sistem yang jelas. "
            "Teliti dan handal dalam mengelola data atau administrasi."
        ),
        "kekuatan": ["Ketelitian & presisi", "Kemampuan organisasi", "Reliabel & disiplin"],
        "warna": "#7F8C8D",
    },
}


# ------------------------------------------------------------------
# Nama kolom CSV → nama fitur internal
# ------------------------------------------------------------------
CSV_FEATURE_COLS: Dict[str, str] = {
    "letter_size_score":  "letter_size",
    "slant_angle":        "slant",
    "pressure_score":     "pressure",
    "spacing_score":      "spacing",
    "readability_score":  "readability",
    "neatness_score":     "neatness",
    "connectivity_score": "connectivity",
    "ornament_score":     "ornament",
    "line_straightness":  "baseline",
    "density_score":      "density",
}


# ------------------------------------------------------------------
# 7 Mapel (rata-rata Semester 4 & 5) — dipakai sebagai fitur RumpunClassifier
# ------------------------------------------------------------------
AKADEMIK_S4_S5_PAIRS: Dict[str, Tuple[str, str]] = {
    "mat":  ("mat_s4",  "mat_s5"),
    "fis":  ("fis_s4",  "fis_s5"),
    "kim":  ("kim_s4",  "kim_s5"),
    "bio":  ("bio_s4",  "bio_s5"),
    "bind": ("bind_s4", "bind_s5"),
    "bing": ("bing_s4", "bing_s5"),
    "info": ("info_s4", "info_s5"),
}


def _rata_rata_semester(nilai_s4: float, nilai_s5: float) -> float:
    """
    Hitung rata-rata nilai semester 4 & 5 untuk satu mata pelajaran.

    Nilai 0 berarti "tidak mengambil pelajaran ini" pada semester tsb,
    sehingga TIDAK ikut dirata-rata (supaya nilai tidak timpang jika
    mapel hanya diambil di salah satu semester):
      - keduanya > 0  → rata-rata biasa
      - salah satu 0  → pakai nilai yang > 0 saja
      - keduanya 0    → hasil 0 (mapel memang tidak diambil)
    """
    vals = [v for v in (nilai_s4, nilai_s5) if v and v > 0]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _norm_program_studi(nama: str) -> str:
    """Bersihkan nama Program Studi (hapus spasi berlebih, standarisasi)."""
    if not isinstance(nama, str):
        return str(nama)
    s = nama.strip()
    s = re.sub(r'\s+', ' ', s)
    fixes = {
        "S1 Manajement":                            "S1 Manajemen",
        "S1 Manajemen ":                            "S1 Manajemen",
        "S1  Manajemen Bisnis":                     "S1 Manajemen Bisnis",
        "Akuntansi ":                               "S1 Akuntansi",
        "Akuntansi":                                "S1 Akuntansi",
        "S1 Akuntansi ":                            "S1 Akuntansi",
        "D4 Teknik Informatika ":                   "D4 Teknik Informatika",
        "Teknik Informatika ":                      "S1 Teknik Informatika",
        "S1 Kesehatan Masyarakat ":                 "S1 Kesehatan Masyarakat",
        "S1 Kesehatam Masyarakat ":                 "S1 Kesehatan Masyarakat",
        "S1 Kesehatam Masyarakat":                  "S1 Kesehatan Masyarakat",
        "S1 Ilmu Kesehatan Masyarakat ":            "S1 Ilmu Kesehatan Masyarakat",
        "S1 Pendidikan Bahasa Inggris ":            "S1 Pendidikan Bahasa Inggris",
        "Pendidikan bahasa inggris ":               "S1 Pendidikan Bahasa Inggris",
        "S1 Bimbingan dan Konseling ":              "S1 Bimbingan dan Konseling",
        "Ilmu Komunikasi ":                         "S1 Ilmu Komunikasi",
        "Ilmu Komunikasi":                          "S1 Ilmu Komunikasi",
        "S1 Teknik Industri ":                      "S1 Teknik Industri",
        "S1 Teknik Sipil ":                         "S1 Teknik Sipil",
        "S1 Psikologi ":                            "S1 Psikologi",
        "Manajemen":                                "S1 Manajemen",
        "Sistem Informasi":                         "S1 Sistem Informasi",
        "S1 Matematika Murni":                      "S1 Matematika",
        "S1 Pendidikan Guru Sekolah Dasar ":        "S1 Pendidikan Guru Sekolah Dasar",
        "S1 Pendidikan Guru Sekolah Dasar (PGSD)":  "S1 Pendidikan Guru Sekolah Dasar",
        "S1 Ilmu Keolahragaan ":                    "S1 Ilmu Keolahragaan",
        "S1 Pendidikan Jasmani Kesehatan dan Rekreasi ": "S1 Pendidikan Jasmani",
        "S1 Pendidikan Kepelatihan Olahraga ":      "S1 Pendidikan Olahraga",
        "D3 Kesekretariatan ":                      "D3 Kesekretariatan",
        "S1 Farmasi ":                              "S1 Farmasi",
        "S1 Teknik Geologi ":                       "S1 Teknik Geologi",
        "S1 Teknik Lingkungan ":                    "S1 Teknik Lingkungan",
        "S1 Hubungan Hubungan Internasional":       "S1 Hubungan Internasional",
        "S1 Pendidikan Agam Islam":                 "S1 Pendidikan Agama Islam",
        "Seni Kuliner":                             "D4 Seni Kuliner",
        "S2 Linguistik Murni ":                     "S1 Sastra Indonesia",
    }
    return fixes.get(s, s)


class DatasetLoader:
    """
    Memuat dan menyiapkan dataset real untuk training ML.

    Cara pakai:
        loader = DatasetLoader(data_dir='data_new')
        X, y   = loader.load_tulisan_csv()
        df_ak  = loader.load_akademik()
    """

    def __init__(self, data_dir: str = "data_new"):
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Dataset Tulisan: CSV pra-ekstraksi → RiasecClassifier
    # ------------------------------------------------------------------
    def load_tulisan_csv(
        self,
        csv_path: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Muat dataset tulisan dari CSV pra-ekstraksi.

        Kolom yang diharapkan di CSV:
          - letter_size_score, slant_angle, pressure_score, spacing_score,
            readability_score, neatness_score, connectivity_score,
            ornament_score, line_straightness, density_score
          - riasec_primary : kode huruf (R / I / A / S / E / C)
          - dataset_split  : 'train' / 'test' (opsional)

        Returns:
            (X, y) — DataFrame 10 fitur dan Series label RIASEC nama lengkap
        """
        if csv_path is None:
            csv_path = os.path.join(self.data_dir, "Dataset_TulisanN.csv")

        df = pd.read_csv(csv_path)

        if split and "dataset_split" in df.columns:
            df = df[df["dataset_split"] == split].reset_index(drop=True)

        # Rename kolom CSV ke nama fitur internal
        df = df.rename(columns=CSV_FEATURE_COLS)

        # Konversi kode huruf ke nama lengkap RIASEC
        df["riasec_label"] = df["riasec_primary"].map(RIASEC_CODE_TO_NAME)
        df = df.dropna(subset=["riasec_label"])

        feature_cols = list(CSV_FEATURE_COLS.values())
        X = df[feature_cols].copy()
        y = df["riasec_label"]

        logger.info(f"CSV tulisan: {len(X)} sampel, distribusi: {y.value_counts().to_dict()}")
        return X, y

    # ------------------------------------------------------------------
    # Dataset Akademik
    # ------------------------------------------------------------------
    def load_akademik(self) -> pd.DataFrame:
        """
        Muat Dataset_AkademikN.xlsx dan standarisasi kolom.

        PENTING: Nilai 0 = siswa tidak mengambil pelajaran → DIBIARKAN 0.
        Hanya sel Excel yang benar-benar kosong (NaN) yang diisi 0.

        Returns:
            DataFrame dengan kolom terstandarisasi + kolom target bersih
        """
        path = os.path.join(self.data_dir, "Dataset_AkademikN.xlsx")
        df = pd.read_excel(path)

        rename = {
            "Matematika Semester 4":       "mat_s4",
            "Fisika Semester 4":           "fis_s4",
            "Kimia Semester 4":            "kim_s4",
            "Biologi Semester 4":          "bio_s4",
            "Bahasa Indonesia Semester 4": "bind_s4",
            "Bahasa Inggris Semester 4":   "bing_s4",
            "Informatika Semester 4":      "info_s4",
            "Matematika Semester 5":       "mat_s5",
            "Fisika Semester 5":           "fis_s5",
            "Kimia Semester 5":            "kim_s5",
            "Biologi Semester 5":          "bio_s5",
            "Bahasa Indonesia Semester 5": "bind_s5",
            "Bahasa Inggris Semester 5":   "bing_s5",
            "Informatika Semester 5":      "info_s5",
            "Rumpun Ilmu":                 "rumpun_ilmu",
            "Program Studi":               "program_studi",
            "Tingkat kesesuaian ":         "tingkat_kesesuaian",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Normalisasi nama rumpun ilmu
        df["rumpun_ilmu"] = df["rumpun_ilmu"].map(
            lambda x: RUMPUN_NORM.get(str(x).strip(), str(x).strip())
        )

        # Bersihkan nama program studi
        df["program_studi"] = df["program_studi"].map(_norm_program_studi)

        # Nilai 0 = tidak ambil pelajaran → TETAP 0
        # Hanya NaN (sel Excel kosong) yang diisi 0
        non_target = {"rumpun_ilmu", "program_studi", "tingkat_kesesuaian"}
        num_cols = [c for c in df.columns if c not in non_target]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # ------------------------------------------------------------
        # Gabungkan Semester 4 & 5 menjadi rata-rata per mata pelajaran
        # (hanya 7 mapel yang akan diinput user, tanpa split semester).
        # PENTING: nilai 0 = "tidak ambil pelajaran". Agar rata-rata tidak
        # timpang saat mapel hanya diambil di salah satu semester, rata-rata
        # dihitung hanya dari nilai yang > 0. Jika keduanya 0 → tetap 0.
        # ------------------------------------------------------------
        for mapel, (col_s4, col_s5) in AKADEMIK_S4_S5_PAIRS.items():
            if col_s4 in df.columns and col_s5 in df.columns:
                df[mapel] = df.apply(
                    lambda row: _rata_rata_semester(row[col_s4], row[col_s5]),
                    axis=1,
                )
            elif col_s4 in df.columns:
                df[mapel] = df[col_s4]
            elif col_s5 in df.columns:
                df[mapel] = df[col_s5]
            else:
                df[mapel] = 0.0

        # Kolom semester 4/5 asli tidak lagi dibutuhkan sebagai fitur model,
        # tapi dibiarkan ada di DataFrame (tidak dihapus) untuk keperluan lain.

        logger.info(f"Akademik: {df.shape[0]} siswa, Rumpun: {df['rumpun_ilmu'].value_counts().to_dict()}")
        return df

    @property
    def akademik_feature_cols(self) -> List[str]:
        """Daftar kolom fitur akademik (bukan target) — 7 mapel (rata-rata smt 4 & 5)."""
        return ["mat", "fis", "kim", "bio", "bind", "bing", "info"]
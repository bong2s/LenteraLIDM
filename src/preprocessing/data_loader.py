"""
=============================================================
MODUL: data_loader.py
=============================================================
TUJUAN:
  Memuat dan menggabungkan 3 dataset (Akademik, Talent, Tulisan)
  menjadi satu DataFrame yang siap dilatih oleh model ML.

ALUR:
  1. Baca Dataset_Akademik.xlsx   → fitur nilai pelajaran + skill
  2. Baca Dataset_Talent.xlsx     → fitur bakat
  3. Baca Dataset_Tulisan.xlsx    → fitur tulisan + skor RIASEC
  4. Gabungkan berdasarkan sample_id
  5. Encode label (RIASEC & jurusan) ke angka
=============================================================
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Konstanta: Daftar jurusan dan karakter RIASEC
# ------------------------------------------------------------------

RIASEC_TYPES = [
    "Realistic",      # R - Realistik  : teknis, praktis, fisik
    "Investigative",  # I - Investigatif: analitis, ilmiah, penasaran
    "Artistic",       # A - Artistik    : kreatif, ekspresi diri
    "Social",         # S - Sosial      : membantu, mengajar, empati
    "Enterprising",   # E - Enterprising: memimpin, wirausaha, persuasif
    "Conventional",   # C - Konvensional: teratur, detail, administrasi
]

# Mapping RIASEC → deskripsi karakter siswa
RIASEC_DESCRIPTIONS = {
    "Realistic": {
        "karakter": "Praktis & Teknis",
        "deskripsi": (
            "Kamu adalah tipe orang yang suka bekerja dengan tangan dan "
            "benda nyata. Kamu terampil secara teknis, tidak suka banyak "
            "teori, lebih suka langsung action. Kamu cocok di lingkungan "
            "lapangan, lab, atau workshop."
        ),
        "kekuatan": ["Keterampilan teknis tinggi", "Berorientasi hasil", "Reliabel & konsisten"],
        "warna": "#E67E22",
    },
    "Investigative": {
        "karakter": "Analitis & Ilmiah",
        "deskripsi": (
            "Kamu adalah pemikir mendalam yang suka menganalisis, "
            "meneliti, dan memecahkan masalah kompleks. Kamu penasaran "
            "dengan cara kerja sesuatu dan suka mencari tahu jawaban. "
            "Kamu nyaman dengan data, logika, dan eksperimen."
        ),
        "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis & sistematis", "Rasa ingin tahu tinggi"],
        "warna": "#2980B9",
    },
    "Artistic": {
        "karakter": "Kreatif & Ekspresif",
        "deskripsi": (
            "Kamu adalah jiwa kreatif yang penuh imajinasi. Kamu suka "
            "mengekspresikan diri melalui berbagai media dan tidak suka "
            "terkekang aturan kaku. Kamu punya sensitivitas estetika yang "
            "tinggi dan melihat dunia dengan cara yang unik."
        ),
        "kekuatan": ["Kreativitas & inovasi tinggi", "Kemampuan estetika", "Berpikir out-of-the-box"],
        "warna": "#8E44AD",
    },
    "Social": {
        "karakter": "Empatik & Komunikatif",
        "deskripsi": (
            "Kamu adalah orang yang hangat dan peduli pada orang lain. "
            "Kamu pandai berkomunikasi, punya empati tinggi, dan senang "
            "membantu, mengajar, atau membimbing. Kamu tumbuh di lingkungan "
            "yang melibatkan interaksi manusia."
        ),
        "kekuatan": ["Komunikasi interpersonal", "Empati & kepedulian", "Kemampuan mengajar/membimbing"],
        "warna": "#27AE60",
    },
    "Enterprising": {
        "karakter": "Pemimpin & Wirausaha",
        "deskripsi": (
            "Kamu adalah tipe pemimpin alami yang ambisius dan percaya diri. "
            "Kamu suka memengaruhi orang, mengambil keputusan, dan "
            "mengejar target. Kamu cocok di dunia bisnis, politik, atau "
            "posisi manajerial."
        ),
        "kekuatan": ["Jiwa kepemimpinan", "Kemampuan persuasi & negosiasi", "Orientasi pada pencapaian"],
        "warna": "#C0392B",
    },
    "Conventional": {
        "karakter": "Teratur & Detail",
        "deskripsi": (
            "Kamu adalah orang yang terorganisir, cermat, dan suka bekerja "
            "dengan sistem yang jelas. Kamu teliti, patuh pada prosedur, "
            "dan handal dalam mengelola data atau administrasi. Kamu cocok "
            "di lingkungan yang terstruktur dan presisi tinggi."
        ),
        "kekuatan": ["Ketelitian & presisi tinggi", "Kemampuan organisasi", "Reliable & disiplin"],
        "warna": "#7F8C8D",
    },
}

# Daftar semua jurusan yang bisa direkomendasikan
ALL_MAJORS = [
    "Akuntansi", "Informatika", "Sistem Informasi", "DKV", "Psikologi",
    "Manajemen", "Teknik Sipil", "Teknik Elektro", "Teknik Mesin",
    "Kedokteran", "Farmasi", "Ilmu Komunikasi", "Pendidikan", "Hukum",
    "Arsitektur", "Biologi", "Kimia", "Matematika", "Statistik",
    "Bisnis Internasional",
]


class DatasetLoader:
    """
    Kelas untuk memuat dan menggabungkan semua dataset.

    Cara pakai:
        loader = DatasetLoader(data_dir="data/raw")
        X_train, X_test, y_riasec_train, ... = loader.load_and_split()
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # FUNGSI UTAMA: load_all()
    # ------------------------------------------------------------------
    def load_all(self) -> pd.DataFrame:
        """
        Baca dan gabungkan semua dataset menjadi 1 DataFrame.

        Returns:
            DataFrame dengan semua fitur + label (dominant_riasec, recommended_major)
        """
        akademik = self._load_akademik()
        talent = self._load_talent()
        tulisan = self._load_tulisan()

        # Gabungkan berdasarkan sample_id
        df = akademik.merge(talent, on="sample_id", suffixes=("", "_talent"))
        df = df.merge(tulisan, on="sample_id", suffixes=("", "_tulisan"))

        # Pilih 1 kolom dominant_riasec dan recommended_major
        # (ambil dari dataset tulisan sebagai ground truth)
        df["dominant_riasec"] = df["dominant_riasec_tulisan"]
        df["recommended_major"] = df["recommended_major_tulisan"]
        df = df.drop(columns=[c for c in df.columns if c.endswith(("_talent", "_tulisan"))])

        logger.info(f"Dataset gabungan: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Kembalikan nama kolom per kelompok fitur.
        Berguna untuk memahami fitur mana yang dipakai model.
        """
        return {
            "akademik": [
                "agama", "pancasila", "bahasa_indonesia", "matematika",
                "ipa", "ips", "bahasa_inggris", "pjok", "informatika",
                "seni_budaya", "logika", "kreativitas", "komunikasi",
                "kepemimpinan", "problem_solving", "teamwork", "literasi",
                "numerasi",
            ],
            "talent": [
                "komunikasi_t", "kepemimpinan_t", "kreativitas_t", "logika_t",
                "teknologi", "riset", "seni", "olahraga", "organisasi",
                "kewirausahaan", "kerja_tim", "problem_solving_t",
            ],
            "tulisan": [
                "letter_size", "slant", "pressure", "spacing",
                "readability", "neatness", "ornament", "connectivity",
                "baseline",
                "realistic", "investigative", "artistic", "social",
                "enterprising", "conventional",
            ],
        }

    # ------------------------------------------------------------------
    # Load per-dataset
    # ------------------------------------------------------------------
    def _load_akademik(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "Dataset_Akademik.xlsx")
        df = pd.read_excel(path)
        logger.info(f"Akademik: {df.shape}")
        return df

    def _load_talent(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "Dataset_Talent.xlsx")
        df = pd.read_excel(path)
        # Rename agar tidak tabrakan saat merge
        rename_map = {
            "komunikasi": "komunikasi_t",
            "kepemimpinan": "kepemimpinan_t",
            "kreativitas": "kreativitas_t",
            "logika": "logika_t",
            "problem_solving": "problem_solving_t",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        logger.info(f"Talent: {df.shape}")
        return df

    def _load_tulisan(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "Dataset_Tulisan.xlsx")
        df = pd.read_excel(path)

        # Encode kolom kategorik menjadi angka
        style_map = {"Neat Print": 0, "Formal Cursive": 1, "Technical Block": 2,
                     "Artistic Script": 3, "Bold Print": 4}
        type_map = {"Print": 0, "Cursive": 1, "Mixed": 2}
        doc_map = {"Notebook": 0, "Formal Letter": 1, "Art Paper": 2, "Grid Paper": 3}

        df["style_category"] = df["style_category"].map(style_map).fillna(0)
        df["writing_type"] = df["writing_type"].map(type_map).fillna(0)
        df["document_type"] = df["document_type"].map(doc_map).fillna(0)

        # Konversi kolom RIASEC ke numerik (bisa berupa string)
        for col in ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(5)

        logger.info(f"Tulisan: {df.shape}")
        return df
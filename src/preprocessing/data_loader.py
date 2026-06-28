"""
=============================================================
MODUL: data_loader.py  (Dataset Real v2)
=============================================================
Dataset baru yang digunakan:
  1. Dataset_AkademikN.xlsx  — 140 siswa, 14 nilai pelajaran (2 semester)
     Target: Rumpun Ilmu (5 kategori) + Program Studi
  2. Dataset_TalentN.xlsx    — Kecerdasan Majemuk Gardner (8 dimensi)
     Target: Job Profession
  3. Dataset_TulisanN/       — 221 gambar tulisan tangan
     Struktur: folder = label Big Five (Openness, Conscientiousness, dll)

Arsitektur ML:
  Gambar  → Big Five → RIASEC (mapping psikologi)
  Akademik → Rumpun Ilmu (5 kategori) + Program Studi
  Talent  → Gardner MI (8 kecerdasan) → bobot RIASEC
=============================================================
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Big Five Personality (OCEAN)
# ------------------------------------------------------------------
BIG_FIVE_TYPES = [
    "Openness",          # Terbuka, kreatif, imajinatif
    "Conscientiousness", # Teratur, disiplin, bertanggung jawab
    "Extraversion",      # Sosial, energik, asertif
    "Agreeableness",     # Kooperatif, empati, ramah
    "Neuroticism",       # Sensitif, cemas, emosional
]

# ------------------------------------------------------------------
# RIASEC Types (Holland)
# ------------------------------------------------------------------
RIASEC_TYPES = [
    "Realistic",      # Teknis, praktis, fisik
    "Investigative",  # Analitis, ilmiah, penasaran
    "Artistic",       # Kreatif, ekspresif
    "Social",         # Membantu, mengajar, empati
    "Enterprising",   # Memimpin, wirausaha
    "Conventional",   # Teratur, detail, administrasi
]

# ------------------------------------------------------------------
# Mapping Big Five → RIASEC (berdasarkan psikologi karier)
# Setiap Big Five memiliki bobot pada 6 tipe RIASEC
# ------------------------------------------------------------------
BIGFIVE_TO_RIASEC: Dict[str, Dict[str, float]] = {
    "Openness": {
        "Realistic": 0.2, "Investigative": 0.7,
        "Artistic": 0.9,  "Social": 0.4,
        "Enterprising": 0.3, "Conventional": 0.1,
    },
    "Conscientiousness": {
        "Realistic": 0.6, "Investigative": 0.5,
        "Artistic": 0.2,  "Social": 0.3,
        "Enterprising": 0.5, "Conventional": 0.9,
    },
    "Extraversion": {
        "Realistic": 0.3, "Investigative": 0.2,
        "Artistic": 0.4,  "Social": 0.8,
        "Enterprising": 0.9, "Conventional": 0.3,
    },
    "Agreeableness": {
        "Realistic": 0.3, "Investigative": 0.3,
        "Artistic": 0.5,  "Social": 0.9,
        "Enterprising": 0.4, "Conventional": 0.4,
    },
    "Neuroticism": {
        "Realistic": 0.2, "Investigative": 0.4,
        "Artistic": 0.7,  "Social": 0.3,
        "Enterprising": 0.2, "Conventional": 0.3,
    },
}


def bigfive_to_riasec_dominant(bigfive_type: str) -> str:
    """Konversi Big Five personality type ke RIASEC dominan."""
    if bigfive_type not in BIGFIVE_TO_RIASEC:
        return "Investigative"
    weights = BIGFIVE_TO_RIASEC[bigfive_type]
    return max(weights, key=weights.get)


# ------------------------------------------------------------------
# Rumpun Ilmu & Program Studi
# ------------------------------------------------------------------
RUMPUN_ILMU = [
    "STEM",               # Science, Technology, Engineering, Mathematics
    "Sosial Humaniora",   # Ilmu Sosial, Hukum, Sastra, dll
    "Bisnis Manajemen",   # Akuntansi, Manajemen, Ekonomi, dll
    "Pendidikan",         # Semua jurusan keguruan
    "Seni Kreatif",       # DKV, Seni Rupa, Musik, dll
]

# Normalisasi nama Rumpun Ilmu dari dataset
RUMPUN_NORM: Dict[str, str] = {
    "STEM (Science, Technology, Engineering, Mathematics)": "STEM",
    "Sosial Humaniora": "Sosial Humaniora",
    "Bisnis dan Manajemen": "Bisnis Manajemen",
    "Pendidikan": "Pendidikan",
    "Seni dan Industri Kreatif": "Seni Kreatif",
}

# RIASEC → Rumpun Ilmu yang paling cocok (untuk rekomendasi)
RIASEC_TO_RUMPUN: Dict[str, List[str]] = {
    "Realistic":      ["STEM", "Sosial Humaniora"],
    "Investigative":  ["STEM", "Sosial Humaniora"],
    "Artistic":       ["Seni Kreatif", "Sosial Humaniora"],
    "Social":         ["Pendidikan", "Sosial Humaniora", "Bisnis Manajemen"],
    "Enterprising":   ["Bisnis Manajemen", "Sosial Humaniora"],
    "Conventional":   ["Bisnis Manajemen", "STEM"],
}

# Deskripsi karakter RIASEC (ditampilkan ke siswa)
RIASEC_DESCRIPTIONS: Dict[str, Dict] = {
    "Realistic": {
        "karakter": "Praktis & Teknis",
        "deskripsi": (
            "Kamu tipe orang yang suka bekerja dengan tangan dan benda nyata. "
            "Terampil secara teknis, berorientasi hasil, dan lebih suka langsung "
            "action daripada banyak teori. Cocok di lingkungan lapangan, lab, atau workshop."
        ),
        "kekuatan": ["Keterampilan teknis tinggi", "Berorientasi hasil", "Reliabel & konsisten"],
        "warna": "#E67E22",
    },
    "Investigative": {
        "karakter": "Analitis & Ilmiah",
        "deskripsi": (
            "Kamu pemikir mendalam yang suka menganalisis, meneliti, dan memecahkan "
            "masalah kompleks. Penasaran dengan cara kerja sesuatu dan nyaman dengan "
            "data, logika, dan eksperimen."
        ),
        "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis & sistematis", "Rasa ingin tahu tinggi"],
        "warna": "#2980B9",
    },
    "Artistic": {
        "karakter": "Kreatif & Ekspresif",
        "deskripsi": (
            "Kamu jiwa kreatif yang penuh imajinasi. Suka mengekspresikan diri "
            "melalui berbagai media dan tidak suka terkekang aturan kaku. Punya "
            "sensitivitas estetika yang tinggi dan cara pandang yang unik."
        ),
        "kekuatan": ["Kreativitas & inovasi tinggi", "Kepekaan estetika", "Berpikir out-of-the-box"],
        "warna": "#8E44AD",
    },
    "Social": {
        "karakter": "Empatik & Komunikatif",
        "deskripsi": (
            "Kamu hangat dan peduli pada orang lain. Pandai berkomunikasi, punya "
            "empati tinggi, dan senang membantu atau membimbing. Tumbuh subur di "
            "lingkungan yang melibatkan interaksi manusia."
        ),
        "kekuatan": ["Komunikasi interpersonal", "Empati & kepedulian", "Kemampuan mengajar/membimbing"],
        "warna": "#27AE60",
    },
    "Enterprising": {
        "karakter": "Pemimpin & Wirausaha",
        "deskripsi": (
            "Kamu tipe pemimpin alami yang ambisius dan percaya diri. Suka "
            "memengaruhi orang, mengambil keputusan, dan mengejar target. "
            "Cocok di dunia bisnis, manajemen, atau posisi kepemimpinan."
        ),
        "kekuatan": ["Jiwa kepemimpinan", "Persuasi & negosiasi", "Orientasi pencapaian"],
        "warna": "#C0392B",
    },
    "Conventional": {
        "karakter": "Teratur & Detail",
        "deskripsi": (
            "Kamu terorganisir, cermat, dan suka bekerja dengan sistem yang jelas. "
            "Teliti, patuh pada prosedur, dan handal dalam mengelola data atau "
            "administrasi. Cocok di lingkungan yang terstruktur dan presisi tinggi."
        ),
        "kekuatan": ["Ketelitian & presisi", "Kemampuan organisasi", "Reliabel & disiplin"],
        "warna": "#7F8C8D",
    },
}


def _norm_program_studi(nama: str) -> str:
    """Bersihkan nama Program Studi (hapus spasi berlebih, standarisasi)."""
    if not isinstance(nama, str):
        return str(nama)
    s = nama.strip()
    s = re.sub(r'\s+', ' ', s)
    # Standarisasi beberapa yang duplikat
    fixes = {
        "S1 Manajement": "S1 Manajemen",
        "S1 Manajemen ": "S1 Manajemen",
        "S1  Manajemen Bisnis": "S1 Manajemen Bisnis",
        "Akuntansi ": "S1 Akuntansi",
        "Akuntansi": "S1 Akuntansi",
        "S1 Akuntansi ": "S1 Akuntansi",
        "D4 Teknik Informatika ": "D4 Teknik Informatika",
        "Teknik Informatika ": "S1 Teknik Informatika",
        "S1 Kesehatan Masyarakat ": "S1 Kesehatan Masyarakat",
        "S1 Kesehatam Masyarakat ": "S1 Kesehatan Masyarakat",
        "S1 Kesehatam Masyarakat":  "S1 Kesehatan Masyarakat",
        "S1 Ilmu Kesehatan Masyarakat ": "S1 Ilmu Kesehatan Masyarakat",
        "S1 Pendidikan Bahasa Inggris ": "S1 Pendidikan Bahasa Inggris",
        "Pendidikan bahasa inggris ": "S1 Pendidikan Bahasa Inggris",
        "S1 Bimbingan dan Konseling ": "S1 Bimbingan dan Konseling",
        "Ilmu Komunikasi ": "S1 Ilmu Komunikasi",
        "Ilmu Komunikasi": "S1 Ilmu Komunikasi",
        "S1 Teknik Industri ": "S1 Teknik Industri",
        "S1 Teknik Sipil ": "S1 Teknik Sipil",
        "S1 Psikologi ": "S1 Psikologi",
        "S1 Manajemen ": "S1 Manajemen",
        "Manajemen": "S1 Manajemen",
        "Sistem Informasi": "S1 Sistem Informasi",
        "S1 Matematika Murni": "S1 Matematika",
        "S1 Pendidikan Guru Sekolah Dasar ": "S1 Pendidikan Guru Sekolah Dasar",
        "S1 Pendidikan Guru Sekolah Dasar (PGSD)": "S1 Pendidikan Guru Sekolah Dasar",
        "S1 Ilmu Keolahragaan ": "S1 Ilmu Keolahragaan",
        "S1 Pendidikan Jasmani Kesehatan dan Rekreasi ": "S1 Pendidikan Jasmani",
        "S1 Pendidikan Kepelatihan Olahraga ": "S1 Pendidikan Olahraga",
        "D3 Kesekretariatan ": "D3 Kesekretariatan",
        "S1 Farmasi ": "S1 Farmasi",
        "S1 Teknik Geologi ": "S1 Teknik Geologi",
        "S1 Teknik Lingkungan ": "S1 Teknik Lingkungan",
        "S1 Hubungan Hubungan Internasional": "S1 Hubungan Internasional",
        "S1 Pendidikan Agam Islam": "S1 Pendidikan Agama Islam",
        "Seni Kuliner": "D4 Seni Kuliner",
        "S2 Linguistik Murni ": "S1 Sastra Indonesia",
    }
    return fixes.get(s, s)


class DatasetLoader:
    """
    Memuat dan menyiapkan dataset real untuk training ML.

    Cara pakai:
        loader = DatasetLoader(data_dir='data_new')
        X_img, y_bigfive = loader.load_tulisan_dataset(img_base_dir)
        df_akademik = loader.load_akademik()
        df_talent   = loader.load_talent()
    """

    def __init__(self, data_dir: str = "data_new"):
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Dataset Tulisan: gambar per folder Big Five
    # ------------------------------------------------------------------
    def load_tulisan_image_paths(
        self,
        img_base: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Muat path gambar dan label Big Five dari struktur folder.

        Struktur folder yang diharapkan:
          img_base/
            Openness/img1.jpg
            Conscientiousness/img2.jpg
            ...

        Returns:
            (paths, labels) — list path gambar dan label Big Five-nya
        """
        if img_base is None:
            img_base = os.path.join(self.data_dir, "Dataset_TulisanN")

        paths, labels = [], []
        for category in BIG_FIVE_TYPES:
            cat_dir = os.path.join(img_base, category)
            if not os.path.isdir(cat_dir):
                logger.warning(f"Folder tidak ditemukan: {cat_dir}")
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(cat_dir, fname))
                    labels.append(category)
            logger.info(f"  {category}: {sum(1 for l in labels if l == category)} gambar dimuat")

        logger.info(f"Total gambar tulisan: {len(paths)}")
        return paths, labels

    # ------------------------------------------------------------------
    # Dataset Akademik
    # ------------------------------------------------------------------
    def load_akademik(self) -> pd.DataFrame:
        """
        Muat Dataset_AkademikN.xlsx dan standarisasi kolom.

        Returns:
            DataFrame dengan kolom terstandarisasi + kolom target bersih
        """
        path = os.path.join(self.data_dir, "Dataset_AkademikN.xlsx")
        df = pd.read_excel(path)

        # Rename kolom ke nama pendek yang mudah dipakai
        rename = {
            "Matematika Semester 4":    "mat_s4",
            "Fisika Semester 4":        "fis_s4",
            "Kimia Semester 4":         "kim_s4",
            "Biologi Semester 4":       "bio_s4",
            "Bahasa Indonesia Semester 4": "bind_s4",
            "Bahasa Inggris Semester 4":   "bing_s4",
            "Informatika Semester 4":   "info_s4",
            "Matematika Semester 5":    "mat_s5",
            "Fisika Semester 5":        "fis_s5",
            "Kimia Semester 5":         "kim_s5",
            "Biologi Semester 5":       "bio_s5",
            "Bahasa Indonesia Semester 5": "bind_s5",
            "Bahasa Inggris Semester 5":   "bing_s5",
            "Informatika Semester 5":   "info_s5",
            "Rumpun Ilmu":              "rumpun_ilmu",
            "Program Studi":            "program_studi",
            "Tingkat kesesuaian ":      "tingkat_kesesuaian",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Normalisasi nama rumpun ilmu
        df["rumpun_ilmu"] = df["rumpun_ilmu"].map(
            lambda x: RUMPUN_NORM.get(str(x).strip(), str(x).strip())
        )

        # Bersihkan nama program studi
        df["program_studi"] = df["program_studi"].map(_norm_program_studi)

        # Isi nilai kosong (0 = tidak ambil pelajaran) dengan median
        num_cols = [c for c in df.columns if c not in
                    ["rumpun_ilmu", "program_studi", "tingkat_kesesuaian"]]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col][df[col] > 0].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 75)

        logger.info(f"Akademik: {df.shape[0]} siswa, {df.shape[1]} kolom")
        logger.info(f"Rumpun Ilmu: {df['rumpun_ilmu'].value_counts().to_dict()}")
        return df

    @property
    def akademik_feature_cols(self) -> List[str]:
        """Daftar kolom fitur akademik (bukan target)."""
        return [
            "mat_s4", "fis_s4", "kim_s4", "bio_s4", "bind_s4", "bing_s4", "info_s4",
            "mat_s5", "fis_s5", "kim_s5", "bio_s5", "bind_s5", "bing_s5", "info_s5",
        ]

    # ------------------------------------------------------------------
    # Dataset Talent: Gardner Multiple Intelligence
    # ------------------------------------------------------------------
    def load_talent(self) -> pd.DataFrame:
        """
        Muat Dataset_TalentN.xlsx (kecerdasan majemuk Gardner).

        Hanya ambil kolom kecerdasan numerik:
          Linguistic, Musical, Bodily, Logical-Mathematical,
          Spatial-Visualization, Interpersonal, Intrapersonal, Naturalist

        Returns:
            DataFrame dengan kolom kecerdasan + job_profession sebagai label
        """
        path = os.path.join(self.data_dir, "Dataset_TalentN.xlsx")
        df = pd.read_excel(path)

        rename = {
            "Linguistic":             "linguistik",
            "Musical":                "musikal",
            "Bodily":                 "kinestetik",
            "Logical - Mathematical": "logika_mat",
            "Spatial-Visualization":  "spasial",
            "Interpersonal":          "interpersonal",
            "Intrapersonal":          "intrapersonal",
            "Naturalist":             "naturalis",
            "Job profession":         "profesi",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Bersihkan profesi
        if "profesi" in df.columns:
            df["profesi"] = df["profesi"].str.strip().str.replace(r'\n', '', regex=True)

        # Hanya ambil kolom yang relevan
        talent_cols = ["linguistik", "musikal", "kinestetik", "logika_mat",
                       "spasial", "interpersonal", "intrapersonal", "naturalis"]
        available = [c for c in talent_cols if c in df.columns]
        if "profesi" in df.columns:
            available += ["profesi"]

        df = df[available].dropna(subset=[c for c in talent_cols if c in df.columns])

        for col in talent_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(10)

        logger.info(f"Talent: {df.shape[0]} profil, {len(available)-1} kecerdasan")
        return df

    @property
    def talent_feature_cols(self) -> List[str]:
        """Daftar kolom fitur kecerdasan Gardner."""
        return [
            "linguistik", "musikal", "kinestetik", "logika_mat",
            "spasial", "interpersonal", "intrapersonal", "naturalis",
        ]


# ------------------------------------------------------------------
# Mapping Gardner MI → bobot RIASEC
# (dipakai di predictor untuk enrichment, bukan training)
# ------------------------------------------------------------------
GARDNER_TO_RIASEC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "linguistik":     {"Social": 0.7, "Artistic": 0.6, "Enterprising": 0.4},
    "musikal":        {"Artistic": 0.9, "Investigative": 0.3},
    "kinestetik":     {"Realistic": 0.8, "Social": 0.3},
    "logika_mat":     {"Investigative": 0.9, "Conventional": 0.6, "Realistic": 0.4},
    "spasial":        {"Artistic": 0.7, "Realistic": 0.5, "Investigative": 0.4},
    "interpersonal":  {"Social": 0.9, "Enterprising": 0.7},
    "intrapersonal":  {"Investigative": 0.5, "Artistic": 0.5},
    "naturalis":      {"Realistic": 0.7, "Investigative": 0.6},
}


def compute_riasec_from_gardner(talent_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Hitung skor RIASEC dari profil kecerdasan majemuk Gardner.

    Args:
        talent_scores: {'linguistik': 12, 'musikal': 6, ...}
                       Nilai asli dari tes Gardner (range bervariasi)

    Returns:
        {'Realistic': 0.3, 'Investigative': 0.7, ...} (dinormalisasi 0-1)
    """
    riasec_raw = {t: 0.0 for t in RIASEC_TYPES}
    riasec_count = {t: 0 for t in RIASEC_TYPES}

    for mi_name, score in talent_scores.items():
        if mi_name not in GARDNER_TO_RIASEC_WEIGHTS:
            continue
        # Normalisasi skor Gardner ke 0-1 (asumsi max = 20 berdasarkan dataset)
        normalized = min(score / 20.0, 1.0)
        for riasec_type, weight in GARDNER_TO_RIASEC_WEIGHTS[mi_name].items():
            riasec_raw[riasec_type] += normalized * weight
            riasec_count[riasec_type] += 1

    # Normalisasi hasil akhir ke 0-1
    total = sum(riasec_raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in riasec_raw.items()}
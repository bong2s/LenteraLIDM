"""
=============================================================
MODUL: train.py
=============================================================
TUJUAN:
  Pipeline pelatihan lengkap – dari dataset mentah hingga
  model .pkl yang siap dipakai API.

ALUR LENGKAP:
  1. Muat 3 dataset Excel
  2. Ekstrak fitur gambar tulisan tangan (OpenCV)
  3. Gabungkan semua fitur
  4. Latih RIASECClassifier
  5. Latih MajorRecommender
  6. Simpan kedua model ke folder models/
  7. Tampilkan laporan performa

CARA JALANKAN:
  python -m src.training.train
  (dari dalam folder ml-handwriting/)
=============================================================
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Tambah root ke Python path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.data_loader import DatasetLoader
from src.preprocessing.image_processor import HandwritingFeatureExtractor
from src.models.riasec_classifier import RIASECClassifier
from src.models.major_recommender import MajorRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fungsi utama
# ------------------------------------------------------------------
def run_training(
    data_dir: str = "data/raw",
    image_dir: str = "data/raw/img",
    model_dir: str = "models",
    report_path: str = "models/training_report.json",
) -> dict:
    """
    Jalankan seluruh pipeline pelatihan.

    Args:
        data_dir   : folder berisi file .xlsx
        image_dir  : folder berisi file gambar tulisan (img_1.png dst)
        model_dir  : folder output untuk .pkl model
        report_path: path untuk menyimpan laporan JSON

    Returns:
        dict berisi semua metrik pelatihan
    """

    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PELATIHAN ML TULISAN TANGAN")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # LANGKAH 1: Muat Dataset
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Memuat dataset...")
    loader = DatasetLoader(data_dir=data_dir)
    df = loader.load_all()
    logger.info(f"Dataset: {df.shape[0]} siswa, {df.shape[1]} kolom")
    logger.info(f"Kolom: {list(df.columns)}")

    # ------------------------------------------------------------------
    # LANGKAH 2: Ekstrak Fitur Gambar
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Mengekstrak fitur gambar tulisan tangan...")
    extractor = HandwritingFeatureExtractor()
    image_features_list = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]  # misal "S001"
        # Konversi S001 → img_1.png, S002 → img_2.png, dst.
        num = int(sample_id.replace("S", ""))
        img_path = os.path.join(image_dir, f"img_{num}.png")

        features = extractor.extract(img_path)
        features["sample_id"] = sample_id
        image_features_list.append(features)

        if num % 10 == 0:
            logger.info(f"  Diproses: {num}/{len(df)} gambar")

    img_df = pd.DataFrame(image_features_list)
    logger.info(f"Fitur gambar: {img_df.shape[1]-1} fitur untuk {img_df.shape[0]} gambar")

    # ------------------------------------------------------------------
    # LANGKAH 3: Gabungkan Fitur
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Menggabungkan semua fitur...")

    # Gabungkan fitur gambar ke dataset utama
    df_full = df.merge(img_df, on="sample_id", how="left")

    # Pilih kolom fitur (hapus non-fitur)
    exclude_cols = [
        "sample_id", "dominant_riasec", "recommended_major",
        "dominant_riasec_x", "dominant_riasec_y",
        "recommended_major_x", "recommended_major_y",
        "style_category", "writing_type", "document_type",  # sudah di-encode di tulisan
        "color_usage",  # bisa jadi numerik/string, skip untuk aman
    ]

    feature_cols = [c for c in df_full.columns if c not in exclude_cols]
    logger.info(f"Total fitur untuk training: {len(feature_cols)}")
    logger.info(f"Fitur: {feature_cols}")

    X = df_full[feature_cols].copy()

    # Pastikan semua kolom numerik
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)

    y_riasec = df_full["dominant_riasec"]
    y_major = df_full["recommended_major"]

    logger.info(f"Shape X: {X.shape}")
    logger.info(f"Distribusi RIASEC: {y_riasec.value_counts().to_dict()}")
    logger.info(f"Distribusi Jurusan: {y_major.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # LANGKAH 4: Latih Model RIASEC
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Melatih RIASEC Classifier...")
    riasec_clf = RIASECClassifier(n_estimators=200, random_state=42)
    riasec_metrics = riasec_clf.train(X, y_riasec)
    logger.info(f"RIASEC Accuracy: {riasec_metrics['cv_accuracy_mean']:.3f}")

    # Tampilkan fitur terpenting
    top_features = riasec_clf.get_feature_importance(top_n=5)
    logger.info("Top 5 fitur paling berpengaruh untuk RIASEC:")
    for f in top_features:
        logger.info(f"  {f['fitur']:30s}: {f['importance']:.4f}")

    # ------------------------------------------------------------------
    # LANGKAH 5: Latih Major Recommender
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Melatih Major Recommender...")
    major_rec = MajorRecommender(random_state=42)
    major_metrics = major_rec.train(X, y_major)
    logger.info(f"Major Rec Accuracy: {major_metrics['cv_accuracy_mean']:.3f}")

    # ------------------------------------------------------------------
    # SIMPAN MODEL
    # ------------------------------------------------------------------
    logger.info("\nMenyimpan model...")
    os.makedirs(model_dir, exist_ok=True)

    riasec_path = os.path.join(model_dir, "riasec_model.pkl")
    major_path = os.path.join(model_dir, "major_model.pkl")

    riasec_clf.save(riasec_path)
    major_rec.save(major_path)

    # Simpan juga nama kolom fitur agar API bisa align input
    feature_meta = {
        "feature_columns": feature_cols,
        "riasec_classes": riasec_metrics["classes"],
        "major_classes": major_metrics["jurusan_tersedia"],
    }
    meta_path = os.path.join(model_dir, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)

    # ------------------------------------------------------------------
    # LAPORAN AKHIR
    # ------------------------------------------------------------------
    report = {
        "status": "SUCCESS",
        "dataset": {
            "n_samples": int(df.shape[0]),
            "n_features": len(feature_cols),
        },
        "riasec_model": riasec_metrics,
        "major_model": major_metrics,
        "saved_files": {
            "riasec": riasec_path,
            "major": major_path,
            "meta": meta_path,
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("✅ PELATIHAN SELESAI!")
    logger.info(f"   RIASEC CV Accuracy : {riasec_metrics['cv_accuracy_mean']:.1%}")
    logger.info(f"   Major  CV Accuracy : {major_metrics['cv_accuracy_mean']:.1%}")
    logger.info(f"   Model disimpan di   : {model_dir}/")
    logger.info(f"   Laporan             : {report_path}")
    logger.info("=" * 60)

    return report


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Jalankan dari folder ml-handwriting/:
    # python -m src.training.train
    report = run_training()
    print("\nLaporan lengkap:")
    print(json.dumps(report, indent=2))
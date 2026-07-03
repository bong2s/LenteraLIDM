"""
=============================================================
MODUL: train.py  (Dataset Real v3 — RIASEC Direct)
=============================================================
Pipeline training lengkap:
  1. Muat CSV fitur tulisan (Dataset_TulisanN.csv) → RiasecClassifier
  2. Muat dataset akademik (140 siswa) → RumpunClassifier
  3. Bangun MajorRecommender dari dataset akademik
  4. Simpan semua model ke models/

Perbedaan dari v2:
  - Langkah 1 membaca CSV pra-ekstraksi, BUKAN folder gambar
  - Tidak ada BigFiveClassifier, tidak ada Dataset_TalentN
  - Nilai 0 di dataset akademik TETAP 0 (tidak diganti median)
  - Output: riasec_model.pkl (bukan bigfive_model.pkl)

Cara jalankan:
  python train_and_save.py
=============================================================
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.data_loader import DatasetLoader, RUMPUN_ILMU
from src.models.riasec_classifier import RiasecClassifier, RumpunClassifier
from src.models.major_recommender import MajorRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_training(
    data_dir:    str = "data_new",
    model_dir:   str = "models",
    report_path: str = "models/training_report.json",
) -> dict:
    """
    Jalankan seluruh pipeline training dataset real v3.

    Args:
        data_dir   : folder berisi dataset (xlsx + CSV)
        model_dir  : folder output .pkl model
        report_path: path laporan JSON
    """
    logger.info("=" * 60)
    logger.info("TRAINING DATASET REAL v3 — RIASEC DIRECT")
    logger.info("=" * 60)

    os.makedirs(model_dir, exist_ok=True)
    loader = DatasetLoader(data_dir=data_dir)

    # ------------------------------------------------------------------
    # LANGKAH 1: CSV fitur tulisan → RiasecClassifier
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Muat fitur tulisan dari CSV...")

    csv_path = os.path.join(data_dir, "Dataset_TulisanN.csv")
    X_tulisan, y_riasec = loader.load_tulisan_csv(csv_path=csv_path, split="train")

    logger.info(f"  Total sampel: {len(X_tulisan)}")
    logger.info(f"  Distribusi RIASEC: {y_riasec.value_counts().to_dict()}")

    logger.info("\n  Melatih RIASEC Classifier...")
    riasec_clf = RiasecClassifier()
    riasec_metrics = riasec_clf.train(X_tulisan, y_riasec)
    logger.info(f"  RIASEC CV Accuracy: {riasec_metrics['cv_accuracy_mean']:.1%}")
    logger.info("  Top fitur RIASEC:")
    for f in riasec_clf.get_feature_importance(top_n=5):
        logger.info(f"    {f['fitur']:20s}: {f['importance']:.4f}")

    # ------------------------------------------------------------------
    # LANGKAH 2: Dataset Akademik → Rumpun Classifier
    # ------------------------------------------------------------------
    logger.info("\n[2/4] Muat dataset akademik...")
    df_ak = loader.load_akademik()

    feat_cols = loader.akademik_feature_cols
    X_ak    = df_ak[feat_cols].copy()
    y_rumpun = df_ak["rumpun_ilmu"]

    logger.info(f"  Siswa: {len(df_ak)}, Fitur: {len(feat_cols)}")
    logger.info(f"  Rumpun Ilmu: {y_rumpun.value_counts().to_dict()}")

    # Nilai 0 = tidak ambil pelajaran → TETAP 0, tidak diganti median
    logger.info("\n  Melatih Rumpun Ilmu Classifier...")
    rumpun_clf = RumpunClassifier()
    rumpun_metrics = rumpun_clf.train(X_ak, y_rumpun)
    logger.info(f"  Rumpun CV Accuracy: {rumpun_metrics['cv_accuracy_mean']:.1%}")

    # ------------------------------------------------------------------
    # LANGKAH 3: Major Recommender
    # ------------------------------------------------------------------
    logger.info("\n[3/4] Membangun Major Recommender...")
    major_rec = MajorRecommender()
    major_rec.build_from_dataset(df_ak)
    logger.info(f"  Program Studi tersedia: {major_rec.n_programs}")

    # ------------------------------------------------------------------
    # LANGKAH 4: Simpan Semua Model
    # ------------------------------------------------------------------
    logger.info("\n[4/4] Menyimpan model...")

    riasec_path = os.path.join(model_dir, "riasec_model.pkl")
    rumpun_path = os.path.join(model_dir, "rumpun_model.pkl")
    major_path  = os.path.join(model_dir, "major_model.pkl")

    riasec_clf.save(riasec_path)
    rumpun_clf.save(rumpun_path)
    major_rec.save(major_path)

    # Metadata
    prodi_rumpun = (
        df_ak.groupby("program_studi")["rumpun_ilmu"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    prodi_score = (
        df_ak.groupby("program_studi")["tingkat_kesesuaian"]
        .mean()
        .round(2)
        .to_dict()
    )

    meta = {
        "versi": "3.0-riasec-direct",
        "riasec_model": {
            "feature_cols":  list(X_tulisan.columns),
            "classes":       riasec_metrics["classes"],
            "cv_accuracy":   riasec_metrics["cv_accuracy_mean"],
        },
        "rumpun_model": {
            "feature_cols": feat_cols,
            "classes":      rumpun_metrics["classes"],
            "cv_accuracy":  rumpun_metrics["cv_accuracy_mean"],
        },
        "major_model": {
            "programs":        sorted(df_ak["program_studi"].unique().tolist()),
            "rumpun_mapping":  prodi_rumpun,
            "avg_score":       prodi_score,
        },
        "dataset_info": {
            "n_tulisan":  len(X_tulisan),
            "n_students": len(df_ak),
            "riasec_distribution": y_riasec.value_counts().to_dict(),
            "rumpun_distribution": y_rumpun.value_counts().to_dict(),
        },
    }

    with open(os.path.join(model_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    report = {
        "status":       "SUCCESS",
        "riasec_model": riasec_metrics,
        "rumpun_model": rumpun_metrics,
        "major_programs": major_rec.n_programs,
        "saved": {
            "riasec": riasec_path,
            "rumpun": rumpun_path,
            "major":  major_path,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING SELESAI!")
    logger.info(f"   RIASEC CV Accuracy : {riasec_metrics['cv_accuracy_mean']:.1%}")
    logger.info(f"   Rumpun CV Accuracy : {rumpun_metrics['cv_accuracy_mean']:.1%}")
    logger.info(f"   Program Studi      : {major_rec.n_programs} jurusan")
    logger.info(f"   Model disimpan di  : {model_dir}/")
    logger.info("=" * 60)
    logger.info("\nLangkah selanjutnya:")
    logger.info("  uvicorn api.main:app --reload --port 8000")

    return report


if __name__ == "__main__":
    report = run_training()
    print(json.dumps(report, indent=2, ensure_ascii=False))
"""
=============================================================
MODUL: train.py  (Dataset Real v2)
=============================================================
Pipeline training lengkap:
  1. Muat gambar per folder Big Five (221 gambar)
  2. Ekstrak 10 fitur OpenCV per gambar
  3. Latih BigFiveClassifier (gambar → Big Five)
  4. Muat dataset akademik (140 siswa)
  5. Latih RumpunClassifier (nilai akademik → Rumpun Ilmu)
  6. Simpan semua model ke models/
  7. Simpan feature_meta.json

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
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.data_loader import (
    DatasetLoader, BIG_FIVE_TYPES, RUMPUN_ILMU,
)
from src.preprocessing.image_processor import HandwritingFeatureExtractor
from src.models.riasec_classifier import BigFiveClassifier, RumpunClassifier
from src.models.major_recommender import MajorRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_training(
    data_dir:   str = "data_new",
    model_dir:  str = "models",
    report_path: str = "models/training_report.json",
) -> dict:
    """
    Jalankan seluruh pipeline training dataset real v2.

    Args:
        data_dir  : folder berisi dataset baru (xlsx + folder gambar)
        model_dir : folder output .pkl model
        report_path: path laporan JSON
    """
    logger.info("=" * 60)
    logger.info("TRAINING DATASET REAL v2 — ML TULISAN TANGAN")
    logger.info("=" * 60)

    os.makedirs(model_dir, exist_ok=True)
    loader    = DatasetLoader(data_dir=data_dir)
    extractor = HandwritingFeatureExtractor()

    # ------------------------------------------------------------------
    # LANGKAH 1: Dataset Tulisan → BigFive Classifier
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Muat gambar tulisan tangan per kategori Big Five...")

    img_base = os.path.join(data_dir, "Dataset_TulisanN")
    paths, labels = loader.load_tulisan_image_paths(img_base)

    logger.info(f"  Total gambar  : {len(paths)}")
    for cat in BIG_FIVE_TYPES:
        n = labels.count(cat)
        logger.info(f"  {cat}: {n} gambar")

    logger.info("\n  Mengekstrak fitur gambar...")
    X_img_list, y_bigfive = [], []
    errors = 0
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            features = extractor.extract(path)
            X_img_list.append(features)
            y_bigfive.append(label)
        except Exception as e:
            logger.warning(f"  Gambar {os.path.basename(path)} error: {e}")
            errors += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Diproses: {i+1}/{len(paths)} gambar")

    logger.info(f"  Berhasil: {len(X_img_list)} gambar, Error: {errors}")

    X_img = pd.DataFrame(X_img_list)
    y_bigfive_series = pd.Series(y_bigfive)

    # Latih BigFive Classifier
    logger.info("\n  Melatih BigFive Classifier...")
    bigfive_clf = BigFiveClassifier()
    bigfive_metrics = bigfive_clf.train(X_img, y_bigfive_series)
    logger.info(f"  BigFive CV Accuracy: {bigfive_metrics['cv_accuracy_mean']:.1%}")
    logger.info("  Top fitur Big Five:")
    for f in bigfive_clf.get_feature_importance(top_n=5):
        logger.info(f"    {f['fitur']:20s}: {f['importance']:.4f}")

    # ------------------------------------------------------------------
    # LANGKAH 2: Dataset Akademik → Rumpun Classifier
    # ------------------------------------------------------------------
    logger.info("\n[2/4] Muat dataset akademik...")
    df_ak = loader.load_akademik()

    feat_cols = loader.akademik_feature_cols
    X_ak = df_ak[feat_cols].copy()
    y_rumpun = df_ak["rumpun_ilmu"]
    y_prodi   = df_ak["program_studi"]
    y_score   = df_ak["tingkat_kesesuaian"]

    logger.info(f"  Siswa: {len(df_ak)}, Fitur: {len(feat_cols)}")
    logger.info(f"  Rumpun Ilmu: {y_rumpun.value_counts().to_dict()}")

    # Isi nilai 0 (tidak ambil pelajaran) dengan median non-zero
    for col in feat_cols:
        med = X_ak[col][X_ak[col] > 0].median()
        X_ak[col] = X_ak[col].replace(0, med if not pd.isna(med) else 75)
        X_ak[col] = X_ak[col].fillna(75)

    # Latih Rumpun Classifier
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

    bigfive_path = os.path.join(model_dir, "bigfive_model.pkl")
    rumpun_path  = os.path.join(model_dir, "rumpun_model.pkl")
    major_path   = os.path.join(model_dir, "major_model.pkl")

    bigfive_clf.save(bigfive_path)
    rumpun_clf.save(rumpun_path)
    major_rec.save(major_path)

    # Simpan metadata
    # Daftar Program Studi dengan Rumpun Ilmu-nya
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
        "versi": "2.0-dataset-real",
        "bigfive_model": {
            "feature_cols": list(X_img.columns),
            "classes": bigfive_metrics["classes"],
            "cv_accuracy": bigfive_metrics["cv_accuracy_mean"],
        },
        "rumpun_model": {
            "feature_cols": feat_cols,
            "classes": rumpun_metrics["classes"],
            "cv_accuracy": rumpun_metrics["cv_accuracy_mean"],
        },
        "major_model": {
            "programs": sorted(df_ak["program_studi"].unique().tolist()),
            "rumpun_mapping": prodi_rumpun,
            "avg_score": prodi_score,
        },
        "dataset_info": {
            "n_images": len(X_img_list),
            "n_students": len(df_ak),
            "rumpun_distribution": y_rumpun.value_counts().to_dict(),
        },
    }

    with open(os.path.join(model_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Laporan training
    report = {
        "status": "SUCCESS",
        "bigfive_model":  bigfive_metrics,
        "rumpun_model":   rumpun_metrics,
        "major_programs": major_rec.n_programs,
        "saved": {
            "bigfive": bigfive_path,
            "rumpun":  rumpun_path,
            "major":   major_path,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING SELESAI!")
    logger.info(f"   BigFive CV Accuracy : {bigfive_metrics['cv_accuracy_mean']:.1%}")
    logger.info(f"   Rumpun  CV Accuracy : {rumpun_metrics['cv_accuracy_mean']:.1%}")
    logger.info(f"   Program Studi       : {major_rec.n_programs} jurusan")
    logger.info(f"   Model disimpan di   : {model_dir}/")
    logger.info("=" * 60)
    logger.info("\nLangkah selanjutnya:")
    logger.info("  uvicorn api.main:app --reload --port 8000")

    return report


if __name__ == "__main__":
    report = run_training()
    print(json.dumps(report, indent=2, ensure_ascii=False))
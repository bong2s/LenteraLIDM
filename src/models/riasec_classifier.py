"""
=============================================================
MODUL: riasec_classifier.py
=============================================================
TUJUAN:
  Melatih dan menyimpan model klasifikasi RIASEC.
  RIASEC adalah kerangka psikologi karier Holland yang membagi
  kepribadian ke 6 tipe: Realistic, Investigative, Artistic,
  Social, Enterprising, Conventional.

MODEL YANG DIGUNAKAN:
  RandomForestClassifier — karena:
  1. Bekerja baik dengan dataset kecil (40 sampel)
  2. Menangani fitur numerik dan kategorik campuran
  3. Tahan terhadap overfitting berkat ensemble
  4. Memberikan feature importance (berguna untuk interpretasi)

PIPELINE:
  Input fitur → StandardScaler (normalisasi) → RandomForest → Prediksi RIASEC
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class RIASECClassifier:
    """
    Classifier untuk memprediksi tipe RIASEC dominan siswa.

    Input:  fitur gabungan (nilai akademik + bakat + tulisan tangan)
    Output: salah satu dari 6 tipe RIASEC

    Cara pakai:
        clf = RIASECClassifier()
        clf.train(X_train, y_train)
        prediksi = clf.predict(X_test)
        clf.save("models/riasec_model.pkl")
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        """
        Args:
            n_estimators: jumlah pohon di RandomForest (lebih banyak = lebih akurat tapi lambat)
            random_state : seed untuk reproducibility
        """
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []

        # Pipeline: normalisasi → model
        # StandardScaler: mengubah semua fitur ke skala 0-1
        # RandomForest: ensemble dari banyak decision tree
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=8,           # hindari overfitting (dataset kecil)
                min_samples_split=2,   # min sampel untuk split node
                min_samples_leaf=1,    # min sampel di daun
                class_weight="balanced",  # tangani imbalanced class
                random_state=random_state,
                n_jobs=-1,             # pakai semua core CPU
            )),
        ])

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Latih model RIASEC.

        Args:
            X: DataFrame fitur (tiap baris = 1 siswa)
            y: Series label RIASEC (misalnya "Conventional", "Investigative")

        Returns:
            dict berisi metrik akurasi dan laporan per kelas
        """
        self.feature_names = list(X.columns)

        # Encode label string → angka (Conventional=0, Enterprising=1, dst)
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Training RIASEC classifier: {X.shape[0]} sampel, {X.shape[1]} fitur")
        logger.info(f"Distribusi kelas: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Cross validation untuk estimasi akurasi yang lebih jujur
        # (penting karena dataset kecil = mudah overfitting)
        cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_encoded))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, X, y_encoded, cv=cv, scoring="accuracy")

        logger.info(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Latih dengan semua data
        self.pipeline.fit(X, y_encoded)

        # Prediksi ulang untuk laporan training
        y_pred = self.pipeline.predict(X)
        train_acc = accuracy_score(y_encoded, y_pred)

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "classes": list(self.label_encoder.classes_),
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

        logger.info(f"Training accuracy: {train_acc:.3f}")
        return metrics

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> str:
        """
        Prediksi tipe RIASEC dominan untuk 1 atau lebih siswa.

        Args:
            X: DataFrame fitur (baris = jumlah siswa)

        Returns:
            string nama tipe RIASEC (misalnya "Investigative")
        """
        X_aligned = self._align_features(X)
        pred_encoded = self.pipeline.predict(X_aligned)
        return self.label_encoder.inverse_transform(pred_encoded)[0]

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Prediksi probabilitas untuk setiap tipe RIASEC.

        Returns:
            dict: {"Conventional": 0.4, "Investigative": 0.3, ...}
        """
        X_aligned = self._align_features(X)
        proba = self.pipeline.predict_proba(X_aligned)[0]
        classes = self.label_encoder.classes_

        result = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    def get_feature_importance(self, top_n: int = 10) -> List[Dict]:
        """
        Tampilkan fitur yang paling berpengaruh pada prediksi RIASEC.
        Berguna untuk interpretasi model kepada siswa.

        Returns:
            list of dict: [{"fitur": "matematika", "importance": 0.12}, ...]
        """
        rf = self.pipeline.named_steps["classifier"]
        importances = rf.feature_importances_

        fi = sorted(
            zip(self.feature_names, importances),
            key=lambda x: -x[1]
        )[:top_n]

        return [{"fitur": name, "importance": round(imp, 4)} for name, imp in fi]

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Simpan model ke file .pkl"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Model RIASEC disimpan: {path}")

    def load(self, path: str):
        """Muat model dari file .pkl"""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        logger.info(f"Model RIASEC dimuat: {path}")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pastikan kolom input sesuai dengan fitur saat training."""
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.feature_names]
"""
=============================================================
MODUL: riasec_classifier.py  (Dataset Real v2)
=============================================================
Dua classifier utama:

  1. BigFiveClassifier
     Input : 10 fitur gambar tulisan tangan
     Output: label Big Five (Openness / Conscientiousness /
             Extraversion / Agreeableness / Neuroticism)
     Dataset: 221 gambar berlabel dari folder Big Five

  2. RumpunClassifier
     Input : 14 nilai akademik (2 semester)
     Output: Rumpun Ilmu (STEM / Sosial Humaniora /
             Bisnis Manajemen / Pendidikan / Seni Kreatif)
     Dataset: 140 data akademik

Keduanya menggunakan RandomForest / GradientBoosting
dengan class_weight untuk menangani ketidakseimbangan kelas.
=============================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


def _balanced_weights(y: pd.Series) -> Dict[str, float]:
    counts = y.value_counts()
    n_total = len(y)
    n_classes = len(counts)
    return {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}


# ------------------------------------------------------------------
# 1. BigFiveClassifier — Gambar → Big Five Personality
# ------------------------------------------------------------------

class BigFiveClassifier:
    """
    Memprediksi Big Five personality dari 10 fitur tulisan tangan.

    Input : dict / DataFrame dengan kolom:
            letter_size, slant, pressure, spacing, readability,
            neatness, connectivity, ornament, baseline, density

    Output: label Big Five string
            {'Openness', 'Conscientiousness', 'Extraversion',
             'Agreeableness', 'Neuroticism'}
    """

    FEATURES = [
        "letter_size", "slant", "pressure", "spacing", "readability",
        "neatness", "connectivity", "ornament", "baseline", "density",
    ]

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []
        self._feature_importances: Dict[str, float] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.classes_ = sorted(y.unique().tolist())
        logger.info(f"BigFive training: {len(X)} sampel, {len(self.classes_)} kelas")
        logger.info(f"  Distribusi: {y.value_counts().to_dict()}")

        X_feat = self._ensure_features(X)
        class_weights = _balanced_weights(y)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.pipeline, X_feat, y,
            cv=cv, scoring="accuracy", return_train_score=True,
        )

        self.pipeline.fit(X_feat, y)

        rf = self.pipeline.named_steps["clf"]
        self._feature_importances = dict(zip(self.FEATURES, rf.feature_importances_.tolist()))

        metrics = {
            "classes":          self.classes_,
            "n_samples":        len(X),
            "cv_accuracy_mean": float(np.mean(cv_results["test_score"])),
            "cv_accuracy_std":  float(np.std(cv_results["test_score"])),
            "train_accuracy":   float(np.mean(cv_results["train_score"])),
        }
        logger.info(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
        return metrics

    def predict(self, features: Dict[str, float]) -> str:
        if self.pipeline is None:
            raise RuntimeError("Model belum dilatih. Panggil load() dulu.")
        X = self._dict_to_df(features)
        return str(self.pipeline.predict(X)[0])

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Model belum dilatih.")
        X = self._dict_to_df(features)
        proba = self.pipeline.predict_proba(X)[0]
        return {cls: float(round(p, 4)) for cls, p in
                zip(self.pipeline.classes_, proba)}

    def get_feature_importance(self, top_n: int = 10) -> List[Dict]:
        if not self._feature_importances:
            return []
        sorted_fi = sorted(self._feature_importances.items(), key=lambda x: -x[1])
        return [{"fitur": k, "importance": round(v, 4)} for k, v in sorted_fi[:top_n]]

    def save(self, path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pipeline":            self.pipeline,
            "classes_":            self.classes_,
            "feature_importances": self._feature_importances,
        }, path, compress=3)
        logger.info(f"BigFive model disimpan: {path}")

    @classmethod
    def load(cls, path: str) -> "BigFiveClassifier":
        obj = cls()
        data = joblib.load(path)
        obj.pipeline = data["pipeline"]
        obj.classes_ = data["classes_"]
        obj._feature_importances = data.get("feature_importances", {})
        logger.info(f"BigFive model dimuat: {path}")
        return obj

    def _ensure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.FEATURES:
            if col not in X.columns:
                X = X.copy()
                X[col] = 5.0
        return X[self.FEATURES]

    def _dict_to_df(self, features: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{f: features.get(f, 5.0) for f in self.FEATURES}])


# ------------------------------------------------------------------
# 2. RumpunClassifier — Akademik → Rumpun Ilmu
# ------------------------------------------------------------------

class RumpunClassifier:
    """
    Memprediksi Rumpun Ilmu dari nilai akademik 2 semester.

    Input : dict dengan 14 key (mat_s4, fis_s4, ..., info_s5)
    Output: 'STEM' | 'Sosial Humaniora' | 'Bisnis Manajemen' |
            'Pendidikan' | 'Seni Kreatif'
    """

    FEATURES = [
        "mat_s4", "fis_s4", "kim_s4", "bio_s4", "bind_s4", "bing_s4", "info_s4",
        "mat_s5", "fis_s5", "kim_s5", "bio_s5", "bind_s5", "bing_s5", "info_s5",
    ]

    FEATURE_LABELS = {
        "mat_s4": "Matematika Smt 4",  "fis_s4": "Fisika Smt 4",
        "kim_s4": "Kimia Smt 4",       "bio_s4": "Biologi Smt 4",
        "bind_s4": "B. Indonesia Smt 4","bing_s4": "B. Inggris Smt 4",
        "info_s4": "Informatika Smt 4", "mat_s5": "Matematika Smt 5",
        "fis_s5": "Fisika Smt 5",       "kim_s5": "Kimia Smt 5",
        "bio_s5": "Biologi Smt 5",      "bind_s5": "B. Indonesia Smt 5",
        "bing_s5": "B. Inggris Smt 5",  "info_s5": "Informatika Smt 5",
    }

    # Mata pelajaran kunci per rumpun (heuristik fallback)
    RUMPUN_KEY_SUBJECTS = {
        "STEM":             ["mat_s4", "mat_s5", "fis_s4", "fis_s5",
                             "kim_s4", "kim_s5", "info_s4", "info_s5"],
        "Sosial Humaniora": ["bind_s4", "bind_s5", "bing_s4", "bing_s5"],
        "Bisnis Manajemen": ["mat_s4", "mat_s5", "bing_s4", "bing_s5"],
        "Pendidikan":       ["bind_s4", "bind_s5", "bio_s4", "bio_s5"],
        "Seni Kreatif":     ["bing_s4", "bing_s5", "bind_s4"],
    }

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []
        self._feature_importances: Dict[str, float] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.classes_ = sorted(y.unique().tolist())
        logger.info(f"Rumpun training: {len(X)} siswa, {len(self.classes_)} rumpun")

        X_feat = self._ensure_features(X)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.pipeline, X_feat, y,
            cv=cv, scoring="accuracy", return_train_score=True,
        )

        self.pipeline.fit(X_feat, y)

        gb = self.pipeline.named_steps["clf"]
        self._feature_importances = dict(zip(self.FEATURES, gb.feature_importances_.tolist()))

        metrics = {
            "classes":          self.classes_,
            "n_samples":        len(X),
            "cv_accuracy_mean": float(np.mean(cv_results["test_score"])),
            "cv_accuracy_std":  float(np.std(cv_results["test_score"])),
            "train_accuracy":   float(np.mean(cv_results["train_score"])),
        }
        logger.info(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
        return metrics

    def predict(self, academic_scores: Dict[str, float]) -> str:
        if self.pipeline is None:
            return self.heuristic_predict(academic_scores)
        X = self._dict_to_df(academic_scores)
        return str(self.pipeline.predict(X)[0])

    def predict_proba(self, academic_scores: Dict[str, float]) -> Dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Model belum dilatih.")
        X = self._dict_to_df(academic_scores)
        proba = self.pipeline.predict_proba(X)[0]
        return {cls: float(round(p, 4)) for cls, p in
                zip(self.pipeline.classes_, proba)}

    def heuristic_predict(self, scores: Dict[str, float]) -> str:
        rumpun_scores = {
            r: float(np.mean([scores.get(s, 75) for s in subjs]))
            for r, subjs in self.RUMPUN_KEY_SUBJECTS.items()
        }
        return max(rumpun_scores, key=rumpun_scores.get)

    def get_top_subjects(self, academic_scores: Dict[str, float], top_n: int = 3) -> List[Dict]:
        sorted_subj = sorted(academic_scores.items(), key=lambda x: -x[1])
        return [
            {"mata_pelajaran": self.FEATURE_LABELS.get(k, k), "nilai": round(v, 1)}
            for k, v in sorted_subj[:top_n]
        ]

    def save(self, path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pipeline":            self.pipeline,
            "classes_":            self.classes_,
            "feature_importances": self._feature_importances,
        }, path, compress=3)
        logger.info(f"Rumpun model disimpan: {path}")

    @classmethod
    def load(cls, path: str) -> "RumpunClassifier":
        obj = cls()
        data = joblib.load(path)
        obj.pipeline = data["pipeline"]
        obj.classes_ = data["classes_"]
        obj._feature_importances = data.get("feature_importances", {})
        logger.info(f"Rumpun model dimuat: {path}")
        return obj

    def _ensure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.FEATURES:
            if col not in X.columns:
                X = X.copy()
                X[col] = 75.0
        return X[self.FEATURES]

    def _dict_to_df(self, scores: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{f: scores.get(f, 75.0) for f in self.FEATURES}])
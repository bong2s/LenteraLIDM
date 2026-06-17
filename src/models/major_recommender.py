"""
=============================================================
MODUL: major_recommender.py
=============================================================
TUJUAN:
  Merekomendasikan TOP-3 jurusan kuliah berdasarkan gabungan:
  - Tipe RIASEC (karakter siswa)
  - Probabilitas RIASEC dari model
  - Nilai akademik
  - Skor bakat

PENDEKATAN:
  Multi-label → ambil probabilitas tiap jurusan → Top 3

KENAPA TIDAK HANYA 1 JURUSAN?
  Karena kepribadian manusia kompleks. Seorang siswa dengan
  RIASEC "Investigative" tapi nilai Seni tinggi perlu
  alternatif seperti Arsitektur selain Informatika.
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Mapping RIASEC → Jurusan (rule-based fallback)
# ------------------------------------------------------------------
RIASEC_TO_MAJORS: Dict[str, List[Dict]] = {
    "Realistic": [
        {"jurusan": "Teknik Sipil",      "alasan": "Kamu suka pekerjaan fisik dan membangun sesuatu yang nyata"},
        {"jurusan": "Teknik Mesin",      "alasan": "Kemampuan teknismu cocok dengan rekayasa mekanis"},
        {"jurusan": "Teknik Elektro",    "alasan": "Ketelitian dan kepraktisanmu sesuai dengan teknik kelistrikan"},
        {"jurusan": "Arsitektur",        "alasan": "Kamu bisa menggabungkan keahlian teknis dengan estetika"},
        {"jurusan": "Teknik Industri",   "alasan": "Kamu ahli mengoptimalkan proses kerja secara praktis"},
    ],
    "Investigative": [
        {"jurusan": "Informatika",       "alasan": "Kemampuan analitismu sangat cocok untuk pemrograman dan AI"},
        {"jurusan": "Kedokteran",        "alasan": "Rasa ingin tahumu yang tinggi sempurna untuk ilmu medis"},
        {"jurusan": "Farmasi",           "alasan": "Kamu teliti dan suka menyelidiki – ideal untuk kimia farmasi"},
        {"jurusan": "Biologi",           "alasan": "Penasaranmu terhadap alam cocok dengan penelitian biologi"},
        {"jurusan": "Statistik",         "alasan": "Kamu suka pola dan data – statistik adalah duniamu"},
    ],
    "Artistic": [
        {"jurusan": "DKV",               "alasan": "Kreativitas visualmu sangat cocok untuk Desain Komunikasi Visual"},
        {"jurusan": "Arsitektur",        "alasan": "Kamu bisa menuangkan kreativitas dalam desain bangunan"},
        {"jurusan": "Ilmu Komunikasi",   "alasan": "Kemampuan ekspresif dan kreatifmu ideal untuk komunikasi media"},
        {"jurusan": "Sastra Inggris",    "alasan": "Kamu punya kepekaan bahasa dan estetika yang tinggi"},
        {"jurusan": "Seni Rupa",         "alasan": "Kamu lahir untuk mengekspresikan diri melalui karya seni"},
    ],
    "Social": [
        {"jurusan": "Psikologi",         "alasan": "Empatimu yang tinggi membuat kamu cocok memahami perilaku manusia"},
        {"jurusan": "Pendidikan",        "alasan": "Jiwa mengajar dan membantumu sangat tepat untuk dunia pendidikan"},
        {"jurusan": "Ilmu Komunikasi",   "alasan": "Kamu pandai berinteraksi – komunikasi adalah kekuatanmu"},
        {"jurusan": "Kesehatan Masyarakat", "alasan": "Kepedulianmu pada orang cocok untuk bidang kesehatan masyarakat"},
        {"jurusan": "Sosiologi",         "alasan": "Kamu suka memahami pola sosial dan perilaku kelompok"},
    ],
    "Enterprising": [
        {"jurusan": "Manajemen",         "alasan": "Jiwa kepemimpinanmu cocok untuk mengelola organisasi"},
        {"jurusan": "Hukum",             "alasan": "Kemampuan persuasi dan argumenmu ideal di bidang hukum"},
        {"jurusan": "Bisnis Internasional", "alasan": "Ambisimu dan visi globalmu cocok untuk bisnis internasional"},
        {"jurusan": "Kewirausahaan",     "alasan": "Kamu lahir untuk membangun bisnis dan menciptakan peluang"},
        {"jurusan": "Ilmu Politik",      "alasan": "Kemampuan leadership dan persuasimu cocok di dunia politik"},
    ],
    "Conventional": [
        {"jurusan": "Akuntansi",         "alasan": "Ketelitian dan kerapianmu sangat cocok untuk akuntansi"},
        {"jurusan": "Sistem Informasi",  "alasan": "Kamu menyukai sistem yang teratur – SI adalah pilihan tepat"},
        {"jurusan": "Administrasi Bisnis", "alasan": "Kemampuan organisasimu ideal untuk mengelola bisnis"},
        {"jurusan": "Matematika",        "alasan": "Kecermatan dan logikamu sangat kuat untuk matematika"},
        {"jurusan": "Perpustakaan & Informasi", "alasan": "Kamu suka mengelola dan mengorganisasi informasi"},
    ],
}


class MajorRecommender:
    """
    Merekomendasikan TOP-3 jurusan berdasarkan kombinasi:
    - Prediksi ML (model terlatih)
    - Rule-based RIASEC mapping
    - Skor akademik sebagai bobot penyesuaian

    Cara pakai:
        rec = MajorRecommender()
        rec.train(X_train, y_major_train)
        top3 = rec.recommend_top3(X_input, riasec_proba)
    """

    def __init__(self, random_state: int = 42):
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Latih model rekomendasi jurusan.

        Args:
            X: DataFrame fitur
            y: Series label jurusan (misal "Informatika", "Akuntansi")

        Returns:
            dict metrik pelatihan
        """
        self.feature_names = list(X.columns)
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Training Major Recommender: {X.shape}, {len(np.unique(y_encoded))} jurusan")

        self.pipeline.fit(X, y_encoded)
        y_pred = self.pipeline.predict(X)

        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_encoded, y_pred)

        # Cross-validation hanya bisa jika setiap kelas punya ≥2 sampel
        min_class_count = int(np.bincount(y_encoded).min())
        n_splits = min(5, min_class_count)

        cv_mean, cv_std = 0.0, 0.0
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.pipeline, X, y_encoded, cv=cv, scoring="accuracy")
            cv_mean = round(cv_scores.mean(), 4)
            cv_std = round(cv_scores.std(), 4)
            logger.info(f"CV Major ({n_splits}-fold): {cv_mean:.3f} ± {cv_std:.3f}")
        else:
            logger.warning(
                f"Dataset jurusan terlalu sedikit per kelas (min={min_class_count}) "
                f"untuk cross-validation. Menggunakan train accuracy saja."
            )
            cv_mean = round(train_acc, 4)
            cv_std = 0.0

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "jurusan_tersedia": list(self.label_encoder.classes_),
            "n_jurusan": len(self.label_encoder.classes_),
        }

        logger.info(f"Major Rec accuracy: {train_acc:.3f}, CV: {cv_mean:.3f}")
        return metrics

    # ------------------------------------------------------------------
    # RECOMMEND TOP-3
    # ------------------------------------------------------------------
    def recommend_top3(
        self,
        X: pd.DataFrame,
        riasec_type: str,
        riasec_proba: Dict[str, float],
        akademik_scores: Dict[str, float] = None,
    ) -> List[Dict]:
        """
        Menghasilkan 3 rekomendasi jurusan dengan alasan.

        Strategi hybrid:
        1. ML model → probabilitas tiap jurusan (jika model tersedia)
        2. Rule-based RIASEC → bobot tambahan untuk jurusan sesuai tipe
        3. Akademik → bobot tambahan untuk jurusan yang butuh nilai tinggi
        4. Ambil Top-3 berdasarkan skor gabungan

        Args:
            X: DataFrame fitur siswa
            riasec_type: tipe RIASEC dominan (misalnya "Investigative")
            riasec_proba: probabilitas tiap tipe RIASEC
            akademik_scores: nilai akademik (opsional, untuk penyesuaian)

        Returns:
            List[Dict] berisi 3 jurusan dengan nama, skor, dan alasan
        """
        # Skor dari ML model (probabilitas)
        ml_scores = self._get_ml_scores(X)

        # Skor dari aturan RIASEC
        rule_scores = self._get_rule_scores(riasec_proba)

        # Gabungkan: 60% ML + 40% rule-based
        combined = {}
        all_jurusan = set(list(ml_scores.keys()) + list(rule_scores.keys()))
        for jurusan in all_jurusan:
            ml = ml_scores.get(jurusan, 0.0)
            rule = rule_scores.get(jurusan, 0.0)
            combined[jurusan] = 0.6 * ml + 0.4 * rule

        # Penyesuaian akademik (opsional)
        if akademik_scores:
            combined = self._apply_akademik_boost(combined, akademik_scores)

        # Urutkan dan ambil Top-3
        sorted_jurusan = sorted(combined.items(), key=lambda x: -x[1])[:3]

        # Tambahkan alasan (dari rule-based RIASEC)
        result = []
        rule_majors = {m["jurusan"]: m["alasan"] for m in RIASEC_TO_MAJORS.get(riasec_type, [])}

        for rank, (jurusan, score) in enumerate(sorted_jurusan, 1):
            alasan = rule_majors.get(
                jurusan,
                f"Kombinasi kemampuan dan minatmu sangat sesuai dengan {jurusan}"
            )
            result.append({
                "rank": rank,
                "jurusan": jurusan,
                "match_score": round(score * 100, 1),  # konversi ke persen
                "alasan": alasan,
            })

        return result

    # ------------------------------------------------------------------
    # Helper: ML Scores
    # ------------------------------------------------------------------
    def _get_ml_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """Dapatkan probabilitas dari model ML untuk tiap jurusan."""
        try:
            X_aligned = self._align_features(X)
            proba = self.pipeline.predict_proba(X_aligned)[0]
            classes = self.label_encoder.classes_
            return {cls: float(p) for cls, p in zip(classes, proba)}
        except Exception as e:
            logger.warning(f"ML score error: {e}")
            return {}

    def _get_rule_scores(self, riasec_proba: Dict[str, float]) -> Dict[str, float]:
        """
        Hitung skor jurusan berdasarkan aturan RIASEC.

        Setiap tipe RIASEC memiliki daftar jurusan yang cocok.
        Bobot jurusan = probabilitas tipe RIASEC × (rank dari bawah / total jurusan)
        """
        scores = {}
        for riasec_type, proba in riasec_proba.items():
            majors = RIASEC_TO_MAJORS.get(riasec_type, [])
            n = len(majors)
            for rank, major_info in enumerate(majors):
                jurusan = major_info["jurusan"]
                # Jurusan di ranking atas (index 0) dapat bobot lebih tinggi
                weight = (n - rank) / n
                scores[jurusan] = scores.get(jurusan, 0.0) + proba * weight

        # Normalisasi ke 0–1
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _apply_akademik_boost(
        self,
        scores: Dict[str, float],
        akademik: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Penyesuaian skor berdasarkan nilai akademik spesifik.

        Contoh: nilai matematika tinggi → boost Informatika, Statistik
        """
        boosts = {
            "Informatika":    ["matematika", "informatika", "logika"],
            "Kedokteran":     ["ipa", "biologi", "kimia"],
            "Akuntansi":      ["matematika", "ips", "numerasi"],
            "DKV":            ["seni_budaya", "kreativitas"],
            "Teknik Sipil":   ["matematika", "ipa", "fisika"],
            "Ilmu Komunikasi":["bahasa_indonesia", "bahasa_inggris", "komunikasi"],
            "Hukum":          ["pkn", "bahasa_indonesia", "literasi"],
            "Psikologi":      ["biologi", "bahasa_indonesia", "komunikasi"],
        }

        boosted = dict(scores)
        for jurusan, relevant_subjects in boosts.items():
            if jurusan in boosted:
                avg_boost = np.mean([
                    akademik.get(subj, 75) / 100
                    for subj in relevant_subjects
                    if akademik.get(subj) is not None
                ] or [0])
                # Boost maksimal 15% dari skor asli
                boosted[jurusan] = boosted[jurusan] * (1 + 0.15 * avg_boost)

        return boosted

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Model Major Recommender disimpan: {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        logger.info(f"Model Major Recommender dimuat: {path}")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.feature_names]
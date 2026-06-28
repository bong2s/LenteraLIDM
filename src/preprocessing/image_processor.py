"""
=============================================================
MODUL: image_processor.py  (Dataset Real v2)
=============================================================
Mengekstrak fitur numerik dari gambar tulisan tangan menggunakan
OpenCV, lalu menghitung kecenderungan Big Five dan RIASEC dari
fitur-fitur tersebut berdasarkan prinsip grafologi.

FITUR YANG DIEKSTRAK (10 fitur OpenCV):
  1. letter_size   — rata-rata ukuran huruf
  2. slant         — sudut kemiringan tulisan
  3. pressure      — tekanan pena (kegelapan piksel)
  4. spacing       — jarak antar huruf/kata
  5. readability   — keterbacaan (kompleksitas kontur)
  6. neatness      — kerapian (konsistensi baseline)
  7. connectivity  — sambungan huruf (cursive vs cetak)
  8. ornament      — hiasan/dekorasi
  9. baseline      — kelurusan baris
  10. density      — kepadatan tinta

OUTPUT TAMBAHAN (dihitung dari 10 fitur di atas):
  - big_five_scores : {'Openness': 0.7, 'Conscientiousness': 0.6, ...}
  - big_five_dominant: 'Openness'
  - riasec_tendency : {'Realistic': 0.3, 'Investigative': 0.7, ...}
=============================================================
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Grafologi: Mapping fitur tulisan → Big Five
# ------------------------------------------------------------------
# Setiap Big Five dihitung sebagai kombinasi berbobot dari fitur gambar
# Bobot berdasarkan literatur grafologi:
#   Openness        : ornamen, kemiringan, konektivitas tinggi
#   Conscientiousness: kerapian, baseline lurus, sedikit ornamen
#   Extraversion    : huruf besar, tekanan kuat, miring kanan
#   Agreeableness   : konektivitas tinggi, jarak longgar, tekanan ringan
#   Neuroticism     : ketidak-konsistenan baseline, tekanan tidak merata

GRAFOLOGI_BIGFIVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Openness": {
        "ornament": +0.35, "slant": +0.25, "connectivity": +0.20,
        "spacing": +0.10, "density": +0.10,
    },
    "Conscientiousness": {
        "neatness": +0.35, "baseline": +0.30, "readability": +0.20,
        "ornament": -0.10, "pressure": +0.05,
    },
    "Extraversion": {
        "letter_size": +0.35, "pressure": +0.30, "slant": +0.20,
        "spacing": +0.10, "density": +0.05,
    },
    "Agreeableness": {
        "connectivity": +0.35, "spacing": +0.25, "pressure": -0.20,
        "neatness": +0.10, "readability": +0.10,
    },
    "Neuroticism": {
        "neatness": -0.40, "baseline": -0.30, "readability": -0.20,
        "ornament": +0.05, "density": +0.05,
    },
}

# Big Five → RIASEC dominant mapping (psikologi karier)
BIGFIVE_TO_RIASEC_DOMINANT: Dict[str, str] = {
    "Openness":          "Artistic",
    "Conscientiousness": "Conventional",
    "Extraversion":      "Enterprising",
    "Agreeableness":     "Social",
    "Neuroticism":       "Artistic",
}

# Big Five → bobot RIASEC (untuk skor probabilitas)
BIGFIVE_TO_RIASEC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Openness": {
        "Realistic": 0.10, "Investigative": 0.25,
        "Artistic": 0.35,  "Social": 0.15,
        "Enterprising": 0.10, "Conventional": 0.05,
    },
    "Conscientiousness": {
        "Realistic": 0.20, "Investigative": 0.15,
        "Artistic": 0.05,  "Social": 0.10,
        "Enterprising": 0.15, "Conventional": 0.35,
    },
    "Extraversion": {
        "Realistic": 0.10, "Investigative": 0.05,
        "Artistic": 0.10,  "Social": 0.25,
        "Enterprising": 0.40, "Conventional": 0.10,
    },
    "Agreeableness": {
        "Realistic": 0.10, "Investigative": 0.10,
        "Artistic": 0.15,  "Social": 0.40,
        "Enterprising": 0.15, "Conventional": 0.10,
    },
    "Neuroticism": {
        "Realistic": 0.05, "Investigative": 0.20,
        "Artistic": 0.40,  "Social": 0.15,
        "Enterprising": 0.05, "Conventional": 0.15,
    },
}


class HandwritingFeatureExtractor:
    """
    Ekstrak fitur tulisan tangan dari gambar menggunakan OpenCV.

    Cara pakai:
        extractor = HandwritingFeatureExtractor()
        features  = extractor.extract("path/ke/gambar.jpg")
        result    = extractor.extract_full("path/ke/gambar.jpg")
        # result berisi: raw_features, big_five_scores, riasec_tendency
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    # ------------------------------------------------------------------
    # API UTAMA: extract() — 10 fitur numerik
    # ------------------------------------------------------------------
    def extract(self, image_path: str) -> Dict[str, float]:
        """Baca gambar dari file dan ekstrak 10 fitur tulisan tangan."""
        img = self._load(image_path)
        if img is None:
            return self._defaults()
        return self._compute_features(img)

    def extract_from_bytes(self, image_bytes: bytes) -> Dict[str, float]:
        """Ekstrak 10 fitur dari bytes gambar (dipakai FastAPI)."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return self._defaults()
        img = cv2.resize(img, self.target_size)
        return self._compute_features(img)

    # ------------------------------------------------------------------
    # API LENGKAP: extract_full() — fitur + Big Five + RIASEC
    # ------------------------------------------------------------------
    def extract_full(self, image_path: str) -> Dict:
        """
        Ekstrak fitur gambar LENGKAP: fitur dasar + Big Five + RIASEC.

        Returns:
            {
                'raw_features'  : {'letter_size': 4.2, ...},  # 10 fitur
                'big_five_scores': {'Openness': 6.8, ...},    # 5 skor (0-10)
                'big_five_dominant': 'Openness',
                'riasec_tendency'  : {'Artistic': 0.35, ...}, # 6 skor (sum=1)
            }
        """
        img = self._load(image_path)
        if img is None:
            return self._full_defaults()
        return self._compute_full(img)

    def extract_full_from_bytes(self, image_bytes: bytes) -> Dict:
        """Versi extract_full() yang menerima bytes (untuk FastAPI)."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return self._full_defaults()
        img = cv2.resize(img, self.target_size)
        return self._compute_full(img)

    # ------------------------------------------------------------------
    # INTERNAL: Hitung semua fitur + Big Five + RIASEC
    # ------------------------------------------------------------------
    def _compute_full(self, gray: np.ndarray) -> Dict:
        raw = self._compute_features(gray)
        big5 = self._compute_bigfive(raw)
        dominant = max(big5, key=big5.get)
        riasec = self._bigfive_to_riasec(big5)
        return {
            "raw_features":     raw,
            "big_five_scores":  big5,
            "big_five_dominant": dominant,
            "riasec_tendency":  riasec,
        }

    # ------------------------------------------------------------------
    # FITUR DASAR: 10 fitur OpenCV
    # ------------------------------------------------------------------
    def _compute_features(self, gray: np.ndarray) -> Dict[str, float]:
        binary = self._binarize(gray)
        raw = {
            "letter_size":  self._letter_size(binary),
            "slant":        self._slant(binary),
            "pressure":     self._pressure(gray),
            "spacing":      self._spacing(binary),
            "readability":  self._readability(binary),
            "neatness":     self._neatness(binary),
            "connectivity": self._connectivity(binary),
            "ornament":     self._ornament(binary),
            "baseline":     self._baseline(binary),
            "density":      self._density(binary),
        }
        return {k: float(np.clip(v, 0.0, 10.0)) for k, v in raw.items()}

    # ------------------------------------------------------------------
    # BIG FIVE dari fitur tulisan (grafologi)
    # ------------------------------------------------------------------
    def _compute_bigfive(self, f: Dict[str, float]) -> Dict[str, float]:
        """
        Hitung 5 skor Big Five dari 10 fitur gambar.
        Setiap skor dalam rentang 0-10.
        """
        scores: Dict[str, float] = {}
        for trait, weights in GRAFOLOGI_BIGFIVE_WEIGHTS.items():
            raw = 5.0  # titik tengah (nilai netral)
            for feature, w in weights.items():
                val = f.get(feature, 5.0)
                raw += w * (val - 5.0)
            scores[trait] = float(np.clip(raw, 1.0, 10.0))
        return {k: round(v, 2) for k, v in scores.items()}

    # ------------------------------------------------------------------
    # RIASEC dari Big Five scores
    # ------------------------------------------------------------------
    def _bigfive_to_riasec(self, big5: Dict[str, float]) -> Dict[str, float]:
        """
        Konversi skor Big Five ke distribusi probabilitas RIASEC.
        Output: dict dengan total mendekati 1.0
        """
        riasec_raw = {t: 0.0 for t in
                      ["Realistic","Investigative","Artistic","Social","Enterprising","Conventional"]}

        total_big5 = sum(big5.values()) or 1.0
        for trait, score in big5.items():
            weight_normalized = score / total_big5
            if trait not in BIGFIVE_TO_RIASEC_WEIGHTS:
                continue
            for riasec_type, w in BIGFIVE_TO_RIASEC_WEIGHTS[trait].items():
                riasec_raw[riasec_type] += weight_normalized * w

        total = sum(riasec_raw.values()) or 1.0
        return {k: round(v / total, 4) for k, v in sorted(riasec_raw.items(),
                                                            key=lambda x: -x[1])}

    # ------------------------------------------------------------------
    # 10 Fitur OpenCV (implementasi)
    # ------------------------------------------------------------------
    def _load(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Gambar tidak bisa dibaca: {path}")
            return None
        return cv2.resize(img, self.target_size)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def _letter_size(self, binary: np.ndarray) -> float:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours if 20 < cv2.contourArea(c) < 5000]
        if not areas:
            return 5.0
        return float(np.clip(1 + (np.mean(areas) - 20) / (5000 - 20) * 9, 1, 10))

    def _slant(self, binary: np.ndarray) -> float:
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
        if lines is None:
            return 5.0
        angles = []
        for line in lines[:30]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return 5.0
        return float(np.clip(5 + np.mean(angles) / 45 * 4, 1, 10))

    def _pressure(self, gray: np.ndarray) -> float:
        return float(np.clip((255 - np.mean(gray)) / 255 * 10, 0, 10))

    def _spacing(self, binary: np.ndarray) -> float:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted(
            [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 5 and cv2.boundingRect(c)[3] > 5],
            key=lambda b: b[0]
        )
        if len(boxes) < 2:
            return 5.0
        gaps = [boxes[i+1][0] - (boxes[i][0] + boxes[i][2])
                for i in range(len(boxes)-1)
                if 0 < boxes[i+1][0] - (boxes[i][0] + boxes[i][2]) < 200]
        return float(np.clip(np.mean(gaps) / 100 * 10, 0, 10)) if gaps else 5.0

    def _readability(self, binary: np.ndarray) -> float:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vals = []
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if area > 30 and peri > 0:
                vals.append((4 * np.pi * area) / (peri ** 2))
        return float(np.clip(np.mean(vals) * 10, 0, 10)) if vals else 5.0

    def _neatness(self, binary: np.ndarray) -> float:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        baselines = [y + h for c in contours
                     for (_, y, _, h) in [cv2.boundingRect(c)]
                     if cv2.contourArea(c) > 50]
        if len(baselines) < 3:
            return 5.0
        return float(np.clip(max(0, 10 - np.std(baselines) / 200 * 10), 0, 10))

    def _connectivity(self, binary: np.ndarray) -> float:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        n_before, _ = cv2.connectedComponents(binary)
        n_after,  _ = cv2.connectedComponents(closed)
        if n_before == 0:
            return 5.0
        return float(np.clip((1 - n_after / max(n_before, 1)) * 10, 0, 10))

    def _ornament(self, binary: np.ndarray) -> float:
        outer, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or not outer:
            return 3.0
        inner = sum(1 for h in hierarchy[0] if h[3] != -1)
        return float(np.clip(inner / max(len(outer), 1) * 15, 0, 10))

    def _baseline(self, binary: np.ndarray) -> float:
        h_proj = np.sum(binary, axis=1) / 255
        smoothed = np.convolve(h_proj, np.ones(5) / 5, mode="same")
        threshold = np.max(smoothed) * 0.1
        valleys = np.where(smoothed < threshold)[0]
        if len(valleys) < 2:
            return 7.0
        diffs = np.diff(valleys)
        diffs = diffs[diffs > 10]
        if len(diffs) < 2:
            return 7.0
        cv = np.std(diffs) / (np.mean(diffs) + 1e-6)
        return float(np.clip(10 - cv * 10, 0, 10))

    def _density(self, binary: np.ndarray) -> float:
        return float(np.clip(np.sum(binary > 0) / binary.size * 10 * 2, 0, 10))

    # ------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------
    def _defaults(self) -> Dict[str, float]:
        return {
            "letter_size": 5.0, "slant": 5.0, "pressure": 5.0,
            "spacing": 5.0, "readability": 5.0, "neatness": 5.0,
            "connectivity": 5.0, "ornament": 3.0, "baseline": 7.0,
            "density": 5.0,
        }

    def _full_defaults(self) -> Dict:
        raw = self._defaults()
        return {
            "raw_features":      raw,
            "big_five_scores":   {t: 5.0 for t in
                                  ["Openness","Conscientiousness","Extraversion",
                                   "Agreeableness","Neuroticism"]},
            "big_five_dominant": "Conscientiousness",
            "riasec_tendency":   {t: round(1/6, 4) for t in
                                  ["Realistic","Investigative","Artistic",
                                   "Social","Enterprising","Conventional"]},
        }
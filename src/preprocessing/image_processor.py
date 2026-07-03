"""
=============================================================
MODUL: image_processor.py  (Dataset Real v3 — RIASEC Direct)
=============================================================
Mengekstrak 10 fitur numerik dari gambar tulisan tangan
menggunakan OpenCV.

Pipeline BigFive telah dihapus sepenuhnya.
extract_full() sekarang merupakan alias sederhana untuk extract().

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
=============================================================
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class HandwritingFeatureExtractor:
    """
    Ekstrak 10 fitur tulisan tangan dari gambar menggunakan OpenCV.
    Tidak ada lagi pipeline Big Five.

    Cara pakai:
        extractor = HandwritingFeatureExtractor()
        features  = extractor.extract("path/ke/gambar.jpg")
        features  = extractor.extract_from_bytes(image_bytes)
    """

    FEATURE_NAMES = [
        "letter_size", "slant", "pressure", "spacing", "readability",
        "neatness", "connectivity", "ornament", "baseline", "density",
    ]

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    # ------------------------------------------------------------------
    # API UTAMA
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

    # Alias untuk kompatibilitas — kembalikan dict fitur saja (bukan dict+bigfive)
    def extract_full(self, image_path: str) -> Dict[str, float]:
        """Alias untuk extract() — dulu mengembalikan dict berisi bigfive, kini hanya fitur."""
        return self.extract(image_path)

    def extract_full_from_bytes(self, image_bytes: bytes) -> Dict[str, float]:
        """Alias untuk extract_from_bytes()."""
        return self.extract_from_bytes(image_bytes)

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
            [cv2.boundingRect(c) for c in contours
             if cv2.boundingRect(c)[2] > 5 and cv2.boundingRect(c)[3] > 5],
            key=lambda b: b[0],
        )
        if len(boxes) < 2:
            return 5.0
        gaps = [
            boxes[i + 1][0] - (boxes[i][0] + boxes[i][2])
            for i in range(len(boxes) - 1)
            if 0 < boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) < 200
        ]
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
        baselines = [
            y + h
            for c in contours
            for (_, y, _, h) in [cv2.boundingRect(c)]
            if cv2.contourArea(c) > 50
        ]
        if len(baselines) < 3:
            return 5.0
        return float(np.clip(max(0, 10 - np.std(baselines) / 200 * 10), 0, 10))

    def _connectivity(self, binary: np.ndarray) -> float:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        n_before, _ = cv2.connectedComponents(binary)
        n_after, _  = cv2.connectedComponents(closed)
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
            "letter_size": 5.0,
            "slant":       5.0,
            "pressure":    5.0,
            "spacing":     5.0,
            "readability": 5.0,
            "neatness":    5.0,
            "connectivity": 5.0,
            "ornament":    3.0,
            "baseline":    7.0,
            "density":     5.0,
        }
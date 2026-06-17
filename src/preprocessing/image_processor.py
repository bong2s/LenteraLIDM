"""
=============================================================
MODUL: image_processor.py
=============================================================
TUJUAN:
  Mengekstrak fitur numerik dari gambar tulisan tangan
  menggunakan OpenCV (Computer Vision).

CARA KERJA SINGKAT:
  Gambar tulisan tangan → Grayscale → Threshold (binerisasi)
  → Deteksi kontur & komponen → Hitung 10 fitur numerik
  → Fitur ini dimasukkan ke model ML

FITUR YANG DIEKSTRAK:
  1. letter_size_score   — rata-rata ukuran huruf (besar/kecil)
  2. slant_angle         — sudut kemiringan tulisan
  3. pressure_score      — tekanan pena (terang/gelap piksel)
  4. spacing_score       — jarak antar huruf/kata
  5. readability_score   — keterbacaan (kompleksitas kontur)
  6. neatness_score      — kerapian (konsistensi baseline)
  7. connectivity_score  — sambungan huruf (cursive vs cetak)
  8. ornament_score      — hiasan/dekorasi di tulisan
  9. line_straightness   — kelurusan baris tulisan
  10. density_score      — kepadatan tinta di halaman
=============================================================
"""

import cv2
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HandwritingFeatureExtractor:
    """
    Kelas utama untuk mengekstrak fitur dari gambar tulisan tangan.

    Cara pakai:
        extractor = HandwritingFeatureExtractor()
        fitur = extractor.extract(path_gambar)
        # Hasilnya: dict berisi 10 nilai numerik
    """

    def __init__(self, target_size: tuple = (512, 512)):
        """
        Args:
            target_size: ukuran resize gambar sebelum diproses
                         (lebar, tinggi) dalam piksel
        """
        self.target_size = target_size

    # ------------------------------------------------------------------
    # FUNGSI UTAMA: extract()
    # ------------------------------------------------------------------
    def extract(self, image_path: str) -> Dict[str, float]:
        """
        Baca gambar dari file dan ekstrak semua fitur tulisan tangan.

        Args:
            image_path: path ke file gambar (.png / .jpg)

        Returns:
            dict dengan 10 fitur numerik (nilai 0.0 – 10.0)
        """
        img = self._load_image(image_path)
        if img is None:
            logger.warning(f"Gambar tidak bisa dibaca: {image_path}")
            return self._default_features()

        return self._extract_features(img)

    def extract_from_bytes(self, image_bytes: bytes) -> Dict[str, float]:
        """
        Ekstrak fitur dari bytes gambar (dipakai oleh FastAPI untuk upload).

        Args:
            image_bytes: isi file gambar dalam bentuk bytes

        Returns:
            dict dengan 10 fitur numerik
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self._default_features()

        img = cv2.resize(img, self.target_size)
        return self._extract_features(img)

    # ------------------------------------------------------------------
    # LANGKAH 1: Muat dan siapkan gambar
    # ------------------------------------------------------------------
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Baca gambar → konversi ke grayscale → resize ke ukuran standar.
        Grayscale mempermudah analisis karena hanya ada 1 channel warna.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, self.target_size)
        return img

    # ------------------------------------------------------------------
    # LANGKAH 2: Binerisasi (Thresholding)
    # ------------------------------------------------------------------
    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Ubah gambar grayscale menjadi hitam-putih murni.

        Otsu's thresholding: secara otomatis memilih nilai ambang batas
        yang memisahkan tinta (hitam) dari kertas (putih).

        Hasilnya: piksel tinta = 255 (putih), latar = 0 (hitam)
        """
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    # ------------------------------------------------------------------
    # LANGKAH 3: Ekstrak semua fitur
    # ------------------------------------------------------------------
    def _extract_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Memanggil semua fungsi ekstraksi fitur dan menggabungkan hasilnya.
        """
        binary = self._binarize(gray)

        features = {
            "letter_size_score":  self._compute_letter_size(binary),
            "slant_angle":        self._compute_slant(binary),
            "pressure_score":     self._compute_pressure(gray),
            "spacing_score":      self._compute_spacing(binary),
            "readability_score":  self._compute_readability(binary),
            "neatness_score":     self._compute_neatness(binary),
            "connectivity_score": self._compute_connectivity(binary),
            "ornament_score":     self._compute_ornament(binary),
            "line_straightness":  self._compute_line_straightness(binary),
            "density_score":      self._compute_density(binary),
        }

        # Normalisasi: semua nilai dijaga di rentang 0–10
        return {k: float(np.clip(v, 0, 10)) for k, v in features.items()}

    # ------------------------------------------------------------------
    # FITUR 1: Ukuran Huruf
    # ------------------------------------------------------------------
    def _compute_letter_size(self, binary: np.ndarray) -> float:
        """
        Hitung rata-rata ukuran huruf menggunakan bounding box kontur.

        Huruf besar → skor tinggi (mendekati 10)
        Huruf kecil → skor rendah (mendekati 1)

        RIASEC mapping:
          Skor tinggi → Realistik (tegas, to the point)
          Skor rendah → Konvensional (teliti, rapi)
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 5.0

        areas = []
        for c in contours:
            area = cv2.contourArea(c)
            if 20 < area < 5000:  # abaikan noise dan benda sangat besar
                areas.append(area)

        if not areas:
            return 5.0

        avg_area = np.mean(areas)
        # Normalisasi: area 20–5000 → skor 1–10
        score = 1 + (avg_area - 20) / (5000 - 20) * 9
        return float(np.clip(score, 1, 10))

    # ------------------------------------------------------------------
    # FITUR 2: Sudut Kemiringan (Slant)
    # ------------------------------------------------------------------
    def _compute_slant(self, binary: np.ndarray) -> float:
        """
        Deteksi sudut kemiringan tulisan menggunakan Hough Line Transform.

        Tulisan miring ke kanan → skor tinggi (impulsif, percaya diri)
        Tulisan tegak → skor tengah (~5)
        Tulisan miring ke kiri → skor rendah (hati-hati, introspektif)

        Cara kerja Hough Lines:
          Algoritma mencari garis lurus dalam gambar. Setiap garis tulisan
          menghasilkan sudut θ. Rata-rata θ = kemiringan dominan.
        """
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

        if lines is None:
            return 5.0

        angles = []
        for line in lines[:30]:  # ambil 30 garis terkuat
            rho, theta = line[0]
            angle_deg = np.degrees(theta) - 90
            if -45 < angle_deg < 45:  # hanya garis vertikal (tulisan)
                angles.append(angle_deg)

        if not angles:
            return 5.0

        mean_angle = np.mean(angles)
        # Kemiringan -45 s/d +45 derajat → skor 1–10
        score = 5 + mean_angle / 45 * 4
        return float(np.clip(score, 1, 10))

    # ------------------------------------------------------------------
    # FITUR 3: Tekanan Pena (Pressure)
    # ------------------------------------------------------------------
    def _compute_pressure(self, gray: np.ndarray) -> float:
        """
        Estimasi tekanan pena dari kegelapan piksel tinta.

        Prinsip:
          Tinta dengan tekanan kuat → piksel lebih gelap (nilai piksel rendah)
          Tinta ringan → piksel lebih terang (nilai piksel tinggi)

        Kita invert: nilai gelap → skor pressure tinggi

        RIASEC mapping:
          Tekanan kuat → Enterprising/Realistik (energik, dominan)
          Tekanan ringan → Investigatif/Artistik (lembut, reflektif)
        """
        mean_intensity = np.mean(gray)
        # 0 (hitam total) → tekanan tinggi, 255 (putih) → tidak ada tinta
        # Invert: semakin gelap semakin tinggi skornya
        score = (255 - mean_intensity) / 255 * 10
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 4: Jarak Antar Huruf (Spacing)
    # ------------------------------------------------------------------
    def _compute_spacing(self, binary: np.ndarray) -> float:
        """
        Hitung jarak rata-rata antar komponen tulisan (huruf/kata).

        Caranya:
          1. Temukan semua bounding box kontur
          2. Urutkan dari kiri ke kanan
          3. Hitung celah horizontal antar kotak

        Jarak lebar → skor tinggi (kepribadian terbuka, membutuhkan ruang)
        Jarak sempit → skor rendah (hemat, introvert)
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) < 2:
            return 5.0

        # Ambil bounding boxes, filter noise
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 5 and h > 5:
                boxes.append((x, y, w, h))

        if len(boxes) < 2:
            return 5.0

        # Urutkan berdasarkan posisi x (kiri ke kanan)
        boxes.sort(key=lambda b: b[0])

        gaps = []
        for i in range(len(boxes) - 1):
            right_edge = boxes[i][0] + boxes[i][2]
            left_next = boxes[i + 1][0]
            gap = left_next - right_edge
            if 0 < gap < 200:  # abaikan jarak antar baris
                gaps.append(gap)

        if not gaps:
            return 5.0

        avg_gap = np.mean(gaps)
        # Gap 0–100 piksel → skor 0–10
        score = avg_gap / 100 * 10
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 5: Keterbacaan (Readability)
    # ------------------------------------------------------------------
    def _compute_readability(self, binary: np.ndarray) -> float:
        """
        Estimasi keterbacaan dari complexitas kontur.

        Ide: huruf yang rapi dan terbaca punya kontur yang sederhana
        (perimeter rendah relatif terhadap area = "sirkularitas" tinggi).
        Huruf susah dibaca punya kontur rumit/berkelok.

        Rumus: compactness = 4π * area / perimeter²
        Nilai mendekati 1 = bentuk lingkaran sempurna (sangat rapi)
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 5.0

        compactness_values = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area > 30 and perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                compactness_values.append(compactness)

        if not compactness_values:
            return 5.0

        avg_compactness = np.mean(compactness_values)
        # compactness 0–1 → skor 0–10
        score = avg_compactness * 10
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 6: Kerapian / Konsistensi Baseline
    # ------------------------------------------------------------------
    def _compute_neatness(self, binary: np.ndarray) -> float:
        """
        Hitung kerapian tulisan dari konsistensi posisi vertikal huruf.

        Cara:
          1. Cari titik bawah tiap kontur (baseline)
          2. Hitung standar deviasi → variasi tinggi → kurang rapi

        Tulisan rapi → baseline konsisten → skor tinggi
        Tulisan acak → baseline bervariasi → skor rendah
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 5.0

        baselines = []
        for c in contours:
            _, y, _, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > 50:
                baselines.append(y + h)  # titik bawah huruf

        if len(baselines) < 3:
            return 5.0

        std_dev = np.std(baselines)
        # Deviasi rendah → sangat rapi (skor 10), deviasi tinggi → tidak rapi
        # Deviasi maks ~200 piksel untuk gambar 512x512
        score = max(0, 10 - (std_dev / 200 * 10))
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 7: Konektivitas (Cursive vs Cetak)
    # ------------------------------------------------------------------
    def _compute_connectivity(self, binary: np.ndarray) -> float:
        """
        Ukur seberapa tersambung huruf dalam tulisan (cursive vs cetak).

        Cara:
          Tulisan cursive punya sedikit komponen terpisah (huruf nyambung)
          Tulisan cetak punya banyak komponen terpisah (huruf lepas)

        Skor tinggi → banyak koneksi = cursive (sosial, ekspresif)
        Skor rendah → sedikit koneksi = cetak (analitis, terstruktur)
        """
        # Gunakan morphological closing untuk menghubungkan huruf yang dekat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Hitung komponen terhubung sebelum dan sesudah closing
        num_before, _ = cv2.connectedComponents(binary)
        num_after, _ = cv2.connectedComponents(closed)

        if num_before == 0:
            return 5.0

        # Rasio pengurangan komponen = tingkat konektivitas
        connectivity_ratio = 1 - (num_after / max(num_before, 1))
        score = connectivity_ratio * 10
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 8: Hiasan / Ornamen
    # ------------------------------------------------------------------
    def _compute_ornament(self, binary: np.ndarray) -> float:
        """
        Deteksi elemen dekoratif atau hiasan dalam tulisan.

        Cara: Hitung jumlah "loop" (lubang dalam huruf) menggunakan
        RETR_CCOMP yang mendeteksi kontur berlevel (luar + dalam).
        Loop ekstra (selain dari huruf seperti 'o', 'a') dianggap ornamen.

        Skor tinggi → banyak ornamen (kreatif, artistik)
        Skor rendah → tulisan polos (realistik, konvensional)
        """
        # Deteksi semua kontur termasuk yang di dalam (holes)
        contours_outer, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_all, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(contours_outer) == 0:
            return 3.0

        # Hitung jumlah "inner holes"
        inner_count = 0
        for h in hierarchy[0]:
            if h[3] != -1:  # parent bukan -1 = kontur dalam (hole)
                inner_count += 1

        outer_count = len(contours_outer)
        ratio = inner_count / max(outer_count, 1)
        # Surat normal: 'o','a','e','p' punya 1 hole per huruf
        # Ratio > 0.3 dianggap banyak ornamen
        score = min(ratio * 15, 10)
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 9: Kelurusan Baris
    # ------------------------------------------------------------------
    def _compute_line_straightness(self, binary: np.ndarray) -> float:
        """
        Ukur seberapa lurus baris tulisan (tidak naik-turun).

        Cara:
          1. Proyeksikan gambar secara vertikal (hitung jumlah piksel per baris)
          2. Temukan puncak (baris yang penuh tinta = garis tulisan)
          3. Ukur konsistensi jarak antar baris

        Baris lurus rapi → skor tinggi (disiplin, terorganisir)
        Baris naik-turun → skor rendah (spontan, tidak formal)
        """
        # Horizontal projection profile
        h_proj = np.sum(binary, axis=1) / 255

        # Smooth projection
        kernel = np.ones(5) / 5
        smoothed = np.convolve(h_proj, kernel, mode='same')

        # Temukan lembah (antar baris)
        threshold = np.max(smoothed) * 0.1
        valleys = np.where(smoothed < threshold)[0]

        if len(valleys) < 2:
            return 7.0

        # Hitung jarak antar lembah (jarak antar baris)
        diffs = np.diff(valleys)
        diffs = diffs[diffs > 10]  # abaikan jarak sangat kecil

        if len(diffs) < 2:
            return 7.0

        cv_diffs = np.std(diffs) / (np.mean(diffs) + 1e-6)
        # Coefficient of variation rendah → baris sangat konsisten
        score = max(0, 10 - cv_diffs * 10)
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # FITUR 10: Kepadatan Tinta (Density)
    # ------------------------------------------------------------------
    def _compute_density(self, binary: np.ndarray) -> float:
        """
        Hitung persentase area yang tertutup tinta.

        Density tinggi → tulisan padat, banyak isi (intens, terperinci)
        Density rendah → tulisan renggang, sedikit teks (minimalis, efisien)
        """
        ink_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        density = ink_pixels / total_pixels
        score = density * 10 * 2  # ×2 karena tulisan jarang memenuhi >50%
        return float(np.clip(score, 0, 10))

    # ------------------------------------------------------------------
    # Default: jika gambar tidak bisa dibaca
    # ------------------------------------------------------------------
    def _default_features(self) -> Dict[str, float]:
        """Kembalikan nilai tengah (5.0) untuk semua fitur jika gambar error."""
        return {
            "letter_size_score": 5.0,
            "slant_angle": 5.0,
            "pressure_score": 5.0,
            "spacing_score": 5.0,
            "readability_score": 5.0,
            "neatness_score": 5.0,
            "connectivity_score": 5.0,
            "ornament_score": 3.0,
            "line_straightness": 7.0,
            "density_score": 5.0,
        }

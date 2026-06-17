"""
=============================================================
FILE: tests/test_api.py
=============================================================
TUJUAN:
  Test otomatis untuk FastAPI ML Server.
  Jalankan: pytest tests/test_api.py -v

  Test ini mengecek:
  1. Health endpoint
  2. Demo endpoint
  3. Predict endpoint dengan gambar dummy
  4. Validasi error handling
=============================================================
"""

import io
import sys
import os
import json
import pytest
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import numpy as np


# Import app (model tidak harus ada untuk test dasar)
def create_test_client():
    """Buat test client."""
    os.chdir(ROOT)
    from api.main import app
    return TestClient(app)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    os.chdir(ROOT)
    from api.main import app
    with TestClient(app) as c:
        yield c


def create_dummy_png() -> bytes:
    """Buat gambar PNG dummy berisi 'tulisan' sederhana."""
    try:
        import cv2
        img = np.ones((200, 400), dtype=np.uint8) * 255
        cv2.putText(img, "Tulisan Test", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()
    except ImportError:
        # Fallback: buat PNG minimal valid menggunakan bytes
        import struct, zlib

        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        w, h = 10, 10
        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
        raw_rows = b"".join(b"\x00" + b"\xff" * w for _ in range(h))
        idat_data = zlib.compress(raw_rows)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", ihdr_data)
            + png_chunk(b"IDAT", idat_data)
            + png_chunk(b"IEND", b"")
        )
        return png


# ------------------------------------------------------------------
# TEST: Health Check
# ------------------------------------------------------------------
class TestHealth:
    def test_health_ok(self, client):
        """Server harus merespons OK."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_root(self, client):
        """Root endpoint harus ada."""
        resp = client.get("/")
        assert resp.status_code == 200


# ------------------------------------------------------------------
# TEST: Demo Endpoint
# ------------------------------------------------------------------
class TestDemo:
    def test_demo_returns_valid_structure(self, client):
        """Demo endpoint harus mengembalikan struktur respons yang benar."""
        resp = client.get("/predict/demo")
        assert resp.status_code == 200
        data = resp.json()

        # Cek field utama ada
        assert "karakter" in data
        assert "riasec_skor" in data
        assert "rekomendasi_jurusan" in data
        assert "fitur_tulisan" in data

    def test_demo_has_3_jurusan(self, client):
        """Harus ada 3 rekomendasi jurusan."""
        resp = client.get("/predict/demo")
        data = resp.json()
        assert len(data["rekomendasi_jurusan"]) == 3

    def test_demo_riasec_scores_sum_near_1(self, client):
        """Total skor RIASEC harus mendekati 1.0."""
        resp = client.get("/predict/demo")
        data = resp.json()
        total = sum(data["riasec_skor"].values())
        assert abs(total - 1.0) < 0.01


# ------------------------------------------------------------------
# TEST: Predict Endpoint
# ------------------------------------------------------------------
class TestPredict:
    def test_predict_with_valid_image(self, client):
        """Predict harus berhasil dengan gambar valid."""
        img_bytes = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        # Bisa 200 (model ada) atau 503 (model belum ada)
        assert resp.status_code in (200, 503)

        if resp.status_code == 200:
            data = resp.json()
            assert "karakter" in data
            assert "rekomendasi_jurusan" in data
            assert len(data["rekomendasi_jurusan"]) == 3

    def test_predict_with_akademik_data(self, client):
        """Predict dengan data akademik harus diterima."""
        img_bytes = create_dummy_png()
        akademik = json.dumps({"matematika": 88, "ipa": 85, "bahasa_inggris": 84})
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"akademik": akademik},
        )
        assert resp.status_code in (200, 503)

    def test_predict_with_talent_data(self, client):
        """Predict dengan data bakat harus diterima."""
        img_bytes = create_dummy_png()
        talent = json.dumps({"logika": 8, "teknologi": 9, "kreativitas": 5})
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"talent": talent},
        )
        assert resp.status_code in (200, 503)

    def test_predict_without_file_returns_422(self, client):
        """Request tanpa file harus dikembalikan 422 (Unprocessable Entity)."""
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_predict_with_wrong_file_type(self, client):
        """File bukan gambar harus dikembalikan 400."""
        resp = client.post(
            "/predict",
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_predict_with_invalid_akademik_json(self, client):
        """JSON akademik yang salah harus dikembalikan 400."""
        img_bytes = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"akademik": "bukan json"},
        )
        assert resp.status_code == 400

    def test_predict_with_out_of_range_value(self, client):
        """Nilai di luar range (>100) harus dikembalikan 422."""
        img_bytes = create_dummy_png()
        akademik = json.dumps({"matematika": 999})  # tidak valid, >100
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"akademik": akademik},
        )
        assert resp.status_code in (400, 422)


# ------------------------------------------------------------------
# TEST: Image Processor (unit test)
# ------------------------------------------------------------------
class TestImageProcessor:
    def test_extract_returns_10_features(self):
        """Ekstraksi gambar harus menghasilkan 10 fitur."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        img_bytes = create_dummy_png()
        features = extractor.extract_from_bytes(img_bytes)
        assert len(features) == 10

    def test_features_in_range_0_to_10(self):
        """Semua nilai fitur harus antara 0 dan 10."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        img_bytes = create_dummy_png()
        features = extractor.extract_from_bytes(img_bytes)
        for name, val in features.items():
            assert 0.0 <= val <= 10.0, f"Fitur '{name}' nilainya {val} di luar range 0-10"

    def test_default_features_on_empty_bytes(self):
        """Bytes kosong harus mengembalikan fitur default."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        features = extractor.extract_from_bytes(b"invalid image data")
        assert "letter_size_score" in features


# ------------------------------------------------------------------
# Jalankan langsung
# ------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
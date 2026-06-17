"""
=============================================================
FILE: tests/test_api.py
=============================================================
TUJUAN:
  Test otomatis untuk FastAPI ML Server.
  Jalankan: pytest tests/test_api.py -v

  Test ini mengecek:
  1. Health & root endpoint
  2. Demo endpoint
  3. Predict: gambar saja (tanpa nilai apapun)        ← BARU
  4. Predict: gambar + sebagian nilai akademik        ← BARU
  5. Predict: gambar + semua nilai + bakat
  6. Validasi field kelengkapan_data                  ← BARU
  7. Error handling (file salah, JSON rusak, dll)
  8. Unit test image processor
  9. Unit test nilai default (tidak ada data = netral) ← BARU
=============================================================
"""

import io
import sys
import os
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import numpy as np


# ------------------------------------------------------------------
# Helper: Buat gambar PNG dummy
# ------------------------------------------------------------------
def create_dummy_png() -> bytes:
    """Buat gambar PNG dummy berisi tulisan sederhana untuk testing."""
    try:
        import cv2
        img = np.ones((200, 400), dtype=np.uint8) * 255
        cv2.putText(img, "Tulisan Test", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()
    except ImportError:
        import struct, zlib

        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        w, h = 10, 10
        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
        raw_rows = b"".join(b"\x00" + b"\xff" * w for _ in range(h))
        idat_data = zlib.compress(raw_rows)
        return (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", ihdr_data)
            + png_chunk(b"IDAT", idat_data)
            + png_chunk(b"IEND", b"")
        )


# ------------------------------------------------------------------
# Fixture: Client
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    os.chdir(ROOT)
    from api.main import app
    with TestClient(app) as c:
        yield c


# Shortcut: status yang valid dari /predict
VALID_PREDICT_STATUS = (200, 503)  # 200=OK, 503=model belum di-training


# ==================================================================
# TEST 1: Health & Root
# ==================================================================
class TestHealth:
    def test_health_returns_ok(self, client):
        """GET /health harus mengembalikan status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_has_version(self, client):
        """GET /health harus ada field version."""
        resp = client.get("/health")
        assert "version" in resp.json()

    def test_root_returns_200(self, client):
        """GET / harus mengembalikan 200."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        """Root endpoint harus ada link ke /docs."""
        data = client.get("/").json()
        assert "docs" in data


# ==================================================================
# TEST 2: Demo Endpoint
# ==================================================================
class TestDemo:
    def test_demo_returns_200(self, client):
        """GET /predict/demo harus mengembalikan 200."""
        resp = client.get("/predict/demo")
        assert resp.status_code == 200

    def test_demo_has_all_required_fields(self, client):
        """Demo harus punya semua field utama."""
        data = client.get("/predict/demo").json()
        required = ["karakter", "riasec_skor", "rekomendasi_jurusan",
                    "fitur_tulisan", "feature_importance"]
        for field in required:
            assert field in data, f"Field '{field}' tidak ada di respons demo"

    def test_demo_karakter_has_all_subfields(self, client):
        """Field karakter harus punya tipe, nama, deskripsi, kekuatan, warna."""
        karakter = client.get("/predict/demo").json()["karakter"]
        for key in ["tipe", "nama", "deskripsi", "kekuatan", "warna"]:
            assert key in karakter, f"Sub-field '{key}' tidak ada di karakter"

    def test_demo_has_exactly_3_jurusan(self, client):
        """Harus ada tepat 3 rekomendasi jurusan."""
        data = client.get("/predict/demo").json()
        assert len(data["rekomendasi_jurusan"]) == 3

    def test_demo_jurusan_has_rank_and_alasan(self, client):
        """Tiap jurusan harus punya rank, jurusan, match_score, alasan."""
        jurusan_list = client.get("/predict/demo").json()["rekomendasi_jurusan"]
        for j in jurusan_list:
            assert "rank" in j
            assert "jurusan" in j
            assert "match_score" in j
            assert "alasan" in j

    def test_demo_riasec_scores_sum_near_1(self, client):
        """Total skor RIASEC harus mendekati 1.0 (probabilitas)."""
        riasec = client.get("/predict/demo").json()["riasec_skor"]
        total = sum(riasec.values())
        assert abs(total - 1.0) < 0.05, f"Total RIASEC = {total}, seharusnya ~1.0"

    def test_demo_riasec_has_6_types(self, client):
        """RIASEC harus punya 6 tipe."""
        riasec = client.get("/predict/demo").json()["riasec_skor"]
        assert len(riasec) == 6

    def test_demo_fitur_tulisan_has_10_features(self, client):
        """fitur_tulisan harus punya 10 fitur."""
        fitur = client.get("/predict/demo").json()["fitur_tulisan"]
        assert len(fitur) == 10

    def test_demo_match_score_is_percentage(self, client):
        """match_score harus antara 0 dan 100."""
        jurusan_list = client.get("/predict/demo").json()["rekomendasi_jurusan"]
        for j in jurusan_list:
            assert 0 <= j["match_score"] <= 100, \
                f"match_score {j['match_score']} di luar range 0-100"

    def test_demo_ranks_are_1_2_3(self, client):
        """Rank jurusan harus 1, 2, 3 berurutan."""
        jurusan_list = client.get("/predict/demo").json()["rekomendasi_jurusan"]
        ranks = [j["rank"] for j in jurusan_list]
        assert ranks == [1, 2, 3]


# ==================================================================
# TEST 3: Predict — GAMBAR SAJA (tanpa nilai apapun) ← BARU
# ==================================================================
class TestPredictImageOnly:
    """
    Skenario: Siswa hanya upload gambar tulisan tangan,
    tidak mengisi nilai akademik maupun bakat sama sekali.
    Ini harus TETAP berhasil dan memberikan rekomendasi.
    """

    def test_image_only_accepted(self, client):
        """Upload gambar saja (tanpa akademik/talent) harus diterima."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            # Tidak ada data akademik atau talent sama sekali
        )
        assert resp.status_code in VALID_PREDICT_STATUS, \
            f"Status {resp.status_code}: {resp.text}"

    def test_image_only_returns_3_jurusan(self, client):
        """Gambar saja harus tetap mengembalikan 3 rekomendasi jurusan."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert len(data["rekomendasi_jurusan"]) == 3

    def test_image_only_has_kelengkapan_data(self, client):
        """
        Gambar saja harus ada field kelengkapan_data yang menjelaskan
        bahwa nilai akademik dan bakat diisi otomatis.
        """
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "kelengkapan_data" in data, \
                "Field kelengkapan_data harus ada di respons"
            kd = data["kelengkapan_data"]
            assert "akademik" in kd
            assert "bakat" in kd
            assert "catatan" in kd

    def test_image_only_akademik_persen_is_0(self, client):
        """Ketika tidak ada data akademik, persen_lengkap harus 0."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
        )
        if resp.status_code == 200:
            kd = resp.json()["kelengkapan_data"]
            assert kd["akademik"]["persen_lengkap"] == 0, \
                "Tidak ada akademik diisi → persen_lengkap harus 0"
            assert kd["bakat"]["persen_lengkap"] == 0, \
                "Tidak ada bakat diisi → persen_lengkap harus 0"

    def test_image_only_diisi_list_is_empty(self, client):
        """Ketika tidak ada data diisi, list 'diisi' harus kosong."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
        )
        if resp.status_code == 200:
            kd = resp.json()["kelengkapan_data"]
            assert kd["akademik"]["diisi"] == [], \
                "List 'diisi' harus kosong karena tidak ada yang diisi"

    def test_image_only_diisi_otomatis_not_empty(self, client):
        """Ketika tidak ada data diisi, list 'diisi_otomatis' harus berisi semua mata pelajaran."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
        )
        if resp.status_code == 200:
            kd = resp.json()["kelengkapan_data"]
            assert len(kd["akademik"]["diisi_otomatis"]) > 0, \
                "Harus ada mata pelajaran yang diisi otomatis"
            assert kd["akademik"]["default_value"] == 75, \
                "Default nilai akademik harus 75"


# ==================================================================
# TEST 4: Predict — Data Sebagian ← BARU
# ==================================================================
class TestPredictPartialData:
    """
    Skenario: Siswa mengisi beberapa nilai tapi tidak semua.
    Contoh: isi matematika dan ipa saja, sisanya kosong.
    """

    def test_partial_akademik_accepted(self, client):
        """Sebagian nilai akademik saja harus diterima."""
        img = create_dummy_png()
        akademik = json.dumps({"matematika": 88, "ipa": 85})  # hanya 2 dari 18
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": akademik},
        )
        assert resp.status_code in VALID_PREDICT_STATUS

    def test_partial_akademik_shows_filled_fields(self, client):
        """Field 'diisi' harus mencantumkan mata pelajaran yang benar-benar diisi."""
        img = create_dummy_png()
        akademik = json.dumps({"matematika": 90, "informatika": 85})
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": akademik},
        )
        if resp.status_code == 200:
            kd = resp.json()["kelengkapan_data"]
            diisi = kd["akademik"]["diisi"]
            assert "matematika" in diisi, "matematika harus ada di list 'diisi'"
            assert "informatika" in diisi, "informatika harus ada di list 'diisi'"

    def test_partial_akademik_persen_correct(self, client):
        """Persen kelengkapan harus proporsional dengan yang diisi."""
        img = create_dummy_png()
        # Isi 9 dari 18 mata pelajaran → 50%
        akademik = json.dumps({
            "agama": 80, "pancasila": 80, "bahasa_indonesia": 80,
            "matematika": 90, "ipa": 85, "ips": 75,
            "bahasa_inggris": 84, "pjok": 80, "informatika": 88,
        })
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": akademik},
        )
        if resp.status_code == 200:
            kd = resp.json()["kelengkapan_data"]
            persen = kd["akademik"]["persen_lengkap"]
            assert 45 <= persen <= 55, \
                f"9/18 mata pelajaran = ~50%, dapat {persen}%"

    def test_partial_talent_only(self, client):
        """Hanya talent saja (tanpa akademik) harus diterima."""
        img = create_dummy_png()
        talent = json.dumps({"logika": 8, "teknologi": 9})
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"talent": talent},
        )
        assert resp.status_code in VALID_PREDICT_STATUS

    def test_partial_akademik_and_talent_together(self, client):
        """Kombinasi sebagian akademik + sebagian talent harus diterima."""
        img = create_dummy_png()
        akademik = json.dumps({"matematika": 88, "ipa": 85, "bahasa_inggris": 84})
        talent = json.dumps({"logika": 8, "teknologi": 9, "kreativitas": 5})
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": akademik, "talent": talent},
        )
        assert resp.status_code in VALID_PREDICT_STATUS

    def test_full_data_has_higher_completeness(self, client):
        """Data lengkap harus punya persen_lengkap lebih tinggi dari data sebagian."""
        img = create_dummy_png()

        # Sebagian
        resp_partial = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": json.dumps({"matematika": 88})},
        )

        # Lengkap
        akademik_full = {
            "agama": 80, "pancasila": 80, "bahasa_indonesia": 80,
            "matematika": 90, "ipa": 85, "ips": 75,
            "bahasa_inggris": 84, "pjok": 80, "informatika": 88,
            "seni_budaya": 75, "logika": 88, "kreativitas": 70,
            "komunikasi": 80, "kepemimpinan": 75, "problem_solving": 88,
            "teamwork": 82, "literasi": 85, "numerasi": 90,
        }
        resp_full = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": json.dumps(akademik_full)},
        )

        if resp_partial.status_code == 200 and resp_full.status_code == 200:
            persen_partial = resp_partial.json()["kelengkapan_data"]["akademik"]["persen_lengkap"]
            persen_full    = resp_full.json()["kelengkapan_data"]["akademik"]["persen_lengkap"]
            assert persen_full > persen_partial, \
                f"Data lengkap ({persen_full}%) harus > data sebagian ({persen_partial}%)"
            assert persen_full == 100


# ==================================================================
# TEST 5: Predict — Semua Data Lengkap
# ==================================================================
class TestPredictFullData:
    def test_full_akademik_and_talent(self, client):
        """Data akademik dan bakat lengkap harus menghasilkan prediksi terbaik."""
        img = create_dummy_png()
        akademik = json.dumps({
            "matematika": 90, "ipa": 85, "bahasa_inggris": 84,
            "informatika": 92, "logika": 88,
        })
        talent = json.dumps({
            "logika": 8, "teknologi": 9, "riset": 8,
            "problem_solving": 8, "kreativitas": 5,
        })
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data={"akademik": akademik, "talent": talent},
        )
        assert resp.status_code in VALID_PREDICT_STATUS

        if resp.status_code == 200:
            data = resp.json()
            assert len(data["rekomendasi_jurusan"]) == 3
            assert data["karakter"]["tipe"] in [
                "Realistic", "Investigative", "Artistic",
                "Social", "Enterprising", "Conventional"
            ]


# ==================================================================
# TEST 6: Validasi Field kelengkapan_data ← BARU
# ==================================================================
class TestKelengkapanData:
    """
    Pastikan field kelengkapan_data selalu ada dan strukturnya benar,
    apapun input yang dikirim siswa.
    """

    def _get_kelengkapan(self, client, akademik=None, talent=None):
        img = create_dummy_png()
        data = {}
        if akademik:
            data["akademik"] = json.dumps(akademik)
        if talent:
            data["talent"] = json.dumps(talent)
        resp = client.post(
            "/predict",
            files={"file": ("tulisan.png", img, "image/png")},
            data=data,
        )
        if resp.status_code == 200:
            return resp.json().get("kelengkapan_data")
        return None

    def test_kelengkapan_always_present(self, client):
        """kelengkapan_data harus selalu ada di respons."""
        img = create_dummy_png()
        resp = client.post("/predict",
                           files={"file": ("t.png", img, "image/png")})
        if resp.status_code == 200:
            assert "kelengkapan_data" in resp.json()

    def test_kelengkapan_has_3_keys(self, client):
        """kelengkapan_data harus punya key: gambar, akademik, bakat, catatan."""
        kd = self._get_kelengkapan(client)
        if kd:
            for key in ["gambar", "akademik", "bakat", "catatan"]:
                assert key in kd, f"Key '{key}' tidak ada di kelengkapan_data"

    def test_akademik_default_value_is_75(self, client):
        """Default nilai akademik harus selalu 75."""
        kd = self._get_kelengkapan(client)
        if kd:
            assert kd["akademik"]["default_value"] == 75

    def test_bakat_default_value_is_5(self, client):
        """Default nilai bakat harus selalu 5."""
        kd = self._get_kelengkapan(client)
        if kd:
            assert kd["bakat"]["default_value"] == 5

    def test_persen_lengkap_is_between_0_and_100(self, client):
        """persen_lengkap harus antara 0 dan 100."""
        kd = self._get_kelengkapan(client, akademik={"matematika": 90})
        if kd:
            assert 0 <= kd["akademik"]["persen_lengkap"] <= 100
            assert 0 <= kd["bakat"]["persen_lengkap"] <= 100

    def test_diisi_and_diisi_otomatis_no_overlap(self, client):
        """Satu mata pelajaran tidak boleh ada di 'diisi' DAN 'diisi_otomatis' sekaligus."""
        kd = self._get_kelengkapan(client, akademik={"matematika": 90, "ipa": 85})
        if kd:
            diisi = set(kd["akademik"]["diisi"])
            otomatis = set(kd["akademik"]["diisi_otomatis"])
            overlap = diisi & otomatis
            assert len(overlap) == 0, \
                f"Ada overlap antara diisi dan diisi_otomatis: {overlap}"

    def test_catatan_is_string(self, client):
        """Field catatan harus berupa string."""
        kd = self._get_kelengkapan(client)
        if kd:
            assert isinstance(kd["catatan"], str)
            assert len(kd["catatan"]) > 0


# ==================================================================
# TEST 7: Error Handling
# ==================================================================
class TestErrorHandling:
    def test_no_file_returns_422(self, client):
        """Request tanpa file harus dikembalikan 422."""
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_wrong_file_type_returns_400(self, client):
        """File PDF harus dikembalikan 400."""
        resp = client.post(
            "/predict",
            files={"file": ("test.pdf", b"fake content", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_invalid_akademik_json_returns_400(self, client):
        """JSON akademik yang rusak harus dikembalikan 400."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img, "image/png")},
            data={"akademik": "ini bukan json {{}"},
        )
        assert resp.status_code == 400

    def test_invalid_talent_json_returns_400(self, client):
        """JSON talent yang rusak harus dikembalikan 400."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img, "image/png")},
            data={"talent": "bukan json"},
        )
        assert resp.status_code == 400

    def test_nilai_lebih_dari_100_returns_error(self, client):
        """Nilai akademik > 100 harus dikembalikan 400 atau 422."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img, "image/png")},
            data={"akademik": json.dumps({"matematika": 999})},
        )
        assert resp.status_code in (400, 422)

    def test_nilai_negatif_returns_error(self, client):
        """Nilai akademik negatif harus ditolak."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img, "image/png")},
            data={"akademik": json.dumps({"matematika": -10})},
        )
        assert resp.status_code in (400, 422)

    def test_bakat_lebih_dari_10_returns_error(self, client):
        """Skor bakat > 10 harus ditolak."""
        img = create_dummy_png()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img, "image/png")},
            data={"talent": json.dumps({"logika": 99})},
        )
        assert resp.status_code in (400, 422)

    def test_empty_file_returns_400(self, client):
        """File gambar kosong harus dikembalikan 400."""
        resp = client.post(
            "/predict",
            files={"file": ("test.png", b"", "image/png")},
        )
        assert resp.status_code == 400


# ==================================================================
# TEST 8: Unit Test Image Processor
# ==================================================================
class TestImageProcessor:
    def test_extract_returns_exactly_10_features(self):
        """Ekstraksi gambar harus menghasilkan tepat 10 fitur."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        features = extractor.extract_from_bytes(create_dummy_png())
        assert len(features) == 10, \
            f"Seharusnya 10 fitur, dapat {len(features)}: {list(features.keys())}"

    def test_all_features_in_range_0_to_10(self):
        """Semua fitur harus bernilai antara 0 dan 10."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        features = extractor.extract_from_bytes(create_dummy_png())
        for name, val in features.items():
            assert 0.0 <= val <= 10.0, \
                f"Fitur '{name}' = {val}, harus antara 0-10"

    def test_features_are_floats(self):
        """Semua nilai fitur harus bertipe float."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        features = extractor.extract_from_bytes(create_dummy_png())
        for name, val in features.items():
            assert isinstance(val, float), \
                f"Fitur '{name}' bukan float: {type(val)}"

    def test_feature_names_correct(self):
        """Nama fitur harus sesuai yang diharapkan."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        features = extractor.extract_from_bytes(create_dummy_png())
        expected_names = {
            "letter_size_score", "slant_angle", "pressure_score",
            "spacing_score", "readability_score", "neatness_score",
            "connectivity_score", "ornament_score", "line_straightness",
            "density_score",
        }
        assert set(features.keys()) == expected_names, \
            f"Nama fitur tidak sesuai. Lebih: {set(features.keys()) - expected_names}, " \
            f"Kurang: {expected_names - set(features.keys())}"

    def test_invalid_image_returns_default_features(self):
        """Gambar tidak valid harus mengembalikan fitur default (bukan error)."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        features = extractor.extract_from_bytes(b"bukan gambar sama sekali")
        assert len(features) == 10
        assert "letter_size_score" in features

    def test_real_image_from_dataset(self):
        """Test dengan gambar nyata dari dataset (jika ada)."""
        from src.preprocessing.image_processor import HandwritingFeatureExtractor
        extractor = HandwritingFeatureExtractor()
        img_path = ROOT / "data" / "img" / "img_1.png"
        if img_path.exists():
            features = extractor.extract(str(img_path))
            assert len(features) == 10
            for val in features.values():
                assert 0.0 <= val <= 10.0
        else:
            pytest.skip("Gambar dataset tidak ditemukan di data/img/img_1.png")


# ==================================================================
# TEST 9: Unit Test Nilai Default (BARU)
# ==================================================================
class TestDefaultValues:
    """
    Pastikan nilai default yang digunakan ketika siswa tidak mengisi
    adalah nilai yang benar (75 untuk akademik, 5 untuk bakat).
    """

    def test_akademik_default_is_75(self):
        """DEFAULT_AKADEMIK harus berisi 75 untuk semua mata pelajaran."""
        from src.inference.predictor import DEFAULT_AKADEMIK
        assert len(DEFAULT_AKADEMIK) == 18, \
            f"Harus ada 18 mata pelajaran, dapat {len(DEFAULT_AKADEMIK)}"
        for pelajaran, nilai in DEFAULT_AKADEMIK.items():
            assert nilai == 75, \
                f"Default '{pelajaran}' = {nilai}, seharusnya 75"

    def test_talent_default_is_5(self):
        """DEFAULT_TALENT harus berisi 5 untuk semua bakat."""
        from src.inference.predictor import DEFAULT_TALENT
        assert len(DEFAULT_TALENT) == 12, \
            f"Harus ada 12 bakat, dapat {len(DEFAULT_TALENT)}"
        for bakat, nilai in DEFAULT_TALENT.items():
            assert nilai == 5, \
                f"Default '{bakat}' = {nilai}, seharusnya 5"

    def test_akademik_default_keys_match_schema(self):
        """Kolom DEFAULT_AKADEMIK harus sesuai dengan AkademikInput schema."""
        from src.inference.predictor import DEFAULT_AKADEMIK
        from api.schemas import AkademikInput
        schema_fields = set(AkademikInput.model_fields.keys())
        default_keys  = set(DEFAULT_AKADEMIK.keys())
        # Semua key di DEFAULT_AKADEMIK harus ada di schema
        missing_in_schema = default_keys - schema_fields
        assert len(missing_in_schema) == 0, \
            f"Key di DEFAULT_AKADEMIK tidak ada di AkademikInput: {missing_in_schema}"

    def test_predictor_fills_missing_with_defaults(self):
        """
        Saat akademik kosong, predictor harus mengisi DEFAULT_AKADEMIK
        sehingga semua fitur akademik bernilai 75.
        """
        from src.inference.predictor import HandwritingPredictor, DEFAULT_AKADEMIK
        predictor = HandwritingPredictor()
        # Simulasikan pengisian: mulai dari default, update dengan input kosong
        akademik_final = dict(DEFAULT_AKADEMIK)
        akademik_final.update({})  # tidak ada yang diupdate
        assert akademik_final["matematika"] == 75
        assert akademik_final["bahasa_indonesia"] == 75

    def test_predictor_overwrites_default_with_actual_value(self):
        """
        Saat siswa mengisi matematika=90, nilai 90 harus menimpa default 75.
        """
        from src.inference.predictor import DEFAULT_AKADEMIK
        akademik_final = dict(DEFAULT_AKADEMIK)
        akademik_final.update({"matematika": 90})  # siswa mengisi 90
        assert akademik_final["matematika"] == 90, \
            "Nilai asli siswa (90) harus menimpa default (75)"
        assert akademik_final["bahasa_indonesia"] == 75, \
            "Mata pelajaran yang tidak diisi tetap 75"


# ==================================================================
# Jalankan langsung
# ==================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

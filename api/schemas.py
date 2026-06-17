"""
=============================================================
MODUL: schemas.py
=============================================================
TUJUAN:
  Mendefinisikan struktur data (Pydantic models) untuk
  request dan response API.

  Pydantic otomatis memvalidasi tipe data, sehingga jika
  web dev mengirim nilai yang salah, API langsung memberi
  pesan error yang jelas.
=============================================================
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ------------------------------------------------------------------
# REQUEST: Data Akademik (nilai pelajaran)
# ------------------------------------------------------------------
class AkademikInput(BaseModel):
    """
    Nilai rapor akademik siswa (semua opsional, default = rata-rata).
    Nilai range: 0–100
    """
    agama:             Optional[float] = Field(None, ge=0, le=100, description="Nilai Agama")
    pancasila:         Optional[float] = Field(None, ge=0, le=100, description="Nilai Pancasila/PKN")
    bahasa_indonesia:  Optional[float] = Field(None, ge=0, le=100, description="Nilai Bahasa Indonesia")
    matematika:        Optional[float] = Field(None, ge=0, le=100, description="Nilai Matematika")
    ipa:               Optional[float] = Field(None, ge=0, le=100, description="Nilai IPA")
    ips:               Optional[float] = Field(None, ge=0, le=100, description="Nilai IPS")
    bahasa_inggris:    Optional[float] = Field(None, ge=0, le=100, description="Nilai Bahasa Inggris")
    pjok:              Optional[float] = Field(None, ge=0, le=100, description="Nilai PJOK")
    informatika:       Optional[float] = Field(None, ge=0, le=100, description="Nilai Informatika")
    seni_budaya:       Optional[float] = Field(None, ge=0, le=100, description="Nilai Seni Budaya")
    logika:            Optional[float] = Field(None, ge=0, le=100, description="Skor Logika")
    kreativitas:       Optional[float] = Field(None, ge=0, le=100, description="Skor Kreativitas")
    komunikasi:        Optional[float] = Field(None, ge=0, le=100, description="Skor Komunikasi")
    kepemimpinan:      Optional[float] = Field(None, ge=0, le=100, description="Skor Kepemimpinan")
    problem_solving:   Optional[float] = Field(None, ge=0, le=100, description="Skor Problem Solving")
    teamwork:          Optional[float] = Field(None, ge=0, le=100, description="Skor Teamwork")
    literasi:          Optional[float] = Field(None, ge=0, le=100, description="Skor Literasi")
    numerasi:          Optional[float] = Field(None, ge=0, le=100, description="Skor Numerasi")

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    class Config:
        json_schema_extra = {
            "example": {
                "matematika": 88,
                "ipa": 85,
                "bahasa_inggris": 84,
                "informatika": 90,
                "logika": 88,
                "kreativitas": 70,
            }
        }


# ------------------------------------------------------------------
# REQUEST: Data Bakat
# ------------------------------------------------------------------
class TalentInput(BaseModel):
    """
    Skor bakat siswa (semua opsional).
    Nilai range: 1–10
    """
    komunikasi:     Optional[float] = Field(None, ge=1, le=10, description="Kemampuan komunikasi (1-10)")
    kepemimpinan:   Optional[float] = Field(None, ge=1, le=10, description="Kemampuan memimpin (1-10)")
    kreativitas:    Optional[float] = Field(None, ge=1, le=10, description="Tingkat kreativitas (1-10)")
    logika:         Optional[float] = Field(None, ge=1, le=10, description="Kemampuan logika (1-10)")
    teknologi:      Optional[float] = Field(None, ge=1, le=10, description="Minat & kemampuan teknologi (1-10)")
    riset:          Optional[float] = Field(None, ge=1, le=10, description="Minat penelitian/riset (1-10)")
    seni:           Optional[float] = Field(None, ge=1, le=10, description="Minat & bakat seni (1-10)")
    olahraga:       Optional[float] = Field(None, ge=1, le=10, description="Minat olahraga (1-10)")
    organisasi:     Optional[float] = Field(None, ge=1, le=10, description="Kemampuan berorganisasi (1-10)")
    kewirausahaan:  Optional[float] = Field(None, ge=1, le=10, description="Minat kewirausahaan (1-10)")
    kerja_tim:      Optional[float] = Field(None, ge=1, le=10, description="Kemampuan kerja tim (1-10)")
    problem_solving: Optional[float] = Field(None, ge=1, le=10, description="Kemampuan pemecahan masalah (1-10)")

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    class Config:
        json_schema_extra = {
            "example": {
                "logika": 8,
                "teknologi": 9,
                "riset": 8,
                "problem_solving": 8,
                "kreativitas": 5,
            }
        }


# ------------------------------------------------------------------
# RESPONSE: Karakter Siswa
# ------------------------------------------------------------------
class KarakterResponse(BaseModel):
    """Hasil analisis karakter/kepribadian RIASEC."""
    tipe: str = Field(description="Tipe RIASEC dominan: Realistic/Investigative/Artistic/Social/Enterprising/Conventional")
    nama: str = Field(description="Nama karakter dalam bahasa Indonesia")
    deskripsi: str = Field(description="Deskripsi lengkap karakter siswa")
    kekuatan: List[str] = Field(description="3 kekuatan utama siswa")
    warna: str = Field(description="Kode warna untuk UI (hex)")


# ------------------------------------------------------------------
# RESPONSE: Rekomendasi Jurusan
# ------------------------------------------------------------------
class JurusanResponse(BaseModel):
    """Satu jurusan yang direkomendasikan."""
    rank: int = Field(description="Peringkat rekomendasi (1 = terbaik)")
    jurusan: str = Field(description="Nama jurusan kuliah")
    match_score: float = Field(description="Persentase kecocokan (0–100)")
    alasan: str = Field(description="Alasan mengapa jurusan ini cocok untuk siswa")


# ------------------------------------------------------------------
# RESPONSE: Fitur Tulisan Tangan
# ------------------------------------------------------------------
class FiturTulisanResponse(BaseModel):
    """Fitur numerik yang diekstrak dari gambar tulisan tangan."""
    letter_size_score:  float = Field(description="Ukuran huruf (1-10, besar=tinggi)")
    slant_angle:        float = Field(description="Kemiringan tulisan (1=kiri, 5=tegak, 10=kanan)")
    pressure_score:     float = Field(description="Tekanan pena (1=ringan, 10=kuat)")
    spacing_score:      float = Field(description="Jarak antar huruf (1=rapat, 10=renggang)")
    readability_score:  float = Field(description="Keterbacaan (1=susah dibaca, 10=sangat jelas)")
    neatness_score:     float = Field(description="Kerapian (1=berantakan, 10=sangat rapi)")
    connectivity_score: float = Field(description="Konektivitas huruf (1=cetak, 10=cursive penuh)")
    ornament_score:     float = Field(description="Hiasan/ornamen (1=polos, 10=banyak hiasan)")
    line_straightness:  float = Field(description="Kelurusan baris (1=naik-turun, 10=lurus)")
    density_score:      float = Field(description="Kepadatan tinta (1=renggang, 10=padat)")


# ------------------------------------------------------------------
# RESPONSE: Feature Importance
# ------------------------------------------------------------------
class FeatureImportanceItem(BaseModel):
    fitur: str = Field(description="Nama fitur/variabel")
    importance: float = Field(description="Tingkat pengaruh (0–1)")


# ------------------------------------------------------------------
# RESPONSE UTAMA: Hasil Prediksi Lengkap
# ------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """
    Respons lengkap API prediksi tulisan tangan.
    Ini adalah yang diterima oleh web developer Laravel.
    """
    status: str = Field(default="success", description="Status: success / error")
    karakter: KarakterResponse = Field(description="Profil karakter/kepribadian siswa")
    riasec_skor: Dict[str, float] = Field(description="Skor probabilitas tiap tipe RIASEC")
    rekomendasi_jurusan: List[JurusanResponse] = Field(description="Top-3 jurusan yang direkomendasikan")
    fitur_tulisan: FiturTulisanResponse = Field(description="Fitur yang diekstrak dari gambar tulisan")
    feature_importance: List[FeatureImportanceItem] = Field(description="Fitur paling berpengaruh pada prediksi")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "karakter": {
                    "tipe": "Investigative",
                    "nama": "Analitis & Ilmiah",
                    "deskripsi": "Kamu adalah pemikir mendalam yang suka menganalisis...",
                    "kekuatan": ["Kemampuan analisis kuat", "Berpikir logis", "Rasa ingin tahu tinggi"],
                    "warna": "#2980B9",
                },
                "riasec_skor": {
                    "Investigative": 0.45,
                    "Conventional": 0.20,
                    "Realistic": 0.15,
                    "Artistic": 0.08,
                    "Social": 0.07,
                    "Enterprising": 0.05,
                },
                "rekomendasi_jurusan": [
                    {"rank": 1, "jurusan": "Informatika", "match_score": 87.3, "alasan": "..."},
                    {"rank": 2, "jurusan": "Statistik", "match_score": 72.1, "alasan": "..."},
                    {"rank": 3, "jurusan": "Sistem Informasi", "match_score": 65.8, "alasan": "..."},
                ],
                "fitur_tulisan": {
                    "letter_size_score": 4.2,
                    "slant_angle": 5.1,
                    "pressure_score": 6.3,
                    "spacing_score": 4.8,
                    "readability_score": 7.2,
                    "neatness_score": 8.1,
                    "connectivity_score": 3.4,
                    "ornament_score": 1.2,
                    "line_straightness": 8.5,
                    "density_score": 5.0,
                },
                "feature_importance": [
                    {"fitur": "conventional", "importance": 0.1832},
                    {"fitur": "investigative", "importance": 0.1654},
                ],
            }
        }


# ------------------------------------------------------------------
# RESPONSE: Error
# ------------------------------------------------------------------
class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None
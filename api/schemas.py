"""
=============================================================
MODUL: schemas.py  (Dataset Real v3 — RIASEC Direct)
=============================================================
Pydantic schemas untuk request dan response FastAPI.

REQUEST  : AkademikInput (multipart form + JSON body)
RESPONSE : PredictResponse

Perubahan dari v2:
  - Hapus BigFiveInfo dan TalentInput
  - Hapus field big_five dari PredictResponse
  - Tambah PerbandinganInfo (status SEJALAN/BERBEDA + penjelasan)
  - profil_karakter sekarang mengacu ke RiasecInfo langsung
=============================================================
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ===========================================================
# REQUEST
# ===========================================================

class AkademikInput(BaseModel):
    """
    Nilai akademik 14 mata pelajaran (Semester 4 dan 5).
    Semua opsional — nilai 0 / None dianggap 'tidak ambil pelajaran ini'.
    """
    # Semester 4
    mat_s4:  Optional[float] = Field(None, ge=0, le=100, description="Matematika Semester 4")
    fis_s4:  Optional[float] = Field(None, ge=0, le=100, description="Fisika Semester 4")
    kim_s4:  Optional[float] = Field(None, ge=0, le=100, description="Kimia Semester 4")
    bio_s4:  Optional[float] = Field(None, ge=0, le=100, description="Biologi Semester 4")
    bind_s4: Optional[float] = Field(None, ge=0, le=100, description="Bahasa Indonesia Semester 4")
    bing_s4: Optional[float] = Field(None, ge=0, le=100, description="Bahasa Inggris Semester 4")
    info_s4: Optional[float] = Field(None, ge=0, le=100, description="Informatika Semester 4")

    # Semester 5
    mat_s5:  Optional[float] = Field(None, ge=0, le=100, description="Matematika Semester 5")
    fis_s5:  Optional[float] = Field(None, ge=0, le=100, description="Fisika Semester 5")
    kim_s5:  Optional[float] = Field(None, ge=0, le=100, description="Kimia Semester 5")
    bio_s5:  Optional[float] = Field(None, ge=0, le=100, description="Biologi Semester 5")
    bind_s5: Optional[float] = Field(None, ge=0, le=100, description="Bahasa Indonesia Semester 5")
    bing_s5: Optional[float] = Field(None, ge=0, le=100, description="Bahasa Inggris Semester 5")
    info_s5: Optional[float] = Field(None, ge=0, le=100, description="Informatika Semester 5")

    def to_dict(self) -> Dict[str, float]:
        d = self.model_dump()
        return {k: (v if v is not None else 0.0) for k, v in d.items()}


# ===========================================================
# RESPONSE
# ===========================================================

class FiturTulisan(BaseModel):
    """10 fitur numerik hasil ekstraksi OpenCV dari gambar tulisan."""
    letter_size:  float = Field(description="Rata-rata ukuran huruf (1-10)")
    slant:        float = Field(description="Kemiringan tulisan (1=kiri, 5=tegak, 10=kanan)")
    pressure:     float = Field(description="Tekanan pena / kegelapan (1-10)")
    spacing:      float = Field(description="Jarak antar huruf/kata (1-10)")
    readability:  float = Field(description="Keterbacaan tulisan (1-10)")
    neatness:     float = Field(description="Kerapian / konsistensi baseline (1-10)")
    connectivity: float = Field(description="Sambungan antar huruf / cursive (1-10)")
    ornament:     float = Field(description="Dekorasi / hiasan tambahan (1-10)")
    baseline:     float = Field(description="Kelurusan baris tulisan (1-10)")
    density:      float = Field(description="Kepadatan tinta di halaman (1-10)")


class RiasecInfo(BaseModel):
    """Tipe karakter RIASEC Holland yang diprediksi dari tulisan tangan."""
    dominant:  str             = Field(description="Tipe RIASEC dominan")
    karakter:  str             = Field(description="Label karakter singkat")
    deskripsi: str             = Field(description="Deskripsi lengkap kepribadian")
    kekuatan:  List[str]       = Field(description="3 kekuatan utama tipe ini")
    warna:     str             = Field(description="Warna hex representasi tipe")
    skor:      Dict[str, float] = Field(description="Skor RIASEC per tipe (0-100)")


class AnalisisAkademik(BaseModel):
    """Hasil analisis nilai akademik."""
    rumpun_ilmu:         str             = Field(description="Rumpun Ilmu yang diprediksi")
    rumpun_probabilitas: Dict[str, float] = Field(description="Probabilitas per Rumpun Ilmu (%)")
    nilai_rata_rata:     float            = Field(description="Rata-rata nilai akademik (hanya pelajaran yang diambil)")
    mata_pelajaran_kuat: List[Dict]       = Field(description="3 mata pelajaran tertinggi (nilai > 0)")


class PerbandinganInfo(BaseModel):
    """Perbandingan antara tipe RIASEC dan Rumpun Ilmu dari nilai akademik."""
    status:     str = Field(description="'SEJALAN' jika RIASEC cocok Rumpun Ilmu, 'BERBEDA' jika tidak")
    penjelasan: str = Field(description="Penjelasan lengkap hasil perbandingan")


class RekomendasiJurusan(BaseModel):
    """Satu rekomendasi program studi."""
    program_studi:   str   = Field(description="Nama program studi")
    rumpun_ilmu:     str   = Field(description="Rumpun ilmu program studi ini")
    alasan:          str   = Field(description="Penjelasan mengapa cocok untuk siswa ini")
    skor_kesesuaian: float = Field(description="Skor kesesuaian keseluruhan (1.0-5.0)")
    prediksi_ipk:    str   = Field(description="Estimasi rentang IPK awal (misal '3.2 – 3.6')")


class PredictResponse(BaseModel):
    """
    Response lengkap API prediksi tulisan tangan v3.

    Struktur:
      riasec_karakter      → tipe RIASEC langsung dari tulisan tangan
      analisis_akademik    → Rumpun Ilmu + nilai akademik
      perbandingan         → apakah RIASEC sejalan/berbeda dengan Rumpun Ilmu
      rekomendasi_jurusan  → TOP-3 (sejalan) atau TOP-5 (berbeda) Program Studi
      fitur_tulisan        → 10 fitur numerik dari gambar
      kelengkapan_data     → info field mana yang diisi/default
    """
    riasec_karakter:     RiasecInfo              = Field(description="Profil karakter RIASEC siswa")
    analisis_akademik:   AnalisisAkademik         = Field(description="Analisis nilai akademik")
    perbandingan:        PerbandinganInfo          = Field(description="Perbandingan RIASEC vs Rumpun Ilmu")
    rekomendasi_jurusan: List[RekomendasiJurusan] = Field(description="TOP-3/5 jurusan yang direkomendasikan")
    fitur_tulisan:       FiturTulisan             = Field(description="10 fitur tulisan tangan (OpenCV)")
    kelengkapan_data:    Dict[str, str]           = Field(description="Status kelengkapan data input")


class HealthResponse(BaseModel):
    status:       str  = Field(description="'ok' jika server berjalan normal")
    model_loaded: bool = Field(description="True jika semua model sudah dimuat")
    version:      str  = Field(default="3.0.0")


# ===========================================================
# Helper untuk konversi predictor output → PredictResponse
# ===========================================================

def build_predict_response(raw: dict) -> PredictResponse:
    """Konversi output dict predictor.predict() ke PredictResponse."""
    rk  = raw["riasec_karakter"]
    aa  = raw["analisis_akademik"]
    per = raw["perbandingan"]
    ft  = raw["fitur_tulisan"]
    rek = raw["rekomendasi_jurusan"]
    kd  = raw["kelengkapan_data"]

    return PredictResponse(
        riasec_karakter=RiasecInfo(
            dominant  = rk["dominant"],
            karakter  = rk["karakter"],
            deskripsi = rk["deskripsi"],
            kekuatan  = rk["kekuatan"],
            warna     = rk["warna"],
            skor      = rk["skor"],
        ),
        analisis_akademik=AnalisisAkademik(
            rumpun_ilmu         = aa["rumpun_ilmu"],
            rumpun_probabilitas = aa["rumpun_probabilitas"],
            nilai_rata_rata     = aa["nilai_rata_rata"],
            mata_pelajaran_kuat = aa["mata_pelajaran_kuat"],
        ),
        perbandingan=PerbandinganInfo(
            status     = per["status"],
            penjelasan = per["penjelasan"],
        ),
        rekomendasi_jurusan=[
            RekomendasiJurusan(**j) for j in rek
        ],
        fitur_tulisan=FiturTulisan(**ft),
        kelengkapan_data=kd,
    )
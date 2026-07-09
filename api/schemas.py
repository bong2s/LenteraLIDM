"""
=============================================================
MODUL: schemas.py  (Dataset Real v3 — RIASEC Direct + Minat)
=============================================================
Pydantic schemas untuk request dan response FastAPI.

REQUEST  : AkademikInput, MinatInput
RESPONSE : PredictResponse

Struktur response v3.1 (+ minat kuesioner):
  riasec_karakter      → RIASEC dari tulisan tangan
  riasec_minat         → RIASEC dari kuesioner 24 soal (opsional)
  analisis_akademik    → Rumpun Ilmu + nilai akademik
  perbandingan_akademik→ RIASEC tulisan vs Rumpun Ilmu
  perbandingan_minat   → RIASEC tulisan vs RIASEC minat (jika minat dikirim)
  rekomendasi_jurusan  → TOP-3 (semua sejalan) atau TOP-5 (ada yang berbeda)
  fitur_tulisan        → 10 fitur numerik dari gambar
  kelengkapan_data     → status field mana yang diisi/default
=============================================================
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ===========================================================
# REQUEST
# ===========================================================

class AkademikInput(BaseModel):
    """
    Nilai akademik 7 mata pelajaran.
    Nilai yang diinput adalah RATA-RATA dari Semester 4 dan 5 (dihitung oleh
    pengguna/frontend sebelum dikirim ke API).
    Semua opsional — nilai 0 / None dianggap 'tidak ambil pelajaran ini'.
    """
    mat:  Optional[float] = Field(None, ge=0, le=100, description="Matematika (rata-rata Smt 4 & 5)")
    fis:  Optional[float] = Field(None, ge=0, le=100, description="Fisika (rata-rata Smt 4 & 5)")
    kim:  Optional[float] = Field(None, ge=0, le=100, description="Kimia (rata-rata Smt 4 & 5)")
    bio:  Optional[float] = Field(None, ge=0, le=100, description="Biologi (rata-rata Smt 4 & 5)")
    bind: Optional[float] = Field(None, ge=0, le=100, description="Bahasa Indonesia (rata-rata Smt 4 & 5)")
    bing: Optional[float] = Field(None, ge=0, le=100, description="Bahasa Inggris (rata-rata Smt 4 & 5)")
    info: Optional[float] = Field(None, ge=0, le=100, description="Informatika (rata-rata Smt 4 & 5)")

    def to_dict(self) -> Dict[str, float]:
        d = self.model_dump()
        return {k: (v if v is not None else 0.0) for k, v in d.items()}


class MinatInput(BaseModel):
    """
    Jawaban kuesioner minat RIASEC (24 soal, masing-masing skala 0-4).

    Format A — 24 jawaban mentah (tiap tipe RIASEC punya 4 soal):
      q_R1..q_R4  : Realistic
      q_I1..q_I4  : Investigative
      q_A1..q_A4  : Artistic
      q_S1..q_S4  : Social
      q_E1..q_E4  : Enterprising
      q_C1..q_C4  : Conventional

    Format B — 6 skor sudah dijumlah dari web (score_R..score_C, max 16):
      score_R, score_I, score_A, score_S, score_E, score_C

    Kirim salah satu format saja. Format A diprioritaskan jika keduanya ada.
    """
    # Format A — jawaban raw
    q_R1: Optional[float] = Field(None, ge=0, le=4)
    q_R2: Optional[float] = Field(None, ge=0, le=4)
    q_R3: Optional[float] = Field(None, ge=0, le=4)
    q_R4: Optional[float] = Field(None, ge=0, le=4)
    q_I1: Optional[float] = Field(None, ge=0, le=4)
    q_I2: Optional[float] = Field(None, ge=0, le=4)
    q_I3: Optional[float] = Field(None, ge=0, le=4)
    q_I4: Optional[float] = Field(None, ge=0, le=4)
    q_A1: Optional[float] = Field(None, ge=0, le=4)
    q_A2: Optional[float] = Field(None, ge=0, le=4)
    q_A3: Optional[float] = Field(None, ge=0, le=4)
    q_A4: Optional[float] = Field(None, ge=0, le=4)
    q_S1: Optional[float] = Field(None, ge=0, le=4)
    q_S2: Optional[float] = Field(None, ge=0, le=4)
    q_S3: Optional[float] = Field(None, ge=0, le=4)
    q_S4: Optional[float] = Field(None, ge=0, le=4)
    q_E1: Optional[float] = Field(None, ge=0, le=4)
    q_E2: Optional[float] = Field(None, ge=0, le=4)
    q_E3: Optional[float] = Field(None, ge=0, le=4)
    q_E4: Optional[float] = Field(None, ge=0, le=4)
    q_C1: Optional[float] = Field(None, ge=0, le=4)
    q_C2: Optional[float] = Field(None, ge=0, le=4)
    q_C3: Optional[float] = Field(None, ge=0, le=4)
    q_C4: Optional[float] = Field(None, ge=0, le=4)

    # Format B — skor sudah dijumlah
    score_R: Optional[float] = Field(None, ge=0, le=16)
    score_I: Optional[float] = Field(None, ge=0, le=16)
    score_A: Optional[float] = Field(None, ge=0, le=16)
    score_S: Optional[float] = Field(None, ge=0, le=16)
    score_E: Optional[float] = Field(None, ge=0, le=16)
    score_C: Optional[float] = Field(None, ge=0, le=16)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


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
    dominant:  str              = Field(description="Tipe RIASEC dominan")
    karakter:  str              = Field(description="Label karakter singkat")
    deskripsi: str              = Field(description="Deskripsi lengkap kepribadian")
    kekuatan:  List[str]        = Field(description="3 kekuatan utama tipe ini")
    warna:     str              = Field(description="Warna hex representasi tipe")
    skor:      Dict[str, float] = Field(description="Probabilitas RIASEC per tipe (%)")


class MinatKarakter(BaseModel):
    """Tipe RIASEC dari hasil kuesioner minat (24 soal)."""
    dominant:    str              = Field(description="Tipe RIASEC dominan dari kuesioner")
    karakter:    str              = Field(description="Label karakter singkat")
    deskripsi:   str              = Field(description="Deskripsi kepribadian")
    kekuatan:    List[str]        = Field(description="3 kekuatan utama tipe ini")
    warna:       str              = Field(description="Warna hex representasi tipe")
    skor_raw:    Dict[str, float] = Field(description="Skor mentah per tipe RIASEC (maks 16)")
    skor_persen: Dict[str, float] = Field(description="Skor dalam persen (total = 100%)")


class AnalisisAkademik(BaseModel):
    """Hasil analisis nilai akademik."""
    rumpun_ilmu:         str              = Field(description="Rumpun Ilmu yang diprediksi")
    rumpun_probabilitas: Dict[str, float] = Field(description="Probabilitas per Rumpun Ilmu (%)")
    nilai_rata_rata:     float            = Field(description="Rata-rata nilai akademik (hanya pelajaran yang diambil)")
    mata_pelajaran_kuat: List[Dict]       = Field(description="3 mata pelajaran tertinggi (nilai > 0)")


class PerbandinganInfo(BaseModel):
    """Perbandingan antara dua sumber data RIASEC."""
    status:     str = Field(description="'SEJALAN' atau 'BERBEDA'")
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
    Response lengkap API prediksi tulisan tangan v3.1 (+ minat kuesioner).

    Struktur:
      riasec_karakter       → RIASEC dari tulisan tangan
      riasec_minat          → RIASEC dari kuesioner (None jika tidak dikirim)
      analisis_akademik     → Rumpun Ilmu + nilai akademik
      perbandingan_akademik → RIASEC tulisan vs Rumpun Ilmu
      perbandingan_minat    → RIASEC tulisan vs RIASEC minat (None jika tidak dikirim)
      rekomendasi_jurusan   → TOP-3 (semua konsisten) atau TOP-5
      fitur_tulisan         → 10 fitur numerik dari gambar
      kelengkapan_data      → status field mana yang diisi/default
    """
    riasec_karakter:       RiasecInfo               = Field(description="Profil RIASEC dari tulisan tangan")
    riasec_minat:          Optional[MinatKarakter]   = Field(None, description="Profil RIASEC dari kuesioner minat (None jika tidak dikirim)")
    analisis_akademik:     AnalisisAkademik           = Field(description="Analisis nilai akademik")
    perbandingan_akademik: PerbandinganInfo            = Field(description="RIASEC tulisan vs Rumpun Ilmu akademik")
    perbandingan_minat:    Optional[PerbandinganInfo]  = Field(None, description="RIASEC tulisan vs RIASEC kuesioner (None jika minat tidak dikirim)")
    rekomendasi_jurusan:   List[RekomendasiJurusan]   = Field(description="TOP-3 (konsisten) atau TOP-5 (ada perbedaan)")
    fitur_tulisan:         FiturTulisan               = Field(description="10 fitur tulisan tangan (OpenCV)")
    kelengkapan_data:      Dict[str, str]             = Field(description="Status kelengkapan data input")


class HealthResponse(BaseModel):
    status:       str  = Field(description="'ok' jika server berjalan normal")
    is_model_loaded: bool = Field(description="True jika semua model sudah dimuat")
    version:      str  = Field(default="3.0.0")


# ===========================================================
# Helper: konversi predictor output → PredictResponse
# ===========================================================

def build_predict_response(raw: dict) -> PredictResponse:
    """Konversi output dict predictor.predict() ke PredictResponse."""
    rk   = raw["riasec_karakter"]
    rm   = raw.get("riasec_minat")          # Optional
    aa   = raw["analisis_akademik"]
    pak  = raw["perbandingan_akademik"]
    pm   = raw.get("perbandingan_minat")    # Optional
    ft   = raw["fitur_tulisan"]
    rek  = raw["rekomendasi_jurusan"]
    kd   = raw["kelengkapan_data"]

    return PredictResponse(
        riasec_karakter=RiasecInfo(
            dominant  = rk["dominant"],
            karakter  = rk["karakter"],
            deskripsi = rk["deskripsi"],
            kekuatan  = rk["kekuatan"],
            warna     = rk["warna"],
            skor      = rk["skor"],
        ),
        riasec_minat=MinatKarakter(
            dominant    = rm["dominant"],
            karakter    = rm["karakter"],
            deskripsi   = rm["deskripsi"],
            kekuatan    = rm["kekuatan"],
            warna       = rm["warna"],
            skor_raw    = rm["skor_raw"],
            skor_persen = rm["skor_persen"],
        ) if rm else None,
        analisis_akademik=AnalisisAkademik(
            rumpun_ilmu         = aa["rumpun_ilmu"],
            rumpun_probabilitas = aa["rumpun_probabilitas"],
            nilai_rata_rata     = aa["nilai_rata_rata"],
            mata_pelajaran_kuat = aa["mata_pelajaran_kuat"],
        ),
        perbandingan_akademik=PerbandinganInfo(
            status     = pak["status"],
            penjelasan = pak["penjelasan"],
        ),
        perbandingan_minat=PerbandinganInfo(
            status     = pm["status"],
            penjelasan = pm["penjelasan"],
        ) if pm else None,
        rekomendasi_jurusan=[RekomendasiJurusan(**j) for j in rek],
        fitur_tulisan=FiturTulisan(**ft),
        kelengkapan_data=kd,
    )
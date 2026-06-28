"""
=============================================================
FILE: train_and_save.py  (Dataset Real v2)
=============================================================
Script satu-klik untuk training semua model ML.

CARA PAKAI — jalankan dari folder manapun:
  python train_and_save.py

Script ini otomatis mencari dataset di:
  1. ./data_new/   (jika ada)
  2. ./data/       (jika ada)
  3. (folder yang kamu tentukan via argumen)

DATASET YANG DIBUTUHKAN (di salah satu folder di atas):
  Dataset_AkademikN.xlsx
  Dataset_TalentN.xlsx
  Dataset_TulisanN/
    Openness/
    Conscientiousness/
    Extraversion/
    Agreeableness/
    Neuroticism/

OUTPUT (disimpan ke models/):
  bigfive_model.pkl
  rumpun_model.pkl
  major_model.pkl
  feature_meta.json
=============================================================
"""

import os
import sys
import json

# ── Tambahkan ROOT ke sys.path ──────────────────────────────
# Bekerja baik dijalankan dari:
#   C:\...\Lentera LIDM\             → ROOT = folder ini
#   C:\...\Lentera LIDM\ml-handwriting\ → ROOT = folder ini
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.training.train import run_training


# ── Auto-detect folder dataset ──────────────────────────────
def find_data_dir(root: str) -> str:
    """
    Cari folder dataset secara otomatis.
    Cek urutan: data_new/ → data/ → root itu sendiri
    """
    candidates = [
        os.path.join(root, "data_new"),
        os.path.join(root, "data"),
        root,  # kalau file xlsx langsung di root
    ]

    required_files = [
        "Dataset_AkademikN.xlsx",
        "Dataset_TalentN.xlsx",
        "Dataset_TulisanN",
    ]

    for folder in candidates:
        if all(os.path.exists(os.path.join(folder, f)) for f in required_files):
            return folder

    return ""  # tidak ditemukan


def main():
    # ── Argumen opsional: python train_and_save.py ./data ────
    if len(sys.argv) > 1:
        data_dir = os.path.abspath(sys.argv[1])
    else:
        data_dir = find_data_dir(ROOT)

    model_dir   = os.path.join(ROOT, "models")
    report_path = os.path.join(model_dir, "training_report.json")

    # ── Validasi ─────────────────────────────────────────────
    required = {
        "Dataset_AkademikN.xlsx": os.path.join(data_dir, "Dataset_AkademikN.xlsx"),
        "Dataset_TalentN.xlsx":   os.path.join(data_dir, "Dataset_TalentN.xlsx"),
        "Dataset_TulisanN/":      os.path.join(data_dir, "Dataset_TulisanN"),
    }

    if not data_dir:
        print("❌ Folder dataset tidak ditemukan!")
        print()
        print("Pastikan folder berikut ADA dan BERISI file dataset:")
        print("   data/Dataset_AkademikN.xlsx")
        print("   data/Dataset_TalentN.xlsx")
        print("   data/Dataset_TulisanN/   (folder berisi subfolder Big Five)")
        print()
        print("Atau jalankan dengan argumen:")
        print("   python train_and_save.py C:\\path\\ke\\folder\\data")
        sys.exit(1)

    missing = {k: v for k, v in required.items() if not os.path.exists(v)}
    if missing:
        print(f"❌ Dataset ditemukan di: {data_dir}")
        print("   Tapi file berikut tidak ada:")
        for nama, path in missing.items():
            print(f"   - {nama}  →  {path}")
        print()
        print("Pastikan nama file dataset PERSIS sama (termasuk huruf besar/kecil):")
        print("   Dataset_AkademikN.xlsx")
        print("   Dataset_TalentN.xlsx")
        print("   Dataset_TulisanN/  (folder)")
        sys.exit(1)

    # ── Tampilkan info sebelum training ──────────────────────
    print("=" * 60)
    print("  ML TULISAN TANGAN — TRAINING DATASET REAL v2")
    print("=" * 60)
    print(f"  📁 Dataset  : {data_dir}")
    print(f"  💾 Output   : {model_dir}")
    print()

    # Tampilkan folder yang ADA di dalam Dataset_TulisanN
    tulisan_dir = os.path.join(data_dir, "Dataset_TulisanN")
    total_gambar = 0
    if os.path.isdir(tulisan_dir):
        subfolders = [d for d in os.listdir(tulisan_dir)
                      if os.path.isdir(os.path.join(tulisan_dir, d))]
        if subfolders:
            for folder in sorted(subfolders):
                folder_path = os.path.join(tulisan_dir, folder)
                n = len([f for f in os.listdir(folder_path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                total_gambar += n
                print(f"  🖼️  {folder}: {n} gambar")
            print(f"  Total: {total_gambar} gambar")
        else:
            print("  ⚠️  Dataset_TulisanN ada tapi KOSONG — tidak ada subfolder!")
            print("  Pastikan kamu sudah extract ZIP dan ada subfolder di dalamnya.")
            sys.exit(1)
    print()

    # ── Jalankan training ────────────────────────────────────
    report = run_training(
        data_dir=data_dir,
        model_dir=model_dir,
        report_path=report_path,
    )

    # ── Ringkasan hasil ──────────────────────────────────────
    print()
    if report.get("status") == "SUCCESS":
        bf  = report["bigfive_model"]
        rum = report["rumpun_model"]
        print("✅ TRAINING BERHASIL!")
        print(f"   BigFive Accuracy : {bf['cv_accuracy_mean']:.1%}")
        print(f"   Rumpun  Accuracy : {rum['cv_accuracy_mean']:.1%}")
        print(f"   Program Studi    : {report['major_programs']} jurusan")
        print()
        print("Langkah selanjutnya — jalankan server:")
        print("   uvicorn api.main:app --reload --port 8000")
    else:
        print("❌ Training gagal. Cek pesan error di atas.")
        sys.exit(1)


if __name__ == "__main__":
    main()
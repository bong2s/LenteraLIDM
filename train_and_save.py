"""
=============================================================
FILE: train_and_save.py  (Dataset Real v3 — RIASEC Direct)
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
  Dataset_TulisanN.csv

OUTPUT (disimpan ke models/):
  riasec_model.pkl
  rumpun_model.pkl
  major_model.pkl
  feature_meta.json
=============================================================
"""

import os
import sys
import json

# ── Tambahkan ROOT ke sys.path ──────────────────────────────
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
        root,
    ]

    required_files = [
        "Dataset_AkademikN.xlsx",
        "Dataset_TulisanN.csv",
    ]

    for folder in candidates:
        if all(os.path.exists(os.path.join(folder, f)) for f in required_files):
            return folder

    return ""


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
        "Dataset_TulisanN.csv":   os.path.join(data_dir, "Dataset_TulisanN.csv"),
    }

    if not data_dir:
        print("❌ Folder dataset tidak ditemukan!")
        print()
        print("Pastikan folder berikut ADA dan BERISI file dataset:")
        print("   data_new/Dataset_AkademikN.xlsx")
        print("   data_new/Dataset_TulisanN.csv")
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
        print("   Dataset_TulisanN.csv")
        sys.exit(1)

    # ── Tampilkan info sebelum training ──────────────────────
    print("=" * 60)
    print("  ML TULISAN TANGAN — TRAINING DATASET REAL v3")
    print("=" * 60)
    print(f"  📁 Dataset  : {data_dir}")
    print(f"  💾 Output   : {model_dir}")
    print()

    # Tampilkan ringkasan CSV tulisan
    csv_path = os.path.join(data_dir, "Dataset_TulisanN.csv")
    try:
        import pandas as pd
        df_csv = pd.read_csv(csv_path)
        print(f"  📄 Dataset_TulisanN.csv : {len(df_csv)} baris")
        if "riasec_primary" in df_csv.columns:
            dist = df_csv["riasec_primary"].value_counts().to_dict()
            print(f"     Distribusi RIASEC   : {dist}")
        if "dataset_split" in df_csv.columns:
            split_dist = df_csv["dataset_split"].value_counts().to_dict()
            print(f"     Split               : {split_dist}")
    except Exception as e:
        print(f"  ⚠️  Tidak bisa preview CSV: {e}")
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
        riasec = report["riasec_model"]
        rum    = report["rumpun_model"]
        print("✅ TRAINING BERHASIL!")
        print(f"   RIASEC Accuracy : {riasec['cv_accuracy_mean']:.1%}")
        print(f"   Rumpun Accuracy : {rum['cv_accuracy_mean']:.1%}")
        print(f"   Program Studi   : {report['major_programs']} jurusan")
        print()
        print("Langkah selanjutnya — jalankan server:")
        print("   uvicorn api.main:app --reload --port 8000")
    else:
        print("❌ Training gagal. Cek pesan error di atas.")
        sys.exit(1)


if __name__ == "__main__":
    main()
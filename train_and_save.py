import sys, os, json, logging

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)s — %(message)s", datefmt="%H:%M:%S")

from src.training.train import run_training

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HANDWRITING ML — TRAINING PIPELINE")
    print("="*60)
    print(f"  Working dir: {os.getcwd()}")
    print(f"  Dataset    : data/*.xlsx")
    print(f"  Gambar     : data/img/*.png")
    print("="*60 + "\n")

    # Cek dataset ada
    required = [
        "data/Dataset_Akademik.xlsx",
        "data/Dataset_Talent.xlsx",
        "data/Dataset_Tulisan.xlsx",
    ]
    for path in required:
        if not os.path.exists(path):
            print(f"❌ File tidak ditemukan: {path}")
            print("   Cek nama file di folder data/ — harus pakai underscore, bukan spasi")
            sys.exit(1)

    if not os.path.isdir("data/img"):
        print("❌ Folder data/img tidak ditemukan.")
        sys.exit(1)

    img_count = len([f for f in os.listdir("data/img") if f.endswith(".png")])
    print(f"✅ Ditemukan {img_count} gambar tulisan tangan")

    try:
        report = run_training(
            data_dir="data",
            image_dir="data/img",
            model_dir="models",
            report_path="models/training_report.json",
        )
        print("\n" + "="*60)
        print("  ✅ TRAINING BERHASIL!")
        print("="*60)
        print(f"  Akurasi RIASEC  (CV): {report['riasec_model']['cv_accuracy_mean']:.1%}")
        print(f"  Akurasi Jurusan (CV): {report['major_model']['cv_accuracy_mean']:.1%}")
        print("\n  Langkah selanjutnya:")
        print("  uvicorn api.main:app --reload --port 8000")
        print("  Lalu buka: http://localhost:8000/docs")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ TRAINING GAGAL: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
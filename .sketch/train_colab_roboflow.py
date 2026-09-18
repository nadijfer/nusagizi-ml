"""Script Training YOLO dengan Dataset Makanan Indonesia (Roboflow Universe).

Script ini dirancang untuk dijalankan di Google Colab (GPU T4 / A100) atau Colab CLI.
Solusi 2: Fine-tuning YOLOv8 / YOLO11 menggunakan dataset yang SUDAH DILABEL oleh komunitas.

Petunjuk Singkat di Google Colab:
1. Ganti runtime ke GPU: Runtime -> Change runtime type -> T4 GPU.
2. Install dependensi:
   !pip install ultralytics roboflow
3. Jalankan script ini:
   !python train_colab_roboflow.py --epochs 30 --imgsz 640
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def check_gpu():
    print("=" * 60)
    print("🔍 MEMERIKSA STATUS GPU")
    print("=" * 60)
    print(f"PyTorch Version : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA Available  : {cuda}")
    if cuda:
        print(f"Device Name     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM      : {vram:.2f} GB")
    else:
        print("⚠️ PERINGATAN: Berjalan di CPU. Disarankan menggunakan GPU Colab.")
    print("=" * 60)


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int):
    """Download pre-labeled dataset dari Roboflow Universe format YOLOv8."""
    print("\n📦 Mengunduh dataset dari Roboflow...")
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Pustaka 'roboflow' belum terpasang. Jalankan: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8")
    print(f"✅ Dataset berhasil diunduh ke: {dataset.location}")
    return f"{dataset.location}/data.yaml"


def train_yolo(
    data_yaml: str,
    model_variant: str = "yolo11s.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
):
    """Latih model YOLO menggunakan data_yaml."""
    print(f"\n🚀 Memulai Fine-Tuning {model_variant}...")
    model = YOLO(model_variant)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True,
        save=True,
        name="nusagizi_food_model",
    )

    print("\n" + "=" * 60)
    print("🎉 Training Selesai!")
    best_weights = Path("runs/detect/nusagizi_food_model/weights/best.pt")
    if best_weights.exists():
        print(f"💾 Model terbaik tersimpan di: {best_weights.resolve()}")
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on Indonesian Food Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path ke data.yaml jika dataset sudah ada di lokal",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Jumlah epoch training (default: 25)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Model checkpoint awal (yolo11n.pt, yolo11s.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Resolusi gambar input"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Ukuran batch per iterasi"
    )

    # Argumen Roboflow (Opsional)
    parser.add_argument("--rf_key", type=str, default=None, help="Roboflow API Key")
    parser.add_argument(
        "--rf_workspace",
        type=str,
        default="a-gsuxa",
        help="Roboflow workspace name (contoh dataset publik)",
    )
    parser.add_argument(
        "--rf_project",
        type=str,
        default="indonesian-food-cd2d4",
        help="Roboflow project name",
    )
    parser.add_argument(
        "--rf_version", type=int, default=1, help="Roboflow dataset version"
    )

    args = parser.parse_args()

    check_gpu()

    data_yaml_path = args.data

    if not data_yaml_path:
        if args.rf_key:
            data_yaml_path = download_roboflow_dataset(
                args.rf_key, args.rf_workspace, args.rf_project, args.rf_version
            )
        else:
            print("\nℹ️  Pemberitahuan:")
            print("Anda belum menentukan path '--data' ataupun '--rf_key'.")
            print("Cara 1: Gunakan dataset lokal:")
            print("  python train_colab_roboflow.py --data /path/to/data.yaml")
            print("Cara 2: Otomatis download dari Roboflow:")
            print("  python train_colab_roboflow.py --rf_key YOUR_API_KEY")
            print("\nDataset rekomendasi untuk dicari di https://universe.roboflow.com:")
            print("- 'Indonesian Food' (oleh a-gsuxa)")
            print("- 'Makanan Tradisional Indonesia'")
            return

    train_yolo(
        data_yaml=data_yaml_path,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()

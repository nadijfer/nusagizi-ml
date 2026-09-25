"""Demo script to detect food ingredients using YOLO-World (Zero-Shot)."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path so nusagizi can be imported directly
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from nusagizi.detector.yolo_world import YOLOWorldDetector


def main():
    parser = argparse.ArgumentParser(description="Nusagizi Food Ingredient Detector (YOLO-World)")
    parser.add_argument(
        "--image",
        type=str,
        default="data/samples/tahu_tempe.jpg",
        help="Path ke file gambar makanan",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.18,
        help="Confidence threshold (default: 0.18)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8s-worldv2.pt",
        help="YOLO-World model checkpoint (default: models/yolov8s-worldv2.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path gambar output hasil deteksi (default: data/output/detected_<nama>.jpg)",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Gambar tidak ditemukan di '{image_path}'")
        print("💡 Tips: Jalankan 'python download_samples.py' terlebih dahulu untuk mengunduh sampel.")
        sys.exit(1)

    print("=" * 60)
    print("🍲 NUSAGIZI FOOD INGREDIENT DETECTOR (ZERO-SHOT YOLO-WORLD)")
    print("=" * 60)
    print(f"📷 Gambar Input      : {image_path.resolve()}")
    print(f"⚙️  Model Checkpoint : {args.model}")
    print(f"🎯 Threshold Conf    : {args.conf}")

    print("\n⏳ Menginisialisasi model YOLO-World...")
    detector = YOLOWorldDetector(model_name=args.model)
    print(f"📋 Kelas Terdaftar   : {len(detector.classes)} bahan makanan")
    print(f"   {', '.join(detector.classes[:8])} ...")

    print("\n🔍 Menjalankan inferensi...")
    result = detector.detect(str(image_path), conf_threshold=args.conf)

    print(f"⚡ Durasi Inferensi  : {result.inference_time_ms:.2f} ms")
    print(f"🥗 Objek Terdeteksi  : {len(result.detected_items)} item")

    # Output detail
    print("\n" + "-" * 60)
    print(f"{'No':<3} | {'Bahan Makanan':<20} | {'Kategori Gizi':<16} | {'Confidence':<10}")
    print("-" * 60)
    for i, item in enumerate(result.detected_items, 1):
        print(
            f"{i:<3} | {item.label:<20} | {item.category:<16} | {item.confidence * 100:>6.1f}%"
        )
    print("-" * 60)

    # Save visual result
    output_path = args.output or f"data/output/detected_{image_path.stem}.jpg"
    saved_path = detector.annotate_and_save(str(image_path), result, output_path)
    print(f"\n💾 Visualisasi gambar dengan bounding box tersimpan di:")
    print(f"   👉 {Path(saved_path).resolve()}")

    # Output structured JSON
    print("\n📄 JSON Output:")
    print(json.dumps(result.to_dict(), indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()

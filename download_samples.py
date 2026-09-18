"""Utility script to download sample Indonesian food images for testing."""

from pathlib import Path
import requests

SAMPLE_URLS = {
    "tahu_tempe.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/44/Tahu_tempe_goreng.jpg",
    "nasi_rames.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/44/Nasi_rames.jpg",
    "tempe_goreng.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/78/Tempe_goreng_snack.JPG",
}

HEADERS = {
    "User-Agent": "NusagiziFoodDetectorBot/1.0 (https://github.com/nusagizi; contact@nusagizi.local)"
}


def download_sample_images(output_dir: str = "samples"):
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Mengunduh sampel gambar makanan Indonesia ke folder '{output_dir}'...")
    downloaded_files = []

    for filename, url in SAMPLE_URLS.items():
        file_path = target_dir / filename
        if file_path.exists():
            print(f"  ⚡ {filename} sudah ada, melewati unduhan.")
            downloaded_files.append(str(file_path))
            continue

        print(f"  ⬇️ Mengunduh {filename} dari Wikimedia Commons...")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(resp.content)
            print(f"  ✅ Tersimpan: {file_path} ({len(resp.content) // 1024} KB)")
            downloaded_files.append(str(file_path))
        except Exception as e:
            print(f"  ❌ Gagal mengunduh {filename}: {e}")

    return downloaded_files


if __name__ == "__main__":
    download_sample_images()

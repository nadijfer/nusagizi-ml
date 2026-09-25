"""Script untuk melakukan crawling gambar makanan Indonesia dari Wikimedia Commons.

Penggunaan:
  uv run python crawl_food_images.py
  uv run python crawl_food_images.py --keywords "tempe goreng, telur dadar, sayur sop" --limit 3
"""

import argparse
from pathlib import Path
import re
import requests
import time

DEFAULT_KEYWORDS = [
    "tempe goreng",
    "tahu goreng",
    "telur dadar",
    "telur rebus",
    "ayam goreng",
    "nasi putih",
    "sayur bayam",
    "bubur ayam",
]

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "NusagiziFoodCrawler/1.0 (https://nusagizi.local; contact@nusagizi.local)"
}


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", name)


def search_wikimedia_images(query: str, limit: int = 5):
    """Mencari URL gambar dari Wikimedia Commons berdasarkan kata kunci."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,  # File namespace
        "gsrsearch": query,
        "gsrlimit": limit * 2,  # Fetch more to allow filtering
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        
        image_urls = []
        for _, page in pages.items():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            url = info.get("url")
            mime = info.get("mime", "")
            if url and ("jpeg" in mime or "png" in mime or "jpg" in mime):
                image_urls.append(url)
            if len(image_urls) >= limit:
                break
        return image_urls
    except Exception as e:
        print(f"⚠️  Gagal mencari untuk query '{query}': {e}")
        return []


def download_images(keywords: list[str], limit_per_keyword: int, output_dir: Path):
    print("=" * 60)
    print("🕷️  NUSAGIZI FOOD IMAGE CRAWLER (WIKIMEDIA COMMONS)")
    print("=" * 60)
    print(f"📁 Direktori Tujuan : {output_dir.resolve()}")
    print(f"🎯 Target Kata Kunci: {len(keywords)} kata kunci")
    print(f"🔢 Limit per Keyword: {limit_per_keyword} gambar")
    print("-" * 60)

    total_downloaded = 0

    for query in keywords:
        clean_query = query.strip()
        if not clean_query:
            continue

        category_folder = output_dir / sanitize_filename(clean_query)
        category_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n🔍 Mencari: '{clean_query}'...")
        urls = search_wikimedia_images(clean_query, limit=limit_per_keyword)
        print(f"   Ditemukan {len(urls)} gambar valid.")

        for idx, url in enumerate(urls, 1):
            ext = Path(url.split("?")[0]).suffix or ".jpg"
            filename = f"{sanitize_filename(clean_query)}_{idx:02d}{ext}"
            file_path = category_folder / filename

            if file_path.exists():
                print(f"   ⚡ [{idx}/{len(urls)}] {filename} sudah ada, dilewati.")
                continue

            try:
                img_resp = requests.get(url, headers=HEADERS, timeout=20)
                img_resp.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(img_resp.content)
                size_kb = len(img_resp.content) // 1024
                print(f"   ✅ [{idx}/{len(urls)}] Tersimpan: {filename} ({size_kb} KB)")
                total_downloaded += 1
                time.sleep(0.3)  # Polite crawling rate limit
            except Exception as e:
                print(f"   ❌ [{idx}/{len(urls)}] Gagal unduh {url}: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 Crawling selesai! Total {total_downloaded} gambar baru diunduh.")
    print(f"📂 Lokasi data: {output_dir.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Crawl Indonesian food images from Wikimedia")
    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help="Daftar kata kunci dipisahkan koma",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Jumlah gambar per kata kunci (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crawled",
        help="Folder penyimpanan gambar (default: data/crawled)",
    )

    args = parser.parse_args()
    kw_list = [k.strip() for k in args.keywords.split(",") if k.strip()]
    download_images(kw_list, args.limit, Path(args.output))


if __name__ == "__main__":
    main()

# AI/ML-eng Task Breakdown

* Tanggal Pembaruan: 18 September 2026
* Status dokumen: Aktif / Dalam Pengerjaan
* Bahasan: Progress Deteksi Bahan Makanan & Roadmap Model ML
---

## Context & Peran ML di Nusagizi
User mengunggah foto makanan (fitur kamera gaya *Locket*), ML menganalisis makanan tersebut:
* Model harus bisa mendeteksi makanan & mengidentifikasi bahan makanan
* Model harus bisa menghitung jumlah/porsi makanan
* Sistem harus memetakan ke kandungan gizi sesuai aturan Kemenkes RI (PMBA 6 bln - 5 thn)

---

## Progres Terkini (Checkpoint 18 Sept 2026)

### 1. Deteksi Bahan Makanan (Computer Vision Murni)
- [x] Riset awal arsitektur: Menggunakan pendekatan ML/CV murni (YOLO) terlebih dahulu dibanding LLM/VLM.
- [x] Implementasi **Solusi 1 (Zero-Shot YOLO-World)** di `src/nusagizi/detector/yolo_world.py`:
  - Model mampu mendeteksi bahan makanan (seperti Tempe, Tahu, Nasi, Telur) secara langsung tanpa perlu proses anotasi data manual.
  - Sudah diuji coba pada foto sampel lokal (`samples/tahu_tempe.jpg`, `samples/nasi_rames.jpg`).
- [x] Pembuatan wrapper custom YOLO (`src/nusagizi/detector/custom_yolo.py`) untuk memuat bobot hasil fine-tuning.
- [x] Pembuatan script CLI demo (`demo_detect.py`) dan crawler sampel gambar awal (`download_samples.py`).

### 2. Catatan Batasan Model Saat Ini (Important)
> [!NOTE]
> **Batasan Kemampuan Saat Ini:**
> Model yang ada sekarang **baru sekadar mendeteksi jenis/bahan makanan secara visual (Object Detection)** pada foto. 
> Model ini **BELUM menghitung nilai gizi** (kalori, protein hewani/nabati, lemak, mikronutrien) dan **BELUM menghitung porsi/volume/gramatur makanan**.

---

## Rencana & Milestone Selanjutnya

1. **Fine-Tuning YOLO untuk Makanan Indonesia (Solusi 2 di Colab)**:
   - Menyiapkan dan melatih model YOLO (YOLOv8 / YOLO11) secara *supervised* menggunakan dataset khusus makanan Indonesia (dari Roboflow Universe / hasil crawling) agar confidence dan akurasi pada menu lokal lebih tinggi.
   - Script training telah disiapkan di `train_colab_roboflow.py`.
2. **Pengumpulan Data (Crawling / Dataset Enrichment)**:
   - Melakukan crawling gambar makanan nusantara & menu MPASI untuk memperkaya data evaluasi dan training.
3. **Modul Estimasi Porsi & Pemetaan Gizi Kemenkes**:
   - Membangun *Nutrition Engine* berbasis database TKPI (Tabel Komposisi Pangan Indonesia) Kemenkes.
   - Menentukan estimasi porsi berbasis Ukuran Rumah Tangga (URT: sdm, potong, butir) yang dipadukan dengan konfirmasi pengguna di UI Locket.



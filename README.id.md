# Single Object Detection dengan MobileNetV2

Model deteksi objek tunggal yang dibangun dari nol di atas backbone MobileNetV2: diberi sebuah gambar, model memprediksi apakah ada objek di dalamnya dan meregresi bounding box-nya (`x, y, w, h`). Dibuat sebagai bagian dari materi Computer Vision bootcamp **rubythalib.ai** (mentor: Daniel Syahputra).

Berbeda dari detektor multi-objek (YOLO, SSD, Faster-RCNN), model ini mengasumsikan **satu objek per gambar** — desain yang sengaja disederhanakan untuk mempelajari komponen inti object detection: backbone bersama, head klasifikasi + regresi, dan loss function gabungan.

📁 **Dataset**: [Google Drive](https://drive.google.com/drive/folders/19ti0TK39f7SYJ06YQuxnXoLvVrp7xNhK?usp=sharing) (241 gambar)

---

## Hasil

Tiga iterasi dilatih, masing-masing menguji satu dugaan tentang apa yang membatasi akurasi bounding box. Semua angka di bawah berasal dari hasil training real (bukan estimasi) — lihat `notebooks/training.ipynb` dan `experiments/training_v2.ipynb` untuk notebook yang sudah dieksekusi lengkap dengan log tiap epoch.

| Versi | Best IoU (test) | IoU (train, epoch akhir) | Gap train/test | Perubahan utama |
|---|---|---|---|---|
| v1 (baseline) | 0.7186 | 0.8382 | 0.120 | Full fine-tune, head gabungan, loss MSE polos, tanpa augmentasi |
| v2 | 0.7276 | 0.8085 | 0.081 | + Augmentasi color-jitter (khusus train split) + partial freeze backbone (14/19 blok MobileNetV2 dibekukan) |
| v3 (saat ini) | **0.8100** | 0.8339 | **0.024** | + Split head (cabang bbox pertahankan grid spasial 4×4, bukan diperas ke 1×1) + IoU loss differentiable ditambahkan di samping MSE |

v3 adalah versi berkinerja terbaik di test set held-out, baik dari IoU mentah maupun gap train/test (overfitting paling kecil). Ini versi yang dipakai di `notebooks/training.ipynb` dan kode serving di repo ini.

Split dataset: 241 gambar → 192 train / 49 test (80/20). v1 dilatih pakai GPU Tesla T4; v2 dan v3 dilatih di CPU (kuota GPU Colab tidak tersedia saat itu) — tetap praktis mengingat ukuran MobileNetV2 dan dataset yang kecil.

### Catatan soal notebook v1

v1 langsung diiterasi menjadi v2 (augmentasi + partial freeze ditambahkan ke script yang sama), sehingga file notebook v1 yang berdiri sendiri tidak tersimpan terpisah — konfigurasi dan hasilnya didokumentasikan di atas dari log run aslinya. `experiments/training_v2.ipynb` adalah notebook tersimpan paling awal yang lengkap.

---

## Temuan utama: kemungkinan localization berbasis "mode", bukan localization per-gambar yang sesungguhnya

Di luar angka IoU agregat, menguji model yang sudah di-serve dengan foto dunia nyata **di luar** distribusi data training (mobil berbeda, negara berbeda, pencahayaan berbeda — diambil dari web, bukan dataset asli) memunculkan pola yang lebih menarik daripada sekadar "kadang akurat, kadang tidak":

- Di dua foto langit cerah berbeda (mobil berbeda, posisi berbeda di frame), v3 menghasilkan bounding box yang selisihnya **kurang dari 1%** di semua koordinat.
- Di foto langit gelap/mendung, v3 menghasilkan kotak yang jelas berbeda (sekitar 1,45× lebih besar) — jadi outputnya bukan konstanta literal.

Pola ini konsisten dengan sebuah dugaan yang layak dicatat, bukan kesimpulan final (cuma segelintir foto luar-distribusi yang diuji, bukan studi sistematis): model mungkin menangkap **statistik global gambar yang kasar** (kecerahan/kontras keseluruhan) dan mengeluarkan salah satu dari beberapa bentuk kotak "tipikal" yang sudah dipelajari untuk kondisi pencahayaan itu, bukan benar-benar mengunci ke tepi objek masing-masing secara presisi. Ini bisa menjelaskan hampir semua yang kita amati di ketiga versi — kotak longgar di pencahayaan dramatis, kotak yang gagal total di adegan jalan-ramai multi-mobil, dan kenapa "memperbaiki" satu kegagalan kualitatif (kotak v2 yang memotong mobil) lewat v3 justru langsung memunculkan kegagalan lain (v3 jadi kotak paling longgar dari ketiganya di gambar yang sama) alih-alih konvergen ke perbaikan yang bersih.

Lihat `results/gallery.jpg` untuk kumpulan lengkap foto uji kualitatif yang dirujuk di atas, dan `results/*.jpg` untuk contoh teranotasi satu-satu.

### Keterbatasan yang diketahui

- **241 gambar adalah dataset kecil.** Ketiga versi menunjukkan gap IoU train/test yang nyata (walau mengecil).
- **Asumsi objek tunggal gagal di adegan ramai.** Diuji dengan foto jalan tol berisi 8+ mobil — model menghasilkan satu kotak raksasa yang meliputi beberapa kendaraan sekaligus, bukan memilih satu (lihat `results/v1_multi_objek_gagal.jpg`). Ini memang wajar terjadi: arsitekturnya tidak punya mekanisme untuk memilih di antara beberapa kandidat objek.
- **Sensitif terhadap kondisi pencahayaan/adegan**, kemungkinan terkait pola "localization berbasis mode" di atas, bukan localization presisi-tepi yang sesungguhnya.
- Tidak ada benchmark held-out sistematis untuk foto-foto uji luar-distribusi (dipakai untuk debugging kualitatif, bukan klaim akurasi yang ketat) — perlakukan "Temuan Utama" di atas sebagai dugaan yang perlu diselidiki lebih lanjut, bukan hasil yang sudah pasti.

---

## Arsitektur

```
Input Image (224×224×3)
        │
        ▼
  Backbone MobileNetV2 (pretrained ImageNet; blok 0-13 dibekukan, 14-18 di-fine-tune)
        │
        ├──► objectness_head: AdaptiveAvgPool2d(1×1) → FC(1280→256) → FC(256→1) → sigmoid → is_object
        │
        └──► bbox_head: Conv1×1(1280→256) → AdaptiveAvgPool2d(4×4) → FC(4096→512) → FC(512→4) → sigmoid → (x, y, w, h)
```

**Loss** (`MixedLoss`): `BCE(is_object) + MSE(bbox) + IoU_loss(bbox)`, komponen IoU cuma dihitung untuk sample yang beneran ada objeknya.

---

## Struktur repo

```
notebooks/
  training.ipynb                     Notebook training utama (v3 — arsitektur terkini), dijalankan di Colab
  model_serving_colab_ngrok.ipynb    Serving via FastAPI + tunnel ngrok, dengan visualisasi bbox (Colab)
experiments/
  training_v2.ipynb                  Iterasi sebelumnya (augmentasi + partial freeze), disimpan sebagai referensi
serving/
  app.py                             Script serving FastAPI lokal (tanpa dependensi Colab/ngrok)
results/
  gallery.jpg                        Perbandingan berdampingan lintas kondisi pencahayaan/adegan dan versi
  v1_*.jpg, v3_*.jpg                 Contoh hasil uji teranotasi satu-satu
requirements.txt
LICENSE
```

---

## Setup & penggunaan

### 1. Training

1. Buka `notebooks/training.ipynb` di Google Colab.
2. Download [dataset](https://drive.google.com/drive/folders/19ti0TK39f7SYJ06YQuxnXoLvVrp7xNhK?usp=sharing) ke Google Drive-mu sendiri, dan sesuaikan `DATASET_FOLDER` di notebook dengan path-nya.
3. Jalankan semua cell. Notebook otomatis backup `best.pt`/`last.pt` ke `MyDrive/single_object_detection/checkpoints/` setelah training selesai.

### 2. Serving

**Lokal:**
```bash
pip install -r requirements.txt
# taruh best.pt di direktori yang sama dengan app.py
cd serving
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Colab (dengan public URL via ngrok):**
Buka `notebooks/model_serving_colab_ngrok.ipynb`, ganti placeholder token ngrok dengan token asli milikmu (ambil gratis di [dashboard.ngrok.com](https://dashboard.ngrok.com)), lalu jalankan semua cell.

Keduanya menyediakan endpoint `POST /predict` (upload file multipart) → response JSON:
```json
{
  "is_object": true,
  "bbox": {"x": 546.7, "y": 324.1, "w": 664.3, "h": 323.5}
}
```

⚠️ **Kompatibilitas arsitektur**: class `ObjectDetectionModel` di kode serving harus cocok dengan checkpoint yang di-load. Kode di repo ini cocok dengan **v3**. Me-load checkpoint v1/v2 ke dalamnya akan gagal dengan error `Missing key(s)/Unexpected key(s)` — ini pernah kejadian sungguhan waktu pengembangan (lihat riwayat commit), jadi perlu diketahui sebelum menukar checkpoint.

---

## Lisensi

MIT — lihat [LICENSE](LICENSE).

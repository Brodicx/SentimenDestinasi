import pandas as pd
from collections import Counter
import re
import os

# =============================
# 1. Load Dataset
# =============================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(APP_DIR, 'dataset_fix.csv')
df = pd.read_csv(csv_path)

# Pastikan kolom penting ada
if 'komentar_bersih' not in df.columns or 'label_sentimen' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'komentar_bersih' dan 'label_sentimen'")

# Normalisasi label sentimen
df['label_sentimen'] = df['label_sentimen'].str.lower().str.strip()

# =============================
# 2. Filter Sentimen Positif
# =============================
df_positif = df[df['label_sentimen'] == 'positif']

# =============================
# 3. Gabungkan Semua Komentar Positif
# =============================
all_positive = ' '.join(df_positif['komentar_bersih'].dropna())

# =============================
# 4. Tokenisasi dan Hitung Frekuensi Kata
# =============================
words_positive = re.findall(r'\b\w{4,}\b', all_positive.lower())
frekuensi_kata = Counter(words_positive)

# =============================
# 5. Daftar Referensi Tempat Wisata di Kabupaten Bandung
# =============================
daftar_tempat = [
    "Kawah Putih", "Ranca Upas", "Situ Patenggang", "Glamping Lakeside", "Pangalengan",
]

# =============================
# 6. Cek Tempat yang Disebut Positif
# =============================
rekomendasi = []
for tempat in daftar_tempat:
    if tempat.lower() in all_positive:
        rekomendasi.append(tempat.title())

# =============================
# 7. Tampilkan Rekomendasi
# =============================
print("==== Rekomendasi Tempat Wisata Berdasarkan Komentar Positif ====\n")
if rekomendasi:
    for i, r in enumerate(rekomendasi, 1):
        print(f"{i}. {r}")
else:
    print("Tidak ditemukan destinasi wisata yang disebutkan dalam komentar positif.")

# =============================
# 8. (Opsional) Tampilkan 10 Kata Positif Terbanyak
# =============================
print("\nTop 5 kata yang paling sering muncul dalam komentar positif:")
for word, count in frekuensi_kata.most_common(5):
    print(f"{word}: {count}")

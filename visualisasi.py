import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Set style
sns.set_theme()
sns.set_palette("husl")

# Baca data
print("Membaca data...")
csv_path = os.path.join(APP_DIR, 'dataset_fix.csv')
df = pd.read_csv(csv_path)
df['label_sentimen'] = df['label_sentimen'].str.strip().str.capitalize()

# Filter hanya 3 label utama
label_utama = ["Positif", "Negatif", "Netral"]
df = df[df['label_sentimen'].isin(label_utama)]

# Konversi kolom waktu jika ada
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['tanggal'] = df['timestamp'].dt.date
    df['jam'] = df['timestamp'].dt.hour
    df['hari'] = df['timestamp'].dt.day_name()

# 1. Visualisasi Distribusi Sentimen (Pie Chart)
plt.figure(figsize=(10, 3))
sentimen_counts = df['label_sentimen'].value_counts()
plt.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Sentimen Komentar')
plt.axis('equal')
plt.savefig('model/visualisasi_pie_sentimen.png')
plt.close()

# 2. Visualisasi Distribusi Sentimen (Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label_sentimen')
plt.title('Distribusi Sentimen Komentar')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Komentar')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model/visualisasi_bar_sentimen.png')
plt.close()

if 'timestamp' in df.columns:
    # 3. Tren Sentimen per Hari
    plt.figure(figsize=(15, 6))
    daily_sentiment = df.groupby(['tanggal', 'label_sentimen']).size().unstack()
    daily_sentiment.plot(kind='line', marker='o')
    plt.title('Tren Sentimen per Hari')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Komentar')
    plt.legend(title='Sentimen')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model/visualisasi_tren_harian.png')
    plt.close()

    # 4. Distribusi Komentar per Jam
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='jam', hue='label_sentimen', multiple="stack")
    plt.title('Distribusi Komentar per Jam')
    plt.xlabel('Jam')
    plt.ylabel('Jumlah Komentar')
    plt.tight_layout()
    plt.savefig('model/visualisasi_distribusi_jam.png')
    plt.close()

    # 5. Distribusi Komentar per Hari dalam Seminggu
    plt.figure(figsize=(12, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(data=df, x='hari', hue='label_sentimen', order=day_order)
    plt.title('Distribusi Komentar per Hari dalam Seminggu')
    plt.xlabel('Hari')
    plt.ylabel('Jumlah Komentar')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model/visualisasi_distribusi_hari.png')
    plt.close()

# 6. Wordcloud untuk setiap sentimen
try:
    from wordcloud import WordCloud
    
    def create_wordcloud(text, title, filename):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(filename)
        plt.close()

    for sentiment in df['label_sentimen'].unique():
        text = ' '.join(df[df['label_sentimen'] == sentiment]['komentar_bersih'])
        create_wordcloud(text, f'Wordcloud untuk Sentimen {sentiment}', 
                        f'model/wordcloud_{sentiment.lower()}.png')
except ImportError:
    print("Wordcloud tidak dapat dibuat. Install wordcloud dengan: pip install wordcloud")

    # 8. Visualisasi Destinasi Wisata Populer (hanya komentar Positif)
if 'destinasi' in df.columns:
    plt.figure(figsize=(12, 6))
    top_destinasi = (
        df[df['label_sentimen'] == 'Positif']['destinasi']
        .value_counts()
        .head(10)
    )
    sns.barplot(x=top_destinasi.values, y=top_destinasi.index, palette='viridis')
    plt.title('10 Destinasi Wisata Terpopuler (Komentar Positif)')
    plt.xlabel('Jumlah Komentar Positif')
    plt.ylabel('Destinasi')
    plt.tight_layout()
    plt.savefig('static/visualisasi/visualisasi_top_destinasi.png')
    plt.close()

    # Pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(top_destinasi, labels=top_destinasi.index, autopct='%1.1f%%', startangle=90)
    plt.title('Persentase Destinasi Wisata Terpopuler')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('static/visualisasi/visualisasi_pie_destinasi.png')
    plt.close()

# Tampilkan ringkasan statistik
print("\nRingkasan Statistik:")
print("\nDistribusi Sentimen:")
print(df['label_sentimen'].value_counts(normalize=True).round(3) * 100, "%")

if 'timestamp' in df.columns:
    print("\nWaktu Paling Aktif:")
   

print("\nTotal Komentar:", len(df)) 
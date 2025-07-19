from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import csv

app = Flask(__name__, static_folder='static')

# ======= Load Model dan Tokenizer =======
model = load_model('model/rnn_model.h5')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

label_map = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}

# Pastikan file dataset tersedia
if not os.path.exists("dataset_fix.csv"):
    pd.DataFrame(columns=['komentar', 'komentar_bersih', 'label_sentimen', 'destinasi']).to_csv("dataset_fix.csv", index=False)

# ======= Route Utama =======
@app.route('/')
def index():
    df = pd.read_csv('dataset_fix.csv')

    # Pastikan label sentimen sudah dalam lowercase/penyesuaian sesuai data Anda
    positif_komentar = list(zip(
        df[df['label_sentimen'].str.lower() == 'positif']['komentar'][:30],
        df[df['label_sentimen'].str.lower() == 'positif']['label_sentimen'][:30]
    ))
    netral_komentar = list(zip(
        df[df['label_sentimen'].str.lower() == 'netral']['komentar'][:30],
        df[df['label_sentimen'].str.lower() == 'netral']['label_sentimen'][:30]
    ))
    negatif_komentar = list(zip(
        df[df['label_sentimen'].str.lower() == 'negatif']['komentar'][:30],
        df[df['label_sentimen'].str.lower() == 'negatif']['label_sentimen'][:30]
    ))

    # Hitung hanya 3 label: Positif, Netral, Negatif
    total = len(df)
    positif = len(df[df['label_sentimen'] == 'positif'])
    netral = len(df[df['label_sentimen'] == 'netral'])
    negatif = len(df[df['label_sentimen'] == 'negatif'])

    data_komentar = list(zip(df['komentar'], df['label_sentimen']))

    visualisasi_dir = os.path.join(app.static_folder, 'visualisasi')
    gambar_list = []
    if os.path.exists(visualisasi_dir):
        for file in os.listdir(visualisasi_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                gambar_list.append('visualisasi/' + file)

    return render_template('index.html',
                           total=total,
                           positif=positif,
                           netral=netral,
                           negatif=negatif,
                           positif_komentar=positif_komentar,
                           netral_komentar=netral_komentar,
                           negatif_komentar=negatif_komentar,
                           gambar_list=gambar_list)

# ======= Route Analisis =======
@app.route('/analisis', methods=['POST'])
def analisis():
    komentar = request.form['komentar']
    komentar_bersih = komentar.lower()
    sequence = tokenizer.texts_to_sequences([komentar_bersih])
    padded = pad_sequences(sequence, maxlen=100)
    pred = model.predict(padded)
    label_index = np.argmax(pred)
    sentimen = label_map[label_index].lower()

    # Tambah komentar baru ke dataset (PASTIKAN AMAN)
    with open('dataset_fix.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([komentar, komentar_bersih, sentimen, "user_input"])

    # Reload dan tampilkan ulang
    df = pd.read_csv('dataset_fix.csv')
    total = len(df)
    positif = len(df[df['label_sentimen'] == 'positif'])
    netral = len(df[df['label_sentimen'] == 'netral'])
    negatif = len(df[df['label_sentimen'] == 'negatif'])

    # Ambil 30 komentar per sentimen
    positif_komentar = list(zip(
        df[df['label_sentimen'].str.lower() == 'positif']['komentar'][:30],
        df[df['label_sentimen'].str.lower() == 'positif']['label_sentimen'][:30]
    ))
    netral_komentar = list(zip(
        df[df['label_sentimen'].str.lower() == 'netral']['komentar'][:30],
        df[df['label_sentimen'].str.lower() == 'netral']['label_sentimen'][:30]
    ))
    negatif_komentar = list(zip(
        df[df['label_sentimen'].str.lower() == 'negatif']['komentar'][:30],
        df[df['label_sentimen'].str.lower() == 'negatif']['label_sentimen'][:30]
    ))

    visualisasi_dir = os.path.join(app.static_folder, 'visualisasi')
    gambar_list = []
    if os.path.exists(visualisasi_dir):
        for file in os.listdir(visualisasi_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                gambar_list.append('visualisasi/' + file)

    return render_template('index.html',
                           total=total,
                           positif=positif,
                           netral=netral,
                           negatif=negatif,
                           positif_komentar=positif_komentar,
                           netral_komentar=netral_komentar,
                           negatif_komentar=negatif_komentar,
                           gambar_list=gambar_list,
                           sentimen=sentimen)

# ======= Run Server =======
if __name__ == '__main__':
    print("Flask server dimulai...")
    app.run(debug=True)

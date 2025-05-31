from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)

# Load model dan data
try:
    tfidf = joblib.load('tfidf_model.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    df_grouped = joblib.load('df_grouped.pkl')
except Exception as e:
    print("Gagal memuat model atau data:", e)
    tfidf = None
    tfidf_matrix = None
    df_grouped = None

# Fungsi rekomendasi
def rekomendasi_dari_review(teks_review_user):
    user_vec = tfidf.transform([teks_review_user])
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    sorted_indices = sim_scores.argsort()[::-1]
    hasil = df_grouped.iloc[sorted_indices][['Nama_hotel', 'User_Rating']]
    return hasil.to_dict(orient='records')

# Endpoint utama
@app.route('/rekomendasi', methods=['GET'])
def rekomendasi_api():
    if not tfidf or not tfidf_matrix or not df_grouped:
        return jsonify({'error': 'Model atau data tidak dimuat dengan benar'}), 500

    review_user = request.args.get('review')
    if not review_user:
        return jsonify({'error': 'Query "review" harus disertakan'}), 400

    hasil = rekomendasi_dari_review(review_user)
    return jsonify({'rekomendasi': hasil})

# Health check
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Gunakan endpoint /rekomendasi?review=... untuk rekomendasi hotel.'})

# Run server dengan port dari environment (Railway)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

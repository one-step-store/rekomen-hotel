from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load model dan data
tfidf = joblib.load('tfidf_model.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
df_grouped = joblib.load('df_grouped.pkl')

# Fungsi rekomendasi
def rekomendasi_dari_review(teks_review_user):
    user_vec = tfidf.transform([teks_review_user])
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    sorted_indices = sim_scores.argsort()[::-1]
    hasil = df_grouped.iloc[sorted_indices][['Nama_hotel', 'User_Rating']]
    return hasil.to_dict(orient='records')

# Endpoint GET dengan query parameter
@app.route('/rekomendasi', methods=['GET'])
def rekomendasi_api():
    review_user = request.args.get('reviewe')
    if not review_user:
        return jsonify({'error': 'Query "reviewe" harus disertakan'}), 400
    
    hasil = rekomendasi_dari_review(review_user)
    return jsonify({'rekomendasi': hasil})

# Tes endpoint
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Gunakan endpoint /rekomendasi?reviewe=... untuk rekomendasi hotel.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Menjalankan Flask di port 8000

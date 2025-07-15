import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords dan tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# 1. Load model dan vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model_terbaik_f1score.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_model.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# 2. Load slangword dictionary
@st.cache_data
def load_slang_xlsm(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return dict(zip(df['slang'], df['formal']))

slang_dict = load_slang_xlsm("Slangword-indonesian.xlsm")  # Ganti path sesuai lokasi file

# 3. Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 4. Fungsi Preprocessing
def preprocess_text(text):
    # 1. Clean text
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    # 2. Lowercase
    text = text.lower()

    # 3. Tokenisasi
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Slangword normalization
    tokens = [slang_dict.get(word, word) for word in tokens]

    # 6. Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # 7. Gabungkan kembali jadi string
    return ' '.join(tokens)

# 5. Label yang digunakan
label_names = [
    "pelayanan positif", "pelayanan negatif", 
    "sdm positif", "sdm negatif", 
    "sarana dan prasarana positif", "sarana dan prasarana negatif"
]

# 6. Tampilan Streamlit
st.set_page_config(page_title="Klasifikasi Multi-Label", layout="centered")
st.title("📊 Klasifikasi Sentimen pada RSUD SYAMRABU Bangkalan")
st.markdown("Masukkan teks ulasan, lalu klik tombol **Prediksi** untuk melihat label yang sesuai.")

# 7. Input dari pengguna
input_text = st.text_area("📝 Masukkan Teks:", height=170)

# 8. Slider threshold
threshold = st.slider("🔧 Threshold untuk prediksi label", 
                      min_value=0.0, max_value=1.0, value=0.4, step=0.05)

# 9. Proses saat tombol diklik
if st.button("Prediksi"):
    if input_text.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong.")
    else:
        # 🔄 Preprocessing
        preprocessed = preprocess_text(input_text)

        # 🔄 Vectorize dan prediksi probabilitas
        X_input = vectorizer.transform([preprocessed])
        Y_proba = model.predict_proba(X_input)
        Y_proba_array = Y_proba.toarray()[0]
        Y_pred = (Y_proba_array >= threshold).astype(int)

        # 🧾 Tampilkan hasil
        st.subheader("🔍 Hasil Prediksi:")
        any_positive = False
        for i, val in enumerate(Y_pred):
            if val == 1:
                st.success(f"✅ {label_names[i]}")
                any_positive = True
        if not any_positive:
            st.info("Tidak ada label yang terdeteksi dengan threshold saat ini.")

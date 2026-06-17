import streamlit as st
import pickle
import pandas as pd

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Analisis Sentimen Shopee",
    page_icon="🛒",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open('model_nb.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center; color:#EE4D2D;'>
🛒 Analisis Sentimen Ulasan Shopee
</h1>
<p style='text-align:center;'>
Aplikasi ini digunakan untuk mengklasifikasikan sentimen ulasan pengguna Shopee menggunakan algoritma Naive Bayes
</p>
""", unsafe_allow_html=True)

st.divider()

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("📌 Informasi")
    st.write("Dataset : 796 Ulasan")
    st.write("Metode : TF-IDF + Naive Bayes")
    st.write("Akurasi : 61,88%")

# ======================
# INPUT
# ======================
review = st.text_area(
    "Masukkan Ulasan",
    placeholder="Contoh: Aplikasi Shopee sangat membantu dan mudah digunakan"
)

if st.button("🔍 Prediksi Sentimen"):

    data = tfidf.transform([review])

    hasil = model.predict(data)[0]

    if hasil == "positif":
        st.success("😊 Sentimen Positif")
    elif hasil == "negatif":
        st.error("😡 Sentimen Negatif")
    else:
        st.warning("😐 Sentimen Netral")

# ======================
# STATISTIK
# ======================
st.subheader("📊 Statistik Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Negatif", "395")
col2.metric("Positif", "280")
col3.metric("Netral", "121")

# ======================
# GRAFIK
# ======================
chart_data = pd.DataFrame({
    'Jumlah': [395, 280, 121]
}, index=['Negatif', 'Positif', 'Netral'])

st.subheader("📈 Distribusi Sentimen")

st.bar_chart(chart_data)

# ======================
# FOOTER
# ======================
st.divider()

st.caption(
    "Penelitian Analisis Sentimen Ulasan Shopee Menggunakan Algoritma Naive Bayes Berbasis Streamlit"
)
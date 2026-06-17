import streamlit as st
import pickle
import pandas as pd
import os

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Analisis Sentimen Shopee",
    page_icon="🛒",
    layout="wide"
)

# ======================
# CSS
# ======================
st.markdown("""
<style>

.stButton>button{
    background-color:#EE4D2D;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:16px;
}

div[data-testid="metric-container"]{
    background-color:#fafafa;
    border:1px solid #e6e6e6;
    padding:15px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model_nb.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center;color:#EE4D2D'>
🛒 Analisis Sentimen Ulasan Shopee
</h1>

<p style='text-align:center'>
Klasifikasi sentimen ulasan pengguna Shopee menggunakan metode TF-IDF dan Naive Bayes
</p>
""", unsafe_allow_html=True)

st.divider()

# ======================
# SIDEBAR
# ======================
with st.sidebar:

    st.title("🛒 Menu")

    menu = st.radio(
        "",
        [
            "🏠 Prediksi Sentimen",
            "📊 Dataset",
            "📝 Feedback",
            "ℹ Tentang"
        ]
    )

    st.divider()

    st.subheader("📌 Informasi")

    st.info("""
Dataset : 796 Ulasan

Metode :
• TF-IDF
• Naive Bayes

Akurasi :
61,88%
""")

# ======================
# PREDIKSI
# ======================
if menu == "🏠 Prediksi Sentimen":

    col1, col2 = st.columns([2,1])

    with col1:

        review = st.text_area(
            "Masukkan Ulasan",
            height=150,
            placeholder="Contoh: Aplikasi Shopee sangat membantu dan mudah digunakan"
        )

        if st.button("🔍 Prediksi Sentimen"):

            if review.strip() == "":
                st.warning("Masukkan ulasan terlebih dahulu")

            else:

                data = tfidf.transform([review])

                hasil = model.predict(data)[0]

                if hasil == "positif":
                    st.success("😊 Sentimen Positif")

                elif hasil == "negatif":
                    st.error("😡 Sentimen Negatif")

                else:
                    st.warning("😐 Sentimen Netral")

    with col2:

        st.metric("Negatif", 395)
        st.metric("Positif", 280)
        st.metric("Netral", 121)

    st.subheader("📈 Distribusi Sentimen")

    chart_data = pd.DataFrame(
        {
            "Jumlah": [395, 280, 121]
        },
        index=["Negatif", "Positif", "Netral"]
    )

    st.bar_chart(chart_data)

# ======================
# DATASET
# ======================
elif menu == "📊 Dataset":

    st.subheader("📊 Dataset Ulasan Shopee")

    df = pd.read_csv("shopee_reviews_Scraping.csv")

    cari = st.text_input("🔎 Cari Data")

    if cari:
        df = df[
            df.astype(str)
            .apply(lambda x: x.str.contains(cari, case=False))
            .any(axis=1)
        ]

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇ Download Dataset",
        data=df.to_csv(index=False),
        file_name="dataset_shopee.csv",
        mime="text/csv"
    )

# ======================
# FEEDBACK
# ======================
elif menu == "📝 Feedback":

    st.subheader("📝 Feedback Pengguna")

    nama = st.text_input("Nama")

    feedback = st.text_area(
        "Masukan dan Saran"
    )

    rating = st.slider(
        "Rating Aplikasi",
        1,
        5,
        5
    )

    if st.button("Kirim Feedback"):

        if nama.strip() == "" or feedback.strip() == "":
            st.warning("Nama dan feedback wajib diisi")

        else:

            data_baru = pd.DataFrame({
                "Nama": [nama],
                "Feedback": [feedback],
                "Rating": [rating]
            })

            if os.path.isfile("feedback.csv"):

                data_baru.to_csv(
                    "feedback.csv",
                    mode="a",
                    header=False,
                    index=False
                )

            else:

                data_baru.to_csv(
                    "feedback.csv",
                    index=False
                )

            st.success("✅ Terima kasih, feedback berhasil disimpan")
            st.balloons()

    st.divider()

    st.subheader("📋 Daftar Feedback")

    if os.path.isfile("feedback.csv"):

        df_feedback = pd.read_csv("feedback.csv")

        st.dataframe(
            df_feedback,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Belum ada feedback")

# ======================
# TENTANG
# ======================
elif menu == "ℹ Tentang":

    st.subheader("ℹ Tentang Aplikasi")

    st.info("""
### Analisis Sentimen Ulasan Shopee

Aplikasi ini digunakan untuk mengklasifikasikan sentimen ulasan pengguna Shopee menjadi:

😊 Positif

😐 Netral

😡 Negatif

Metode yang digunakan :

✔ TF-IDF

✔ Naive Bayes

Jumlah Dataset : 796

Akurasi Model : 61,88%

Framework : Streamlit
""")

# ======================
# FOOTER
# ======================
st.divider()

st.caption(
    "Penelitian Analisis Sentimen Ulasan Shopee Menggunakan Algoritma Naive Bayes Berbasis Streamlit"
)

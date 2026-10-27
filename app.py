
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="Klasifikasi Tiket Pelanggan",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 Klasifikasi Tiket Layanan Pelanggan")
st.write(
    "Aplikasi demonstrasi untuk memprediksi kategori tiket "
    "berdasarkan deskripsi teks menggunakan TF-IDF dan "
    "Multinomial Naive Bayes."
)

@st.cache_resource
def train_model():
    df = pd.read_csv("customer_support_tickets.csv")

    # Menggunakan deskripsi sebagai fitur dan Ticket Type sebagai target
    df = df.dropna(subset=["Ticket Description", "Ticket Type"])
    df["Ticket Description"] = df["Ticket Description"].astype(str)

    X = df["Ticket Description"]
    y = df["Ticket Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        ("model", MultinomialNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return model, accuracy, f1, len(df)

try:
    with st.spinner("Menyiapkan model klasifikasi..."):
        model, accuracy, f1, total_data = train_model()

    st.success("Model berhasil disiapkan!")

    col1, col2 = st.columns(2)
    col1.metric("Data tiket", f"{total_data:,}")
    col2.metric("Akurasi pengujian", f"{accuracy:.2%}")

    st.caption(
        f"Weighted F1-score pada data pengujian: {f1:.2%}"
    )

    st.divider()
    st.subheader("Coba prediksi tiket")

    description = st.text_area(
        "Masukkan deskripsi tiket pelanggan:",
        placeholder=(
            "Contoh: I have a problem with my payment "
            "and need assistance."
        ),
        height=150
    )

    if st.button("Prediksi Kategori", type="primary"):
        if description.strip():
            prediction = model.predict([description])[0]

            st.subheader("Hasil Prediksi")
            st.info(f"Kategori tiket: **{prediction}**")
        else:
            st.warning("Masukkan deskripsi tiket terlebih dahulu.")

    st.divider()
    st.caption(
        "Catatan: Aplikasi ini dibuat untuk demonstrasi akademik. "
        "Performa model pada eksperimen masih rendah sehingga "
        "hasil prediksi tidak boleh dianggap selalu benar."
    )

except FileNotFoundError:
    st.error(
        "Dataset tidak ditemukan. Pastikan file "
        "customer_support_tickets.csv berada di folder project."
    )
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

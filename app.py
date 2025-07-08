# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("dataset_klasifikasi_tanah.xlsx")
X = df[["pH", "Kelembapan (%)"]]
y = df["Jenis Tanah"]

# Split dan train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

# Judul
st.title("Aplikasi Klasifikasi Jenis Tanah (KNN)")

# Tampilkan akurasi
st.write(f"**Akurasi Model:** {akurasi * 100:.2f}%")

# Input pengguna
st.header("Masukkan Data Tanah Baru:")
ph = st.number_input("Nilai pH", min_value=0.0, max_value=14.0, value=6.5)
kelembapan = st.number_input("Kelembapan (%)", min_value=0, max_value=100, value=40)

if st.button("Prediksi Jenis Tanah"):
    hasil = model.predict([[ph, kelembapan]])[0]
    st.success(f"Prediksi: Jenis Tanah adalah **{hasil}**")

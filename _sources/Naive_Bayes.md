# Analisis Mendalam Klasifikasi Naive Bayes pada Dataset Iris

Dokumentasi ini memberikan panduan komprehensif mulai dari teori probabilitas dasar, penurunan rumus matematika, hingga implementasi tingkat lanjut menggunakan Python. Dataset yang digunakan direferensikan dari file `IRIS.csv.xlsx`.

---

## 1. Teori Dasar Probabilitas dan Teorema Bayes

Naive Bayes bukan sekadar satu algoritma, melainkan keluarga algoritma klasifikasi yang berakar pada **Teorema Bayes**. Teorema ini mendeskripsikan probabilitas suatu kejadian berdasarkan pengetahuan sebelumnya (prior knowledge) mengenai kondisi yang terkait dengan kejadian tersebut.

Secara matematis, Teorema Bayes dirumuskan sebagai:

$$P(C_k | X) = \frac{P(X | C_k) \cdot P(C_k)}{P(X)}$$

**Penjelasan Komponen:**
1. **$P(C_k | X)$ (Posterior Probability)**: Peluang observasi data $X$ masuk ke dalam kelas $C_k$ (misal: kelas *Iris-setosa*). Inilah yang ingin kita cari.
2. **$P(C_k)$ (Prior Probability)**: Peluang awal dari kelas $C_k$ sebelum melihat data $X$. Jika dari 150 data Iris terdapat 50 Setosa, maka prior untuk Setosa adalah $50/150 = 0.33$.
3. **$P(X | C_k)$ (Likelihood)**: Peluang munculnya fitur $X$ asalkan kita tahu datanya berasal dari kelas $C_k$.
4. **$P(X)$ (Evidence)**: Probabilitas total dari fitur $X$. Karena nilai ini sama untuk semua kelas, seringkali diabaikan dalam perhitungan komparatif (kita hanya membandingkan pembilangnya saja).

**Sifat "Naive" (Naif)**:
Algoritma ini disebut "naif" karena mengasumsikan bahwa **setiap fitur $X_1, X_2, ..., X_n$ saling independen bersyarat** (tidak saling memengaruhi). Sehingga likelihood bisa dipecah menjadi perkalian:

$$P(X | C_k) = P(x_1 | C_k) \times P(x_2 | C_k) \times \dots \times P(x_n | C_k)$$

---

## 2. Gaussian Naive Bayes (Menangani Data Numerik)

Dataset `IRIS.csv.xlsx` berisi data numerik kontinu (*sepal_length, sepal_width, petal_length, petal_width*). Kita tidak bisa menghitung probabilitas kejadian pasti dari angka kontinu. Sebagai solusinya, kita menggunakan asumsi **Distribusi Gaussian (Normal)**.

Model mengasumsikan distribusi data untuk setiap kelas berbentuk lonceng (bell curve). Fungsi Kepadatan Peluang (Probability Density Function / PDF) untuk fitur $x_i$ pada kelas $C_k$ adalah:

$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma^2_{i,k}}} \exp\left(-\frac{(x_i - \mu_{i,k})^2}{2\sigma^2_{i,k}}\right)$$

**Langkah Kerja Algoritma secara Matematis:**
1. **Hitung Rata-rata ($\mu$)**: Model menghitung rata-rata dari setiap fitur untuk masing-masing kelas.
2. **Hitung Varians ($\sigma^2$)**: Model menghitung penyebaran data (varians) dari setiap fitur untuk masing-masing kelas.
3. **Hitung Likelihood**: Saat ada data baru $X_{baru}$, model memasukkan nilai fitur tersebut ke dalam rumus PDF Gaussian di atas.
4. **Prediksi**: Kelas yang menghasilkan nilai perkalian Prior dan Likelihood terbesar adalah kelas prediksinya (disebut metode *Maximum A Posteriori* atau MAP).

---

## 3. Implementasi Python Tingkat Lanjut

Pada implementasi ini, kita tidak hanya melatih model, tetapi juga:
- Melakukan **Visualisasi Distribusi Data** untuk memvalidasi asumsi Gaussian.
- Membedah isi model `GaussianNB` untuk **melihat nilai rata-rata ($\mu$) dan varians ($\sigma^2$)** yang dihitung model (membuktikan rumusnya bekerja).
- Membuat **Confusion Matrix** untuk analisis error klasifikasi.

```python
# ==========================================
# IMPORT LIBRARY
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Mengatur tampilan visualisasi agar lebih rapi
sns.set_theme(style="whitegrid")

# ==========================================
# 1. MEMUAT DATASET
# ==========================================
# Membaca dataset dari file Excel yang direferensikan
try:
    # Karena ekstensinya .xlsx, kita gunakan read_excel
    df = pd.read_excel('IRIS.csv.xlsx')
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print("Error: File IRIS.csv.xlsx tidak ditemukan.")

# ==========================================
# 2. EKSPLORASI DATA & VALIDASI ASUMSI GAUSSIAN
# ==========================================
# Naive Bayes Gaussian berasumsi data terdistribusi normal (berbentuk lonceng).
# Mari kita visualisasikan distribusi 'petal_length' untuk masing-masing spesies.
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='petal_length', hue='species', fill=True)
plt.title('Distribusi Petal Length per Spesies (Validasi Asumsi Gaussian)')
plt.xlabel('Petal Length')
plt.ylabel('Density (Kepadatan)')
plt.show()

# ==========================================
# 3. PERSIAPAN DATA (SPLIT)
# ==========================================
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. TRAINING MODEL GAUSSIAN NAIVE BAYES
# ==========================================
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# ==========================================
# 5. MEMBEDAH MATEMATIKA DI DALAM MODEL (PENDALAMAN)
# ==========================================
print("\n--- PENDALAMAN MATEMATIKA MODEL ---")
print("Kelas yang ditemukan oleh model:", model_nb.classes_)
print("Probabilitas Prior masing-masing kelas (P(C)):", np.exp(model_nb.class_log_prior_))

# Melihat rata-rata (Mu) yang dihitung model untuk kelas pertama (misal Setosa)
print("\nRata-rata (Mu) fitur untuk masing-masing kelas:")
df_mu = pd.DataFrame(model_nb.theta_, columns=X.columns, index=model_nb.classes_)
print(df_mu)

# Melihat varians (Sigma^2) yang dihitung model
print("\nVarians (Sigma^2) fitur untuk masing-masing kelas:")
df_var = pd.DataFrame(model_nb.var_, columns=X.columns, index=model_nb.classes_)
print(df_var)

# ==========================================
# 6. PREDIKSI DAN EVALUASI MENDALAM
# ==========================================
y_pred = model_nb.predict(X_test)

print("\n--- EVALUASI PERFORMA ---")
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Laporan Detail Klasifikasi:")
print(classification_report(y_test, y_pred))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model_nb.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model_nb.classes_, yticklabels=model_nb.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Prediksi Model')
plt.ylabel('Kenyataan Aktual')
plt.show()
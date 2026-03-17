---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Preprocessing: Missing Values (WKNN) & Normalisasi
 
Materi ini mencakup dua tahapan preprocessing penting:
 
1. **Missing Values** — Penyelesaian dengan WKNN (Manual + Code)
2. **Normalisasi Data** — Min-Max, Z-Score, Decimal Scaling
 
---
 
## 1. Missing Values dengan WKNN
 
### 1.1 Apa Itu WKNN?
 
**WKNN (Weighted K-Nearest Neighbour Imputation)** adalah pengembangan dari metode **KNNI** standar untuk mengisi nilai yang hilang (*missing values*).
 
| Metode | Cara Kerja |
|---|---|
| **KNNI Standar** | Rata-rata sederhana dari K tetangga terdekat |
| **WKNN** | Rata-rata **tertimbang** — tetangga yang lebih mirip mendapat **bobot lebih besar** |
 
### 1.2 Rumus WKNN
 
**Langkah 1 — Hitung Ukuran Kemiripan ($s_i$)**
 
$$\frac{1}{s_i} = \sum_{h \in O_i \cap O_j} (y_{ih} - y_{jh})^2$$
 
- $O_i$ = himpunan atribut **teramati** (tidak hilang) pada data target
- Hanya atribut yang tersedia yang digunakan untuk menghitung jarak
- Semakin **kecil** $\frac{1}{s_i}$ → semakin **mirip** → $s_i$ semakin **besar**
 
**Langkah 2 — Estimasi Nilai Hilang (Weighted Average)**
 
$$\hat{y}_{ih} = \frac{\displaystyle\sum_{j \in I_{Kth}} s_i(y_j) \cdot y_{jh}}{\displaystyle\sum_{j \in I_{Kth}} s_i(y_j)}$$
 
- **Pembilang**: jumlah perkalian Bobot Kemiripan × Nilai Tetangga
- **Penyebut**: jumlah total Bobot Kemiripan
 
---
 
### 1.3 Studi Kasus — Data IPK, Penghasilan Orang Tua, JML
 
Data yang diberikan:
 
| No | IPK | PO (Rp) | JML |
|---|---|---|---|
| 1 | 2 | 2.000.000 | 2 |
| 2 | 3 | 3.000.000 | 3 |
| 3 | 4 | 2.000.000 | 2 |
| 4 | 2 | 2.000.000 | 3 |
| 5 | 3 | 3.000.000 | 2 |
| 6 | 4 | 4.000.000 | 3 |
| **7** | **2** | **3.000.000** | **?** |
 
**Tujuan**: Estimasi nilai **JML** yang hilang pada baris ke-7.
 
### 1.4 Penyelesaian Manual
 
**Tahap 1 — Normalisasi Min-Max**
 
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
 
| No | IPK | PO | JML |
|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 0.5 | 0.5 | 1.0 |
| 3 | 1.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 1.0 |
| 5 | 0.5 | 0.5 | 0.0 |
| 6 | 1.0 | 1.0 | 1.0 |
| **7** | **0.0** | **0.5** | **?** |
 
**Tahap 2 — Hitung $\frac{1}{s_i}$ dan $s_i$**
 
Atribut teramati pada baris 7: **IPK = 0.0** dan **PO = 0.5** (JML hilang, tidak dipakai).
 
| Tetangga | $(IPK_7-IPK_j)^2$ | $(PO_7-PO_j)^2$ | $\frac{1}{s_i}$ (jumlah) | $s_i$ |
|---|---|---|---|---|
| Baris 1 | 0.00 | 0.25 | 0.25 | 4.000 |
| Baris 2 | 0.25 | 0.00 | 0.25 | 4.000 |
| Baris 3 | 1.00 | 0.25 | 1.25 | 0.800 |
| Baris 4 | 0.00 | 0.25 | 0.25 | 4.000 |
| Baris 5 | 0.25 | 0.00 | 0.25 | 4.000 |
| Baris 6 | 1.00 | 0.25 | 1.25 | 0.800 |
 
**Tahap 3 — Weighted Average**
 
$$\hat{JML}_{norm} = \frac{4.0(0)+4.0(1)+0.8(0)+4.0(1)+4.0(0)+0.8(1)}{4.0+4.0+0.8+4.0+4.0+0.8} = \frac{8.8}{17.6} = 0.5$$
 
**Denormalisasi:**
 
$$JML = 0.5 \times (3 - 2) + 2 = \boxed{2.5}$$
 
---
 
### 1.5 Import Library
 
```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
 
print("Library berhasil diimport!")
print(f"  numpy  : {np.__version__}")
print(f"  pandas : {pd.__version__}")
```
 
### 1.6 Data Awal
 
```{code-cell} ipython3
data = {
    'IPK': [2, 3, 4, 2, 3, 4, 2],
    'PO':  [2_000_000, 3_000_000, 2_000_000, 2_000_000,
            3_000_000, 4_000_000, 3_000_000],
    'JML': [2, 3, 2, 3, 2, 3, np.nan]
}
 
df = pd.DataFrame(data, index=range(1, 8))
print("DATA ASLI (dengan missing value):")
print("=" * 40)
print(df.to_string())
print(f"\nJumlah nilai hilang : {df.isnull().sum().sum()} cell")
print(f"Lokasi              : baris 7, kolom JML")
```
 
### 1.7 Normalisasi Min-Max
 
```{code-cell} ipython3
def minmax_normalize(df):
    """Normalisasi Min-Max: (x - min) / (max - min), output [0, 1]"""
    df_norm = df.copy().astype(float)
    params  = {}
    for col in df.columns:
        non_null    = df[col].dropna()
        col_min     = non_null.min()
        col_max     = non_null.max()
        params[col] = {'min': col_min, 'max': col_max}
        if col_max != col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
    return df_norm, params
 
 
df_norm, norm_params = minmax_normalize(df)
 
print("DATA SETELAH NORMALISASI MIN-MAX:")
print("=" * 40)
print(df_norm.round(4).to_string())
print("\nParameter Normalisasi:")
for col, p in norm_params.items():
    print(f"  {col:5s} -> min={p['min']:>12,.0f}  max={p['max']:>12,.0f}")
```
 
### 1.8 Hitung Kemiripan
 
```{code-cell} ipython3
def hitung_kemiripan(df_norm, target_idx, missing_col):
    """
    Hitung kemiripan s_i antara baris target dan semua tetangga.
    Hanya menggunakan atribut yang TERAMATI (tidak hilang).
    """
    observed_cols = [c for c in df_norm.columns if c != missing_col]
    target_row    = df_norm.loc[target_idx, observed_cols]
    neighbors     = df_norm.drop(index=target_idx)
 
    results = []
    for idx, row in neighbors.iterrows():
        sq_sum     = np.sum((target_row.values - row[observed_cols].values) ** 2)
        similarity = np.inf if sq_sum == 0 else 1.0 / sq_sum
        results.append({
            'Tetangga'          : f'Baris {idx}',
            '1/s_i (jarak^2)'  : round(sq_sum, 4),
            's_i (kemiripan)'  : round(similarity, 4),
            f'{missing_col}_j' : row[missing_col]
        })
    return pd.DataFrame(results)
 
 
sim_df = hitung_kemiripan(df_norm, target_idx=7, missing_col='JML')
print("TABEL KEMIRIPAN — TARGET: Baris 7")
print("=" * 55)
print(sim_df.to_string(index=False))
```
 
### 1.9 Estimasi Nilai Hilang
 
```{code-cell} ipython3
def wknn_impute_single(df_norm, norm_params, target_idx, missing_col, K=None):
    """Estimasi SATU nilai hilang menggunakan Weighted KNN Imputation."""
    sim_df   = hitung_kemiripan(df_norm, target_idx, missing_col)
    inf_mask = sim_df['s_i (kemiripan)'] == np.inf
 
    if inf_mask.any():
        est_norm = sim_df.loc[inf_mask, f'{missing_col}_j'].mean()
    else:
        sim_df = sim_df.sort_values('s_i (kemiripan)', ascending=False)
        if K is not None:
            sim_df = sim_df.head(K)
        weights   = sim_df['s_i (kemiripan)'].values
        values    = sim_df[f'{missing_col}_j'].values
        pembilang = np.sum(weights * values)
        penyebut  = np.sum(weights)
        est_norm  = pembilang / penyebut
        print(f"Pembilang  (sum s_i x {missing_col}_j) = {pembilang:.4f}")
        print(f"Penyebut   (sum s_i)              = {penyebut:.4f}")
        print(f"{missing_col} (norm)              = {est_norm:.4f}")
 
    p       = norm_params[missing_col]
    est_val = est_norm * (p['max'] - p['min']) + p['min']
    print(f"\nDenormalisasi:")
    print(f"  {missing_col} = {est_norm:.4f} x ({p['max']} - {p['min']}) + {p['min']}")
    print(f"  {missing_col} = {est_val:.4f}")
    return est_val
 
 
print("ESTIMASI NILAI HILANG (Baris 7, JML)")
print("=" * 45)
jml_estimasi = wknn_impute_single(
    df_norm, norm_params, target_idx=7, missing_col='JML'
)
print(f"\n>>> JML yang hilang diestimasi = {jml_estimasi:.2f} <<<")
```
 
### 1.10 WKNN General (Otomatis)
 
```{code-cell} ipython3
def wknn_imputation(df, K=None, verbose=True):
    """WKNN Imputation otomatis untuk seluruh DataFrame."""
    df_result            = df.copy().astype(float)
    df_norm_all, np_all  = minmax_normalize(df_result)
 
    missing_pos = [
        (i, c) for i in df_result.index
        for c in df_result.columns
        if pd.isna(df_result.loc[i, c])
    ]
 
    if not missing_pos:
        print("Tidak ada nilai hilang!")
        return df_result
 
    print(f"Ditemukan {len(missing_pos)} nilai hilang:")
    for (idx, col) in missing_pos:
        print(f"  -> baris {idx}, kolom '{col}'")
    print()
 
    for (idx, col) in missing_pos:
        obs    = [c for c in df_result.columns
                  if c != col and not pd.isna(df_result.loc[idx, c])]
        t_vals = df_norm_all.loc[idx, obs].values
        sims, nvals = [], []
        for nidx in df_result.index:
            if nidx == idx or pd.isna(df_result.loc[nidx, col]):
                continue
            sq = np.sum((t_vals - df_norm_all.loc[nidx, obs].values) ** 2)
            sims.append(np.inf if sq == 0 else 1.0 / sq)
            nvals.append(df_norm_all.loc[nidx, col])
 
        w, v = np.array(sims), np.array(nvals)
        if K is not None:
            top  = np.argsort(w)[::-1][:K]
            w, v = w[top], v[top]
 
        est_n   = np.mean(v[np.isinf(w)]) if np.any(np.isinf(w)) \
                  else np.sum(w * v) / np.sum(w)
        p       = np_all[col]
        est_val = est_n * (p['max'] - p['min']) + p['min']
        df_result.loc[idx, col] = est_val
        if verbose:
            print(f"  Baris {idx}, Kolom '{col}' -> diisi: {est_val:.4f}")
 
    return df_result
 
 
print("WKNN IMPUTATION — OTOMATIS")
print("=" * 45)
df_filled = wknn_imputation(df)
print("\nHASIL AKHIR:")
print("=" * 45)
print(df_filled.to_string())
print(f"\nMissing values tersisa: {df_filled.isnull().sum().sum()}")
```
 
---
 
## 2. WKNN untuk Data Klasifikasi
 
Jika data memiliki **label kelas**, gunakan **Same-Class Weighting** — hanya tetangga dengan kelas yang sama yang dipertimbangkan.
 
| ID | Pengalaman | Skor Tes | Kelas |
|---|---|---|---|
| y1 | 2 | 70 | 0 (Gagal) |
| y2 | 3 | 75 | 0 (Gagal) |
| y3 | 5 | 85 | 1 (Lulus) |
| y4 | 6 | 88 | 1 (Lulus) |
| y5 | 1 | 65 | 0 (Gagal) |
| y6 | 7 | 92 | 1 (Lulus) |
| **y7** | **5.5** | **?** | **1 (Lulus)** |
 
### 2.1 Data Klasifikasi
 
```{code-cell} ipython3
data_klas = {
    'Pengalaman': [2,  3,  5,  6,  1,  7,  5.5],
    'Skor_Tes':  [70, 75, 85, 88, 65, 92, np.nan],
    'Kelas':     [0,  0,  1,  1,  0,  1,  1]
}
df_klas = pd.DataFrame(data_klas, index=[f'y{i}' for i in range(1, 8)])
print("DATA KLASIFIKASI:")
print("=" * 38)
print(df_klas.to_string())
```
 
### 2.2 Hitung Kemiripan dan Filter Kelas
 
```{code-cell} ipython3
target_pengalaman = 5.5
target_kelas      = 1
 
tetangga = df_klas.dropna(subset=['Skor_Tes']).copy()
tetangga['selisih']   = tetangga['Pengalaman'] - target_pengalaman
tetangga['kuadrat']   = tetangga['selisih'] ** 2
tetangga['kemiripan'] = 1.0 / tetangga['kuadrat']
 
print("LANGKAH 1 — Kemiripan Semua Tetangga:")
print("=" * 58)
print(tetangga[['Pengalaman','Skor_Tes','Kelas',
                'selisih','kuadrat','kemiripan']].to_string())
 
same_class = tetangga[tetangga['Kelas'] == target_kelas].copy()
print(f"\nLANGKAH 2 — Filter Kelas = {target_kelas} (Lulus):")
print("=" * 45)
print(same_class[['Pengalaman','Skor_Tes','Kelas','kemiripan']].to_string())
```
 
### 2.3 Estimasi Skor Tes
 
```{code-cell} ipython3
w         = same_class['kemiripan'].values
v         = same_class['Skor_Tes'].values
pembilang = np.sum(w * v)
penyebut  = np.sum(w)
skor_pred = pembilang / penyebut
 
print("LANGKAH 3 — Weighted Average:")
print("=" * 45)
print("\nRincian Pembilang:")
for wi, vi, idx in zip(w, v, same_class.index):
    print(f"  {idx}: {wi:.3f} x {vi} = {wi*vi:.3f}")
print(f"\nPembilang = {pembilang:.3f}")
print(f"Penyebut  = {penyebut:.4f}")
print(f"\n>>> Skor Tes diestimasi = {skor_pred:.2f} <<<")
```
 
---
 
## 3. Normalisasi Data
 
### 3.1 Mengapa Normalisasi Diperlukan?
 
Normalisasi adalah **tahapan preprocessing** untuk menyeragamkan skala fitur data sebelum digunakan pada algoritma machine learning.
 
**Alasan pentingnya normalisasi:**
- Algoritma berbasis jarak (KNN, SVM, K-Means) **sensitif terhadap skala**
- Gradient descent konvergen **lebih cepat**
- Mencegah fitur bernilai besar **mendominasi** fitur bernilai kecil
 
### 3.2 Tiga Metode Normalisasi
 
| No | Metode | Rumus | Output |
|---|---|---|---|
| 1 | **Min-Max** | $\dfrac{x - x_{min}}{x_{max} - x_{min}}$ | $[0,\ 1]$ |
| 2 | **Z-Score** | $\dfrac{x - \mu}{\sigma}$ | $\approx[-3,\ 3]$ |
| 3 | **Decimal Scaling** | $\dfrac{x}{10^j},\ j = \lceil\log_{10}(\max\|x\|)\rceil$ | $[-1,\ 1]$ |
 
---
 
### 3.3 Import Library & Data
 
```{code-cell} ipython3
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
 
# Data contoh dengan skala berbeda
data_raw = pd.DataFrame({
    'IPK'    : [2.5, 3.0, 3.5, 2.0, 3.8, 4.0, 3.2],
    'PO'     : [2_000_000, 3_000_000, 5_000_000, 1_500_000,
                4_000_000, 8_000_000, 3_500_000],
    'JML_SKS': [120, 140, 130, 110, 145, 150, 135]
}, index=[f'D{i+1}' for i in range(7)])
 
print("DATA ASLI (skala berbeda-beda):")
print("=" * 52)
print(data_raw.to_string())
print("\nStatistik Deskriptif:")
print(data_raw.describe().round(2))
```
 
---
 
### 3.4 Metode 1 — Min-Max Normalization
 
**Konsep:** Menggeser dan menykalakan data sehingga nilai minimum menjadi 0 dan nilai maksimum menjadi 1.
 
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$
 
**Kelebihan:** Mudah dipahami, output range tetap [0, 1]  
**Kekurangan:** Sensitif terhadap outlier (outlier menarik seluruh data ke ujung range)
 
```{code-cell} ipython3
# ── Fungsi Manual ──────────────────────────────────────────
def minmax_manual(df):
    """
    Min-Max Normalization (fungsi manual).
    Rumus  : (x - min) / (max - min)
    Output : [0, 1]
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    info   = {}
    for col in df.columns:
        x_min       = df[col].min()
        x_max       = df[col].max()
        info[col]   = {'x_min': x_min, 'x_max': x_max}
        result[col] = (df[col] - x_min) / (x_max - x_min)
    return result, info
 
# ── Dengan sklearn ─────────────────────────────────────────
scaler_mm   = MinMaxScaler(feature_range=(0, 1))
data_mm_sk  = pd.DataFrame(
    scaler_mm.fit_transform(data_raw),
    columns=data_raw.columns,
    index=data_raw.index
).round(4)
 
# ── Fungsi manual ──────────────────────────────────────────
data_mm_man, mm_info = minmax_manual(data_raw)
data_mm_man = data_mm_man.round(4)
 
# ── Tampilkan ──────────────────────────────────────────────
print("METODE 1: MIN-MAX NORMALIZATION")
print("Rumus: (x - x_min) / (x_max - x_min)")
print("=" * 52)
 
print("\nParameter (min & max tiap kolom):")
for col, v in mm_info.items():
    print(f"  {col:8s} -> x_min = {v['x_min']:>12,.1f}  |  x_max = {v['x_max']:>12,.1f}")
 
print("\nHasil [sklearn]:")
print(data_mm_sk)
 
print("\nHasil [Manual]:")
print(data_mm_man)
 
# Verifikasi kesamaan
selisih = (data_mm_sk - data_mm_man).abs().max().max()
print(f"\nSelisih sklearn vs manual : {selisih:.6f}  (0 = identik)")
print(f"Range output              : [{data_mm_sk.min().min():.2f}, {data_mm_sk.max().max():.2f}]")
```
 
```{code-cell} ipython3
# Contoh perhitungan langkah demi langkah untuk IPK
print("Contoh Perhitungan Manual — Kolom IPK:")
print("=" * 45)
ipk    = data_raw['IPK']
x_min  = ipk.min()
x_max  = ipk.max()
print(f"  x_min = {x_min}  |  x_max = {x_max}")
print()
for idx, val in ipk.items():
    hasil = (val - x_min) / (x_max - x_min)
    print(f"  {idx}: ({val} - {x_min}) / ({x_max} - {x_min}) = {hasil:.4f}")
```
 
---
 
### 3.5 Metode 2 — Z-Score Standardization
 
**Konsep:** Mengubah distribusi data sehingga memiliki mean = 0 dan standar deviasi = 1.
 
$$x' = \frac{x - \mu}{\sigma}$$
 
dimana $\mu$ = mean dan $\sigma$ = standar deviasi.
 
**Kelebihan:** Cocok untuk data berdistribusi normal, tidak terpengaruh outlier sebesar Min-Max  
**Kekurangan:** Output tidak memiliki batas range yang tetap
 
```{code-cell} ipython3
# ── Fungsi Manual ──────────────────────────────────────────
def zscore_manual(df):
    """
    Z-Score Standardization (fungsi manual).
    Rumus  : (x - mean) / std
    Output : mean = 0, std = 1
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    info   = {}
    for col in df.columns:
        mu          = df[col].mean()
        sigma       = df[col].std()
        info[col]   = {'mean': mu, 'std': sigma}
        result[col] = (df[col] - mu) / sigma
    return result, info
 
# ── Dengan sklearn ─────────────────────────────────────────
scaler_std  = StandardScaler()
data_zs_sk  = pd.DataFrame(
    scaler_std.fit_transform(data_raw),
    columns=data_raw.columns,
    index=data_raw.index
).round(4)
 
# ── Fungsi manual ──────────────────────────────────────────
data_zs_man, zs_info = zscore_manual(data_raw)
data_zs_man = data_zs_man.round(4)
 
# ── Tampilkan ──────────────────────────────────────────────
print("METODE 2: Z-SCORE STANDARDIZATION")
print("Rumus: (x - mean) / std")
print("=" * 52)
 
print("\nParameter (mean & std tiap kolom):")
for col, v in zs_info.items():
    print(f"  {col:8s} -> mean = {v['mean']:>12,.2f}  |  std = {v['std']:>12,.2f}")
 
print("\nHasil [sklearn]:")
print(data_zs_sk)
 
print("\nHasil [Manual]:")
print(data_zs_man)
 
print(f"\nVerifikasi mean setelah Z-Score : {data_zs_sk.mean().round(10).to_dict()}")
print(f"Verifikasi std  setelah Z-Score : {data_zs_sk.std().round(4).to_dict()}")
```
 
```{code-cell} ipython3
# Contoh perhitungan langkah demi langkah untuk IPK
print("Contoh Perhitungan Manual — Kolom IPK:")
print("=" * 45)
ipk   = data_raw['IPK']
mu    = ipk.mean()
sigma = ipk.std()
print(f"  mean (mu)    = {mu:.4f}")
print(f"  std  (sigma) = {sigma:.4f}")
print()
for idx, val in ipk.items():
    hasil = (val - mu) / sigma
    print(f"  {idx}: ({val} - {mu:.4f}) / {sigma:.4f} = {hasil:.4f}")
```
 
---
 
### 3.6 Metode 3 — Decimal Scaling
 
**Konsep:** Membagi setiap nilai dengan pangkat 10 tertentu sehingga nilai absolut maksimum menjadi kurang dari 1.
 
$$x' = \frac{x}{10^j}$$
 
dimana $j$ adalah bilangan bulat terkecil sehingga $\max(|x'|) < 1$:
 
$$j = \lceil \log_{10}(\max|x|) \rceil$$
 
**Kelebihan:** Sangat sederhana, mempertahankan proporsi antar data  
**Kekurangan:** Output range bergantung pada skala data asli
 
```{code-cell} ipython3
# ── Fungsi Manual ──────────────────────────────────────────
def decimal_scaling_manual(df):
    """
    Decimal Scaling (fungsi manual).
    Rumus  : x / 10^j
    j      : ceil(log10(max|x|))
    Output : nilai absolut maks < 1
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    info   = {}
    for col in df.columns:
        max_abs     = df[col].abs().max()
        j           = math.ceil(math.log10(max_abs)) if max_abs >= 1 else 1
        info[col]   = {'max_abs': max_abs, 'j': j, 'pembagi': 10**j}
        result[col] = df[col] / (10 ** j)
    return result, info
 
# ── Tidak ada sklearn khusus, pakai fungsi manual ──────────
data_ds_man, ds_info = decimal_scaling_manual(data_raw)
data_ds_man = data_ds_man.round(6)
 
# ── Tampilkan ──────────────────────────────────────────────
print("METODE 3: DECIMAL SCALING")
print("Rumus: x / 10^j   |   j = ceil(log10(max|x|))")
print("=" * 52)
 
print("\nParameter (nilai j & pembagi tiap kolom):")
for col, v in ds_info.items():
    print(f"  {col:8s} -> max|x| = {v['max_abs']:>12,.1f}  |  j = {v['j']}  |  pembagi = {v['pembagi']:>12,.0f}")
 
print("\nHasil [Manual]:")
print(data_ds_man)
 
print(f"\nVerifikasi max|x'| tiap kolom (harus < 1):")
for col in data_ds_man.columns:
    print(f"  {col:8s} -> max|x'| = {data_ds_man[col].abs().max():.6f}")
```
 
```{code-cell} ipython3
# Contoh perhitungan langkah demi langkah untuk semua kolom
print("Contoh Perhitungan Manual — Decimal Scaling:")
print("=" * 52)
for col in data_raw.columns:
    max_abs = data_raw[col].abs().max()
    j       = math.ceil(math.log10(max_abs)) if max_abs >= 1 else 1
    pembagi = 10 ** j
    print(f"\nKolom {col}:")
    print(f"  max|x|  = {max_abs:.1f}")
    print(f"  j       = ceil(log10({max_abs:.1f})) = {j}")
    print(f"  pembagi = 10^{j} = {pembagi:,}")
    for idx, val in data_raw[col].items():
        hasil = val / pembagi
        print(f"  {idx}: {val:>12,.1f} / {pembagi:>12,.0f} = {hasil:.6f}")
```
 
---
 
### 3.7 Perbandingan Ketiga Metode
 
```{code-cell} ipython3
# ── Tabel perbandingan hasil normalisasi (kolom IPK) ───────
print("PERBANDINGAN HASIL NORMALISASI — Kolom IPK:")
print("=" * 60)
 
tabel = pd.DataFrame({
    'Data Asli'       : data_raw['IPK'].values,
    'Min-Max'         : data_mm_man['IPK'].values,
    'Z-Score'         : data_zs_man['IPK'].values,
    'Decimal Scaling' : data_ds_man['IPK'].values,
}, index=data_raw.index)
 
print(tabel.to_string())
 
print("\n\nStatistik Setelah Normalisasi — Kolom IPK:")
print("=" * 60)
stat = pd.DataFrame({
    'Metode'         : ['Data Asli','Min-Max','Z-Score','Decimal Scaling'],
    'Min'            : [data_raw['IPK'].min(),    data_mm_man['IPK'].min(),
                        data_zs_man['IPK'].min(), data_ds_man['IPK'].min()],
    'Max'            : [data_raw['IPK'].max(),    data_mm_man['IPK'].max(),
                        data_zs_man['IPK'].max(), data_ds_man['IPK'].max()],
    'Mean'           : [data_raw['IPK'].mean(),   data_mm_man['IPK'].mean(),
                        data_zs_man['IPK'].mean(),data_ds_man['IPK'].mean()],
    'Std'            : [data_raw['IPK'].std(),    data_mm_man['IPK'].std(),
                        data_zs_man['IPK'].std(), data_ds_man['IPK'].std()],
    'Max |x_norm|'   : [data_raw['IPK'].abs().max(),    data_mm_man['IPK'].abs().max(),
                        data_zs_man['IPK'].abs().max(), data_ds_man['IPK'].abs().max()],
}).round(4)
print(stat.to_string(index=False))
```
 
```{code-cell} ipython3
# ── Visualisasi perbandingan ────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
fig.suptitle('Perbandingan Metode Normalisasi — Kolom IPK',
             fontsize=13, fontweight='bold', y=1.02)
 
datasets = [
    ('Data Asli',       data_raw['IPK'],     '#4e79a7'),
    ('Min-Max',         data_mm_man['IPK'],  '#f28e2b'),
    ('Z-Score',         data_zs_man['IPK'],  '#e15759'),
    ('Decimal Scaling', data_ds_man['IPK'],  '#59a14f'),
]
labels = list(data_raw.index)
 
for ax, (nama, vals, warna) in zip(axes, datasets):
    bars = ax.bar(labels, vals, color=warna, alpha=0.85,
                  edgecolor='white', linewidth=1.2)
    ax.set_title(nama, fontweight='bold', fontsize=11)
    ax.set_xlabel('Data')
    ax.set_ylabel('Nilai')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=9)
    rng = float(vals.max()) - float(vals.min())
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                float(v) + rng * 0.04,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
 
plt.tight_layout()
plt.savefig('perbandingan_normalisasi.png', dpi=120, bbox_inches='tight')
plt.show()
print("Grafik tersimpan: perbandingan_normalisasi.png")
```
![Perbandingan Normalisasi — Kolom IPK](perbandingan_normalisasi.png) 

```{code-cell} ipython3
# ── Visualisasi semua kolom (heatmap perbandingan) ──────────
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
fig.suptitle('Perbandingan Normalisasi — Semua Kolom',
             fontsize=13, fontweight='bold', y=1.01)
 
kolom_list  = ['IPK', 'PO', 'JML_SKS']
metode_dict = {
    'Data Asli'       : data_raw,
    'Min-Max'         : data_mm_man,
    'Z-Score'         : data_zs_man,
    'Decimal Scaling' : data_ds_man,
}
warna_dict = {
    'Data Asli'       : '#4e79a7',
    'Min-Max'         : '#f28e2b',
    'Z-Score'         : '#e15759',
    'Decimal Scaling' : '#59a14f',
}
 
for row_i, kolom in enumerate(kolom_list):
    for col_i, (metode, data) in enumerate(metode_dict.items()):
        ax    = axes[row_i][col_i]
        vals  = data[kolom]
        bars  = ax.bar(labels, vals, color=warna_dict[metode],
                       alpha=0.85, edgecolor='white', linewidth=1)
        ax.set_title(f'{metode}\n({kolom})', fontsize=9, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('Data', fontsize=8)
 
plt.tight_layout()
plt.savefig('perbandingan_semua_kolom.png', dpi=120, bbox_inches='tight')
plt.show()
print("Grafik tersimpan: perbandingan_semua_kolom.png")
```
![Perbandingan Normalisasi — Semua Kolom](perbandingan_semua_kolom.png) 

---
 
### 3.8 Panduan Memilih Metode
 
```{code-cell} ipython3
panduan = pd.DataFrame({
    'Metode'          : ['Min-Max', 'Z-Score', 'Decimal Scaling'],
    'Rumus'           : ['(x-min)/(max-min)', '(x-mean)/std', 'x / 10^j'],
    'Output Range'    : ['[0, 1]', '~[-3, 3]', '< 1 (absolut)'],
    'Cocok Untuk'     : [
        'Data tanpa outlier ekstrim, butuh range tetap [0,1]',
        'Data berdistribusi normal (Gaussian)',
        'Data dengan magnitude besar, konversi desimal sederhana'
    ],
    'Kelemahan'       : [
        'Sensitif terhadap outlier',
        'Output tidak berbatas tetap',
        'Tidak mempertimbangkan distribusi data'
    ]
})
 
print("PANDUAN MEMILIH METODE NORMALISASI:")
print("=" * 75)
for _, row in panduan.iterrows():
    print(f"\n  Metode  : {row['Metode']}")
    print(f"  Rumus   : {row['Rumus']}")
    print(f"  Output  : {row['Output Range']}")
    print(f"  Cocok   : {row['Cocok Untuk']}")
    print(f"  Lemah   : {row['Kelemahan']}")
    print("  " + "-"*60)
```
 
---
 
## Ringkasan
 
| Topik | Poin Utama |
|---|---|
| **WKNN** | Tetangga mirip diberi bobot lebih besar saat imputasi |
| **Same-Class** | Filter tetangga satu kelas untuk estimasi lebih akurat |
| **Min-Max** | Skalakan ke $[0,1]$ — sensitif terhadap outlier |
| **Z-Score** | Standarisasi ke mean=0, std=1 — cocok distribusi normal |
| **Decimal Scaling** | Bagi dengan $10^j$ — sederhana, cocok data berskala besar |
 
**Catatan penting:** Selalu hitung parameter normalisasi dari **data training saja**, lalu terapkan ke data testing untuk menghindari *data leakage*.
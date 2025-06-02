import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("📊 Dashboard Analisis Harga Handphone & Segmentasi Pasar (Semarang)")

# ======================================
# LOAD DATA
# Pastikan file CSV sudah disiapkan dengan benar
# ======================================
df_produk = pd.read_csv("dataDashboardPhabletSample.csv")  # Data produk level
df_toko = pd.read_csv("dataDashboardTokoAggregated.csv")   # Data toko level
unique_shopnames = df_produk['SHOPNAME'].drop_duplicates().reset_index(drop=True)
df_toko['SHOPNAME'] = unique_shopnames

# Pastikan kolom 'segment' ada di df_produk (hasil mapping cluster)
if 'segment' not in df_produk.columns and 'cluster_label' in df_produk.columns:
    cluster_to_segment = {
        0: 'Entry Level',
        1: 'Mid-Range',
        2: 'Flagship'
    }
    df_produk['segment'] = df_produk['cluster_label'].map(cluster_to_segment)

# Pastikan kolom 'SHOPSIZE_PRED' (hasil prediksi Logistic Regression) ada di df_toko
if 'SHOPSIZE_PRED' not in df_toko.columns and 'SHOPSIZE_ENC_PRED' in df_toko.columns:
    # Map label numerik ke nama kategori toko (sesuaikan dengan mapping kamu)
    label_map_toko = {
        0: 'LARGE',
        1: 'MEDIUM',
        2: 'SMALL',
        3: 'XLARGE'
    }
    df_toko['SHOPSIZE_PRED'] = df_toko['SHOPSIZE_ENC_PRED'].map(label_map_toko)

# ======================================
# Sidebar: Pilih View Dashboard
# ======================================
view_option = st.sidebar.selectbox("Pilih Tampilan Dashboard", ["Produk (Segmentasi Pasar)", "Toko (Prediksi Kategori)"])

# ========================
# DASHBOARD PRODUK
# ========================
if view_option == "Produk (Segmentasi Pasar)":
    st.header("📦 Dashboard Produk - Segmentasi Pasar")
    # Filter Brand
    brands = df_produk['BRAND'].unique().tolist()
    brands.sort()
    brands.insert(0, "Semua")
    selected_brand = st.sidebar.selectbox("Filter Merek", brands)

    # Filter Segmen
    segments = df_produk['segment'].unique().tolist()
    segments.sort()
    segments.insert(0, "Semua")
    selected_segment = st.sidebar.selectbox("Filter Segmen Pasar", segments)

    # Filter data
    df_filtered = df_produk.copy()
    if selected_brand != "Semua":
        df_filtered = df_filtered[df_filtered['BRAND'] == selected_brand]
    if selected_segment != "Semua":
        df_filtered = df_filtered[df_filtered['segment'] == selected_segment]

    # Tampilkan data terfilter
    st.subheader(f"Data Produk (Merek: {selected_brand}, Segmen: {selected_segment})")
    st.dataframe(df_filtered[['PRODUCT', 'SHOPNAME', 'BRAND','MODEL', 'PRICE', 'RAM', 'ROM', 'segment']].head(14224))

    # Visualisasi boxplot harga per segmen
    st.subheader("Distribusi Harga per Segmen Pasar")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df_filtered, x='segment', y='PRICE', palette='pastel', ax=ax1)
    ax1.set_xlabel("Segmen Pasar")
    ax1.set_ylabel("Harga (PRICE)")
    st.pyplot(fig1)

    # Visualisasi sebaran RAM vs ROM berdasarkan segmen
    st.subheader("Sebaran Produk berdasarkan RAM dan ROM dengan Segmen")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df_filtered, x='RAM', y='ROM', hue='segment', palette='Set2', ax=ax2)
    ax2.set_xlabel("RAM (GB)")
    ax2.set_ylabel("ROM (GB)")
    ax2.set_title("Sebaran Produk berdasarkan RAM, ROM & Segmen Pasar")
    st.pyplot(fig2)

    # Ringkasan statistik harga dan spesifikasi per segmen
    st.subheader("Rangkuman Statistik per Segmen Pasar")
    summary_segmen = df_produk.groupby('segment').agg({
        'PRICE': ['mean', 'min', 'max', 'median'],
        'RAM': ['mean', 'min', 'max'],
        'ROM': ['mean', 'min', 'max']
    }).round(2)
    st.dataframe(summary_segmen)

# ========================
# DASHBOARD TOKO
# ========================
elif view_option == "Toko (Prediksi Kategori)":
    st.header("🏬 Dashboard Toko - Prediksi Kategori Toko")

    # Filter kategori toko
    toko_categories = df_toko['SHOPSIZE'].unique().tolist()
    toko_categories.sort()
    toko_categories.insert(0, "Semua")
    selected_kategori = st.sidebar.selectbox("Filter Kategori Toko", toko_categories)

    df_toko_filtered = df_toko.copy()
    if selected_kategori != "Semua":
        df_toko_filtered = df_toko_filtered[df_toko_filtered['SHOPSIZE'] == selected_kategori]

    # Tampilkan data toko
    st.subheader(f"Data Toko - Kategori: {selected_kategori}")
    st.dataframe(df_toko_filtered[[ 'SHOPNAME','SHOPSIZE','APPLE', 'EVERCOSS/CROSS', 'INFINIX', 'IQOO', 'ITEL', 'LUNA', 'NUBIA',
    'OPPO', 'POCO', 'REALME', 'REDMI', 'SAMSUNG', 'TECNO', 'VIVO', 'XIAOMI', 'ZTE']].head(50))

    filtered_produk_df = df_produk[df_produk['SHOPNAME'].isin(df_toko_filtered['SHOPNAME'])]

    # Visualisasi rata-rata fitur per kategori toko
    st.subheader("Rata-rata Fitur per Kategori Toko")
    summary_toko = filtered_produk_df.groupby('SHOPNAME').agg({
            'SALES': ['mean', 'min', 'max'],
            'PRICE': ['mean', 'min', 'max'],
            'RAM': ['mean', 'min', 'max'],
            'ROM': ['mean', 'min', 'max'],
        }).round(2)

    st.dataframe(summary_toko)

    # # Jika ada data PCA dan prediksi label cluster toko, bisa tambahkan evaluasi clustering di sini
    # if 'PCA1' in df_toko.columns and 'PCA2' in df_toko.columns and 'cluster_label' in df_toko.columns:
    #     st.subheader("Evaluasi Clustering Toko (Jika Tersedia)")
    #     sil_score = silhouette_score(df_toko[['PCA1', 'PCA2']], df_toko['cluster_label'])
    #     st.success(f"Silhouette Score: {sil_score:.4f}")


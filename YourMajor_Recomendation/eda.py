import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

sns.set_style('whitegrid')

# Path ke model pipeline
PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'your_major_recomendation_pipeline.pkl')
ASSET_DIR = os.path.dirname(os.path.abspath(_file_))

@st.cache_resource
def load_model():
    df = joblib.load(PIPELINE_PATH)['dataset_lengkap']
    return df

def run():
    st.title('📊 Exploratory Data Analysis (EDA)')
    st.markdown('''
    Halaman ini menampilkan eksplorasi data dari *86.569 siswa UTBK 2019 Saintek*
    dengan *279 jurusan* dan *7 kategori bidang*.
    ''')
    st.markdown('---')

    df = load_model()
    total = len(df)
    nilai_cols = ['nilai_biologi', 'nilai_fisika', 'nilai_kimia', 'nilai_matematika',
                  'nilai_kmb', 'nilai_kpu', 'nilai_kua', 'nilai_ppu']

    # ===================================================================
    # 1. DISTRIBUSI SISWA PER BIDANG (Kategori Jurusan)
    # ===================================================================
    st.subheader('1️⃣ Distribusi Siswa per Bidang')

    col_img, _ = st.columns([3, 1])
    with col_img:
        st.image(os.path.join(ASSET_DIR, 'kategori_jurusan.jpeg'),
                 caption='Kategori Jurusan — Distribusi siswa per bidang (Saintek)',
                 use_container_width=True)
    st.markdown('')

    # Statistik
    cat_dist = df['kategori_jurusan'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('*Distribusi siswa:*')
        for cat, cnt in cat_dist.items():
            st.markdown(f'- *{cat}*: {cnt:,} ({cnt/total*100:.2f}%)')
    with col2:
        st.markdown(f'*Total: {total:,} siswa*')
        st.markdown(f'*Domain: Saintek (IPA)*')
        st.markdown(f'*Tahun: 2019*')

    st.markdown("""
    > 💡 *Insight:* *Teknik* (29,12%) mendominasi diikuti *Kesehatan* (26,04%) dan *Science* (22,38%).
    > Ketiganya mencakup *77,54%* dari total siswa. Sementara *Sosial* (2,92%) paling sedikit diminati —
    > wajar karena dataset ini khusus Saintek.
    """)

    st.markdown('---')

    # ===================================================================
    # 2. RATA-RATA NILAI KESELURUHAN (AVG Nilai-Nilai)
    # ===================================================================
    st.subheader('2️⃣ Rata-rata Nilai Keseluruhan (86.569 Siswa)')

    col_img, _ = st.columns([3, 1])
    with col_img:
        st.image(os.path.join(ASSET_DIR, 'avg_nilai_nilai.jpeg'),
                 caption='AVG Nilai-Nilai — Rata-rata 8 mata uji seluruh siswa',
                 use_container_width=True)
    st.markdown('')

    # Tabel pendukung
    avg_values = df[nilai_cols].mean().sort_values(ascending=False)
    avg_list = list(avg_values.items())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('*Rata-rata per Mata Uji (descending):*')
        for i, (col, val) in enumerate(avg_list, 1):
            subject = col.replace('nilai_', '').upper()
            st.markdown(f'{i:>2}. *{subject}*: {val:.2f}')
    with col2:
        st.markdown('*Highlight:*')
        st.markdown(f'📈 *Tertinggi*: {avg_list[0][0].replace("nilai_", "").upper()} ({avg_list[0][1]:.2f})')
        st.markdown(f'📉 *Terendah*: {avg_list[-1][0].replace("nilai_", "").upper()} ({avg_list[-1][1]:.2f})')
        st.markdown(f'📊 *Rata-rata total*: {avg_values.mean():.2f}')

    st.markdown("""
    > 💡 *Insight:* Nilai *KPU* (569,94) dan *KUA* (569,15) konsisten paling tinggi,
    > sementara *Matematika* (529,49) dan *Biologi* (537,14) paling rendah.
    > Ini menarik karena Matematika sering dianggap mata uji tersulit di Saintek.
    """)

    st.markdown('---')

    # ===================================================================
    # 3. BOX PLOT — SEBARAN NILAI PER MATA UJI
    # ===================================================================
    st.subheader('3️⃣ Sebaran Nilai per Mata Uji (Box Plot)')

    col_img, _ = st.columns([3, 1])
    with col_img:
        st.image(os.path.join(ASSET_DIR, 'box_plot.jpeg'),
                 caption='Box Plot Outlier Numeric — Sebaran nilai per mata uji',
                 use_container_width=True)
    st.markdown('')

    # Stats table
    nama_nilai = ['Nilai Biologi', 'Nilai Fisika', 'Nilai Kimia', 'Nilai Matematika',
                  'Nilai KMB', 'Nilai KPU', 'Nilai KUA', 'Nilai PPU']
    stats_df = df[nilai_cols].describe().T
    stats_df.index = nama_nilai
    stats_df = stats_df[['min', '25%', '50%', '75%', 'max', 'mean', 'std']]
    stats_df = stats_df.round(1)
    stats_df.columns = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean', 'Std']
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("""
    > 💡 *Insight:* Nilai maksimum mencapai *1.123* (Matematika — kemungkinan nilai bonus/eksperimen),
    > sementara nilai minimum ada di *193* (KPU). Box plot menunjukkan distribusi cukup simetris
    > dengan median di kisaran 520-570. Banyak outlier di atas, tapi jarang di bawah — artinya
    > siswa cenderung punya nilai tinggi di beberapa mata uji tertentu.
    """)

    st.markdown('---')

    # ===================================================================
    # 4. HEATMAP — RATA-RATA NILAI PER KATEGORI
    # ===================================================================
    st.subheader('4️⃣ Rata-rata Nilai per Bidang (Heatmap)')

    col_img, _ = st.columns([3, 1])
    with col_img:
        st.image(os.path.join(ASSET_DIR, 'heatmap_kategori.jpeg'),
                 caption='AVG Nilai Pada Kategori Jurusan — Rata-rata nilai per bidang',
                 use_container_width=True)
    st.markdown('')

    # Data pendukung
    cols_short = ['Bio', 'Fis', 'Kim', 'Mat', 'KMB', 'KPU', 'KUA', 'PPU']
    avg_df = df.groupby('kategori_jurusan')[nilai_cols].mean().round(2)
    avg_df.columns = cols_short

    max_val = avg_df.max().max()
    max_loc = avg_df.stack().idxmax()
    min_val = avg_df.min().min()
    min_loc = avg_df.stack().idxmin()

    st.markdown(f"""
    > 💡 *Insight:* *{max_loc[0]}* punya nilai rata-rata tertinggi di *{max_loc[1]}* ({max_val:.2f}),
    > sementara *{min_loc[0]}* punya nilai terendah di *{min_loc[1]}* ({min_val:.2f}).
    > Menariknya, *Pendidikan* konsisten lebih rendah di hampir semua mata uji,
    > sementara *Teknik* dan *Kesehatan* mendominasi nilai-nilai tertinggi.
    """)

    st.markdown('---')

    # ===================================================================
    # 5. TOP JURUSAN PALING DIMINATI
    # ===================================================================
    st.subheader('5️⃣ Jurusan Dengan Minat Terbanyak')

    col_img, _ = st.columns([3, 1])
    with col_img:
        st.image(os.path.join(ASSET_DIR, 'minat_terbanyak.jpeg'),
                 caption='Jurusan Dengan Minat Terbanyak — Top 40 jurusan paling diminati',
                 use_container_width=True)
    st.markdown('')

    # Tabel top 10
    st.markdown('*Top 10 Jurusan Paling Diminati:*')
    top10 = df['jurusan_tujuan'].value_counts().head(10).reset_index()
    top10.columns = ['Jurusan', 'Jumlah Siswa']
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.markdown("""
    > 💡 *Insight:* *PENDIDIKAN DOKTER* (≈6.000 siswa) menjadi jurusan paling diminati —
    > hampir *2x lipat* dari *Teknik Sipil* (#2, ≈3.800). Menariknya, *Kedokteran* (≈1.700)
    > hanya di peringkat #12, menunjukkan siswa lebih memilih *Pendidikan Dokter* yang jenjangnya
    > lebih pendek. Jurusan IT/komputer seperti *Teknik Informatika* (#4, ≈3.000) dan
    > *Sistem Informasi* juga masuk 15 besar.
    """)

    st.markdown('---')
    st.markdown('*© 2026 YourMajor*')

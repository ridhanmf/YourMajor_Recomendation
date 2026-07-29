import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Path ke model pipeline
PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'your_major_recomendation_pipeline.pkl')


@st.cache_resource
def load_model():
    """Load pipeline artefacts"""
    pipe = joblib.load(PIPELINE_PATH)
    return pipe['scaler'], pipe['knn_model'], pipe['nilai_cols'], pipe['dataset_lengkap']


def run():
    st.title('🔮 Prediksi & Rekomendasi Jurusan')
    st.markdown('''
    Masukkan **8 nilai UTBK** kamu, lalu sistem akan mencari **100 siswa paling mirip**
    dari **86.569 data** UTBK 2019 Saintek dan meranking jurusan terbaik untukmu.
    ''')
    st.markdown('---')

    scaler, nn, nilai_cols, df = load_model()

    # Input nilai
    st.subheader('📝 Masukkan Nilai UTBK')

    col1, col2 = st.columns(2)

    with col1:
        nilai_biologi = st.number_input('🟢 Nilai Biologi', min_value=0, max_value=1000, value=600, step=10)
        nilai_fisika = st.number_input('🔵 Nilai Fisika', min_value=0, max_value=1000, value=600, step=10)
        nilai_kimia = st.number_input('🟡 Nilai Kimia', min_value=0, max_value=1000, value=600, step=10)
        nilai_matematika = st.number_input('🔴 Nilai Matematika', min_value=0, max_value=1000, value=600, step=10)

    with col2:
        nilai_kmb = st.number_input('🧠 Nilai KMB', min_value=0, max_value=1000, value=600, step=10)
        nilai_kpu = st.number_input('📊 Nilai KPU', min_value=0, max_value=1000, value=600, step=10)
        nilai_kua = st.number_input('📐 Nilai KUA', min_value=0, max_value=1000, value=600, step=10)
        nilai_ppu = st.number_input('📝 Nilai PPU', min_value=0, max_value=1000, value=600, step=10)

    inputs = [nilai_biologi, nilai_fisika, nilai_kimia, nilai_matematika,
              nilai_kmb, nilai_kpu, nilai_kua, nilai_ppu]

    st.markdown('---')

    if st.button('🎯 Cari Rekomendasi', type='primary', use_container_width=True):
        with st.spinner('Mencari siswa dengan nilai paling mirip...'):

            # Transform input
            vals = np.array(inputs).reshape(1, -1)
            vals_scaled = scaler.transform(vals)

            # Cari tetangga
            distances, indices = nn.kneighbors(vals_scaled)
            neighbors = df.iloc[indices[0]].copy()
            neighbors['_distance'] = distances[0]

            # Ranking jurusan
            jurusan_rank = neighbors['jurusan_tujuan'].value_counts().head(10)
            total = len(neighbors)

            # Distribusi kategori
            kategori_dist = neighbors['kategori_jurusan'].value_counts()
            kategori_dominan = kategori_dist.index[0]
            kategori_pct = kategori_dist.iloc[0] / total * 100

            # Top kategori
            top_kategori = [(cat, round(cnt / total * 100, 1))
                           for cat, cnt in kategori_dist.head(4).items()]

            # === TAMPILKAN HASIL ===
            st.success('✅ Rekomendasi ditemukan!')

            # Baris 1: Bidang
            st.subheader('📊 Bidang yang Direkomendasikan')

            col_cat = st.columns(len(top_kategori))
            for i, (cat, pct) in enumerate(top_kategori):
                with col_cat[i]:
                    if cat == kategori_dominan:
                        st.markdown(f"**🟢 {cat}**")
                    else:
                        st.markdown(f"⬜ {cat}")
                    st.progress(pct / 100, text=f'{pct:.0f}%')

            st.info(f'➡️ **Bidang terpilih: {kategori_dominan}** ({kategori_pct:.0f}% dari 100 tetangga terdekat)')

            # Baris 2: Rekomendasi Jurusan
            st.subheader('🏫 Rekomendasi Jurusan Terbaik')

            # Tampilkan 5 besar
            for rank, (jurusan, cnt) in enumerate(jurusan_rank.items(), start=1):
                pct = cnt / total * 100
                if rank == 1:
                    icon = '🥇'
                elif rank == 2:
                    icon = '🥈'
                elif rank == 3:
                    icon = '🥉'
                else:
                    icon = f'{rank}.'

                st.markdown(f'{icon} **{jurusan}** — {cnt} siswa ({pct:.1f}%)')

            # Detail
            with st.expander('📋 Informasi Tambahan'):
                st.markdown('**Nilai Kamu:**')
                cols_short = ['Bio', 'Fis', 'Kim', 'Mat', 'KMB', 'KPU', 'KUA', 'PPU']
                for c, v in zip(cols_short, inputs):
                    st.markdown(f'- {c}: {v}')

                st.markdown('')
                st.markdown('**Distribusi Bidang dari 100 Tetangga Terdekat:**')
                for cat, cnt in kategori_dist.items():
                    pct = cnt / total * 100
                    bar = '🟢' if cat == kategori_dominan else '⬜'
                    st.markdown(f'{bar} {cat}: {cnt} siswa ({pct:.0f}%)')

                st.markdown('')
                st.markdown('**Rata-rata jarak ke tetangga:**')
                st.markdown(f'{distances[0].mean():.0f} poin')

    st.markdown('---')
    st.markdown('**© 2026 Muhammad Izzat — Final Project**')

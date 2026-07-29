import sys
import os

# Tambah path biar bisa import eda.py & prediction.py dari folder yang sama
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import eda
import prediction


st.set_page_config(
    page_title='Your Major Recommendation',
    page_icon='🎓',
    layout='wide'
)


def main():
    st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">🎓 Your Major Recommendation</div>', unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        'Pilih Halaman',
        ('Home', 'EDA', 'Prediksi')
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Final Project**')
    st.sidebar.markdown('Rekomendasi Jurusan & Universitas')
    st.sidebar.markdown('Berdasarkan Nilai UTBK 2019 Saintek')
    st.sidebar.markdown('')
    st.sidebar.markdown('by **Muhammad Izzat**')

    if page == 'Home':
        show_home()
    elif page == 'EDA':
        eda.run()
    elif page == 'Prediksi':
        prediction.run()


def show_home():
    st.title('🎓 Your Major Recommendation')
    st.markdown('---')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Selamat Datang!

        Aplikasi ini membantu calon mahasiswa menemukan **jurusan dan universitas**
        yang paling sesuai dengan **nilai UTBK** mereka.

        ### Cara Kerja

        Aplikasi menggunakan pendekatan **K-Nearest Neighbors** berdasarkan nilai UTBK:

        - **Input** 8 nilai UTBK
        - **Cari** 100 siswa paling mirip dari 86.569 data UTBK 2019 Saintek
        - **Ranking** jurusan berdasarkan kemiripan

        ### Cara Pakai

        1. Buka halaman **EDA** untuk lihat eksplorasi dataset
        2. Buka halaman **Prediksi** untuk masukkan nilai dan lihat rekomendasi
        """)

    with col2:
        st.markdown("""
        **Info Dataset**

        | Item | Detail |
        |:-----|:------:|
        | **Sumber** | UTBK 2019 Saintek |
        | **Siswa** | 86.569 |
        | **Nilai** | 8 mata uji |
        | **Jurusan** | 279 pilihan |
        | **Bidang** | 7 kategori |

        **Teknologi**

        ⚡ K-Nearest Neighbors
        📊 Streamlit
        🐍 Python
        """)

    st.markdown('---')
    st.markdown('**© 2026 Muhammad Izzat — Final Project**')


if __name__ == '__main__':
    main()

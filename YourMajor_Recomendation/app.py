import sys
import os

# Tambah path biar bisa import eda.py & prediction.py dari folder yang sama
sys.path.insert(0, os.path.dirname(os.path.abspath(_file_)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import eda
import prediction
from PIL import Image

ASSET_DIR = os.path.dirname(os.path.abspath(_file_))

icon = Image.open('logo_removebg.png')
st.set_page_config(
    page_title='YourMajor Recommendation',
    page_icon='icon',
    layout='wide'
)

def main():
    # Logo
    st.sidebar.image('logo_removebg.png', use_container_width=True)
    st.sidebar.markdown('---')

    page = st.sidebar.selectbox(
        'Pilih Halaman',
        ('Home', 'EDA', 'Prediksi')
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('Rekomendasi Jurusan & Universitas')
    st.sidebar.markdown('Berdasarkan Nilai UTBK 2019 Saintek')
    st.sidebar.markdown('')
    st.sidebar.markdown("""
*Created by:*
* Muhammad Izzat
* Ridhan Firdaus
* Nicholas Calvin
""")

    if page == 'Home':
        show_home()
    elif page == 'EDA':
        eda.run()
    elif page == 'Prediksi':
        prediction.run()


def show_home():
    st.title('🎓 Your Major Recommendation')
    st.image('gambar_kampus.jpeg')
    st.markdown('---')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Welcome!

        Aplikasi ini membantu calon mahasiswa menemukan *jurusan dan universitas*
        yang paling sesuai dengan *nilai UTBK* mereka.

        ### Workflow

        Aplikasi menggunakan pendekatan *K-Nearest Neighbors* berdasarkan nilai UTBK:

        - *Input* 8 nilai UTBK
        - *Cari* 100 siswa paling mirip dari 86.569 data UTBK 2019 Saintek
        - *Ranking* jurusan berdasarkan kemiripan

        ### Pages
        
        1. *Home* Beranda & Informasi Project
        1. *EDA* Eksplorasi & Visualisasi Data
        2. *Prediksi* Rekomendasi Jurusan
        """)

    with col2:
        st.markdown("""
        *Info Dataset*

        | Item | Detail |
        |:-----|:------:|
        | *Sumber* | UTBK 2019 Saintek |
        | *Siswa* | 86.569 |
        | *Nilai* | 8 mata uji |
        | *Jurusan* | 279 pilihan |
        | *Bidang* | 7 kategori |

        """)

    st.markdown('---')
    st.markdown('*© 2026 YourMajor*')


if _name_ == '_main_':
    main()

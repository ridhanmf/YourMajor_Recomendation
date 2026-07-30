import sys
import os

# Tambah path biar bisa import eda.py & prediction.py dari folder yang sama
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import eda
import prediction
from PIL import Image

# =========================
# Path Asset
# =========================
LOGO_PATH = os.path.join(BASE_DIR, "logo_removebg.png")
KAMPUS_PATH = os.path.join(BASE_DIR, "gambar_kampus.jpeg")

# =========================
# Page Config
# =========================
icon = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="YourMajor Recommendation",
    page_icon=icon,
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>

/* =========================
   SIDEBAR
========================= */

/* Background sidebar */
section[data-testid="stSidebar"]{
    background-color: white;
}

/* Semua teks sidebar menjadi hitam */
section[data-testid="stSidebar"] *{
    color: black !important;
}

/* Garis pemisah */
section[data-testid="stSidebar"] hr{
    border-color: #D3D3D3;
}

/* =========================
   SELECTBOX
========================= */

/* Kotak selectbox */
div[data-baseweb="select"] > div{
    background-color: white !important;
    color: black !important;
    border: 1px solid #D3D3D3 !important;
}

/* Tulisan di dalam selectbox */
div[data-baseweb="select"] span{
    color: black !important;
}

/* Menu dropdown */
div[data-baseweb="popover"]{
    background-color: white !important;
}

/* Tulisan pada menu dropdown */
div[data-baseweb="popover"] *{
    color: black !important;
}

/* =========================
   MAIN PAGE
========================= */

/* Background tetap hitam */
[data-testid="stAppViewContainer"]{
    background-color:#0E1117;
}

/* Tulisan halaman utama tetap putih */
[data-testid="stAppViewContainer"] *{
    color:white;
}

/* Tabel tetap putih */
table{
    color:white !important;
}

/* Link */
a{
    color:#8ab4f8 !important;
}

</style>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

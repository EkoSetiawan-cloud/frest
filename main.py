import streamlit as st
from input_data_pnbp import input_data_page
from aggregate_data_pnbp import aggregate_data_page
from clean_data_pnbp import clean_data_page
from split_data_pnbp import split_data_page
from ets_rolling_eval_pnbp import ets_rolling_eval_page
from residual_analysis_pnbp import residual_analysis_page

st.set_page_config(page_title="PREDIKSI PNBP", layout="wide")
st.sidebar.title("Prediksi PNBP ETS Rolling Forecast")

steps = [
    "1. Input Data",
    "2. Agregasi Data",
    "3. Bersih Data",
    "4. Split Train-Test",
    "5. Exponential Smoothing (ETS) Rolling Forecast",
    "6. Residual Analysis"
]
step = st.sidebar.radio("Pilih langkah:", steps)

if step == "1. Input Data":
    input_data_page()
elif step == "2. Agregasi Data":
    aggregate_data_page()
elif step == "3. Bersih Data":
    clean_data_page()
elif step == "4. Split Train-Test":
    split_data_page()
elif step == "5. Exponential Smoothing (ETS) Rolling Forecast":
    ets_rolling_eval_page()
elif step == "6. Residual Analysis":
    residual_analysis_page()

st.sidebar.markdown("---")
st.sidebar.info("Jalankan modul berurutan untuk hasil optimal.")

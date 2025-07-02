import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def residual_analysis_page():
    st.header("Residual Analysis (Analisis Error Prediksi)")

    # Cek apakah hasil rolling forecast ARIMA/ETS ada
    opsi = []
    if 'rolling_eval_result_arima' in st.session_state:
        opsi.append("ARIMA")
    if 'rolling_eval_result_ets' in st.session_state:
        opsi.append("ETS")
    if not opsi:
        st.warning("Belum ada hasil rolling forecast ARIMA/ETS yang bisa dianalisis.")
        return

    # Dropdown pilih model
    model_name = st.selectbox("Pilih model untuk residual analysis:", opsi)

    # Ambil hasil rolling sesuai model
    if model_name == "ARIMA":
        df_rolling = st.session_state['rolling_eval_result_arima']
    else:
        df_rolling = st.session_state['rolling_eval_result_ets']

    # Tampilkan debug
    st.write("Preview data residual (5 baris):")
    st.write(df_rolling.head())

    if df_rolling is not None and not df_rolling.empty:
        actuals = df_rolling['Actual'].astype(float)
        predictions = df_rolling['Forecast'].astype(float)
        residuals = predictions - actuals
        residuals_clean = residuals.dropna()

        st.success(f"Residual yang dianalisis berasal dari model: **{model_name}**")

        if len(residuals_clean) == 0:
            st.warning("Semua residual NaN. Model gagal fit di rolling forecast, atau parameter tidak cocok.")
            return

        st.subheader("Residual (Prediksi - Aktual)")
        st.line_chart(residuals_clean.values)

        st.subheader("Density Plot of Residuals")
        fig2, ax2 = plt.subplots()
        pd.Series(residuals_clean).plot(kind='kde', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Statistik Residual:")
        st.write(pd.Series(residuals_clean).describe())

        st.subheader("Autocorrelation Plot (ACF) of Residuals")
        fig3, ax3 = plt.subplots()
        plot_acf(residuals_clean, lags=min(8, max(1, len(residuals_clean)//2-1)), ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("Data rolling forecast tidak ditemukan atau kosong untuk model ini. Jalankan rolling forecast yang sesuai.")

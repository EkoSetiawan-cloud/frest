import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def residual_analysis_page():
    st.header("🔎 6️⃣ Residual Analysis (Analisis Error Prediksi)")

    st.markdown("""
    <div style="background-color:#F3E5F5; padding:14px; border-radius:10px; margin-bottom:10px;">
    <b>Apa itu Residual Analysis?</b><br>
    Residual adalah selisih antara nilai prediksi dan aktual. Analisis residual sangat penting untuk mengecek apakah error prediksi sudah acak (white noise), atau masih ada pola tertentu yang menandakan model perlu diperbaiki.
    </div>
    """, unsafe_allow_html=True)

    opsi = []
    if 'rolling_eval_result_ets' in st.session_state:
        opsi.append("ETS")
    if not opsi:
        st.warning("Belum ada hasil rolling forecast ETS yang bisa dianalisis. Selesaikan prediksi di langkah sebelumnya.")
        return

    model_name = st.selectbox("Pilih model untuk residual analysis:", opsi, index=0)

    # Ambil hasil rolling
    if model_name == "ETS":
        df_rolling = st.session_state['rolling_eval_result_ets']
    else:
        st.warning("Hanya model ETS yang tersedia.")
        return

    st.markdown("#### 👁️ Preview Data Residual")
    st.dataframe(df_rolling)

    if df_rolling is not None and not df_rolling.empty:
        actuals = df_rolling['Actual'].astype(float)
        predictions = df_rolling['Forecast'].astype(float)
        residuals = predictions - actuals
        residuals_clean = residuals.dropna()

        st.success(f"Analisis residual berdasarkan model: **{model_name}**")

        # Tabel & Grafik Residual
        st.markdown("#### 📈 Residual (Forecast - Actual) per Tahun")
        df_resid = df_rolling[["Tahun"]].copy()
        df_resid["Residual"] = residuals
        st.dataframe(df_resid)

        fig, ax = plt.subplots()
        ax.plot(df_resid["Tahun"], df_resid["Residual"], marker="o", linestyle='-', color="#7E57C2")
        ax.axhline(0, ls="--", color="gray", lw=1)
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Residual")
        ax.set_title("Residual (Forecast - Actual) per Tahun")
        st.pyplot(fig)

        # Deteksi outlier & bias
        max_resid = residuals_clean.abs().max()
        mean_resid = residuals_clean.mean()
        n_outlier = (residuals_clean.abs() > 2*residuals_clean.std()).sum()
        bias_info = "Negatif" if mean_resid < -0.01*abs(residuals_clean.max()) else ("Positif" if mean_resid > 0.01*abs(residuals_clean.max()) else "Netral")

        st.markdown(f"""
            <div style='background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:8px;'>
            <b>Statistik Residual:</b><br>
            Rata-rata: <b>{mean_resid:,.2f}</b> ({bias_info})<br>
            Std Deviasi: <b>{residuals_clean.std():,.2f}</b><br>
            Max Outlier: <b>{max_resid:,.2f}</b><br>
            Jumlah Outlier (|residual| > 2 std): <b>{n_outlier}</b>
            </div>
        """, unsafe_allow_html=True)

        # Density Plot Residuals
        st.markdown("#### 🟣 Density Plot of Residuals")
        fig2, ax2 = plt.subplots()
        pd.Series(residuals_clean).plot(kind='kde', ax=ax2, color="#7E57C2")
        ax2.axvline(0, color="gray", ls="--", lw=1)
        st.pyplot(fig2)
        st.markdown(f"""
            <div style='color:gray; font-size:0.95rem;'>
            <i>Distribusi residual harus mendekati normal (lonceng). Jika tidak simetris atau banyak outlier, model bisa diperbaiki lagi.</i>
            </div>
        """, unsafe_allow_html=True)

        # ACF Plot Residuals
        st.markdown("#### 🔗 Autocorrelation Plot (ACF) of Residuals")
        fig3, ax3 = plt.subplots()
        plot_acf(residuals_clean, lags=min(8, max(1, len(residuals_clean)//2-1)), ax=ax3, color="#7E57C2")
        st.pyplot(fig3)
        st.markdown(f"""
            <div style='color:gray; font-size:0.95rem;'>
            <i>Jika lag ACF berada di dalam batas biru, residual sudah acak (white noise). Jika ada lag di luar, berarti error masih berkorelasi.</i>
            </div>
        """, unsafe_allow_html=True)

        # Interpretasi otomatis
        if abs(mean_resid) > 0.1 * residuals_clean.std():
            st.warning("⚠️ Rata-rata residual tidak nol, model cenderung bias.")
        if n_outlier > 0:
            st.info(f"ℹ️ Ada {n_outlier} outlier pada residual. Periksa kemungkinan data anomali.")
        if len(residuals_clean) > 3 and (pd.Series(residuals_clean).autocorr(lag=1) > 0.5):
            st.warning("⚠️ Residual saling berkorelasi (tidak white noise). Model perlu di-improve atau cek transformasi data.")

        st.success("✅ Analisis residual selesai. Hasil sudah bisa dipakai untuk interpretasi & improvement model.")
    else:
        st.warning("Data rolling forecast tidak ditemukan atau kosong untuk model ini.")

    st.markdown("""
    ---
    <span style="color:gray; font-size:14px;">
    <b>Tips:</b> Residual yang baik = acak, tidak bias, tidak banyak outlier. Jika tidak, coba tuning parameter model atau cek kualitas data.
    </span>
    """, unsafe_allow_html=True)

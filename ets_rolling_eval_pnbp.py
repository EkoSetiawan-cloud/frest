import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def ets_rolling_eval_page():
    st.header("🔵 5️⃣ Exponential Smoothing (ETS) Rolling Forecast & Evaluasi")
    st.markdown("""
    <div style="background-color:#EDE7F6; padding:14px; border-radius:10px; margin-bottom:10px;">
    <b>Deskripsi:</b><br>
    <b>Exponential Smoothing (ETS)</b> adalah salah satu metode time series forecasting yang mampu menangkap pola trend dan musiman.<br>
    <b>Rolling Forecast</b> berarti model selalu di-update tiap tahun dengan data terbaru agar evaluasi lebih realistis.
    </div>
    """, unsafe_allow_html=True)

    df_train = st.session_state.get('train_df')
    df_test = st.session_state.get('test_df')
    if df_train is not None and df_test is not None and not df_train.empty and not df_test.empty:
        with st.expander("ℹ️ Penjelasan Pilihan Model ETS", expanded=False):
            st.markdown("""
            <ul>
                <li><b>Tipe Trend</b>: <i>add</i> = trend naik/turun linear, <i>mul</i> = pertumbuhan relatif/persen, <i>None</i> = tanpa trend</li>
                <li><b>Tipe Seasonal</b>: <i>add</i> = musiman tetap, <i>mul</i> = musiman proporsional, <i>None</i> = tanpa musiman</li>
            </ul>
            """, unsafe_allow_html=True)

        trend = st.selectbox("Tipe Trend", ["add", "mul", None], index=0)
        seasonal = st.selectbox("Tipe Seasonal", [None, "add", "mul"], index=0)
        st.info(f"Model ETS yang dipilih: trend = {trend}, seasonal = {seasonal}")

        train_series = df_train['Total_PNBP'].astype(float).values
        test_series = df_test['Total_PNBP'].astype(float).values

        history = list(train_series)
        predictions = []
        error_msgs = []
        for t in range(len(test_series)):
            try:
                model = ExponentialSmoothing(
                    history,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=None
                )
                model_fit = model.fit(optimized=True)
                yhat = model_fit.forecast(1)[0]
            except Exception as e:
                yhat = np.nan
                error_msgs.append(f"Tahun {df_test['Tahun'].iloc[t]}: {e}")
            predictions.append(yhat)
            history.append(test_series[t])

        # Evaluasi
        actuals = test_series
        mae = np.mean(np.abs(np.array(predictions) - actuals))
        rmse = np.sqrt(np.mean((np.array(predictions) - actuals) ** 2))
        mape = np.mean(np.abs((np.array(predictions) - actuals) / actuals)) * 100

        # Penilaian otomatis performa
        if mape < 10:
            cat_perf = "🔵 <b>Sangat Baik</b>"
        elif mape < 20:
            cat_perf = "🟢 <b>Baik</b>"
        elif mape < 30:
            cat_perf = "🟡 <b>Cukup</b>"
        else:
            cat_perf = "🔴 <b>Kurang</b> (perbaiki parameter/model atau cek data)"

        # Tabel hasil rolling
        df_eval = pd.DataFrame({
            "Tahun": df_test['Tahun'],
            "Actual": actuals,
            "Forecast": predictions
        })
        st.markdown("#### 🗃️ Tabel Hasil Prediksi vs Aktual")
        st.dataframe(df_eval, use_container_width=True)

        # Download tabel prediksi
        csv = df_eval.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_ETS.csv",
            mime="text/csv"
        )

        # Plot
        st.markdown("#### 📊 Visualisasi Prediksi vs Aktual (ETS Rolling Forecast)")
        fig, ax = plt.subplots()
        ax.plot(df_eval["Tahun"], df_eval["Actual"], label="Actual", marker="o")
        ax.plot(df_eval["Tahun"], df_eval["Forecast"], label="ETS Forecast", marker="o")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total PNBP")
        ax.set_title("ETS Forecast vs Actual (Rolling Forecast)")
        ax.legend()
        st.pyplot(fig)

        # Evaluasi metrik
        st.markdown("#### 📦 Evaluasi Hasil Prediksi (ETS)")
        col1, col2, col3, col4 = st.columns([1.5,1.5,1,2])
        col1.markdown(f"""<div style='font-size:1.3rem; font-weight:500; padding-bottom:2px'>{mae:,.2f}</div><div style='color:gray; font-size:0.95rem;'>MAE</div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div style='font-size:1.3rem; font-weight:500; padding-bottom:2px'>{rmse:,.2f}</div><div style='color:gray; font-size:0.95rem;'>RMSE</div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div style='font-size:1.3rem; font-weight:500; padding-bottom:2px'>{mape:.2f}%</div><div style='color:gray; font-size:0.95rem;'>MAPE</div>""", unsafe_allow_html=True)
        col4.markdown(f"<b>Kategori Akurasi:</b><br>{cat_perf}", unsafe_allow_html=True)


        if error_msgs:
            with st.expander("⚠️ Terdapat error pada beberapa langkah rolling:", expanded=False):
                for msg in error_msgs:
                    st.error(msg)

        # SIMPAN MODEL TERAKHIR (ETS)
        st.session_state['rolling_eval_result_ets'] = df_eval

        st.success("✅ Proses rolling forecast ETS selesai. Lanjut ke *Residual Analysis* untuk analisis error prediksi.")
    else:
        st.info("Belum ada data train-test. Selesaikan langkah sebelumnya terlebih dahulu.")

    st.markdown("""
    ---
    <span style="color:gray; font-size:14px;">
    <b>Tips:</b> Coba beberapa kombinasi parameter trend/seasonal untuk hasil optimal. Cek residual error setelah rolling forecast.
    </span>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def ets_rolling_eval_page():
    st.header("🔵 Exponential Smoothing (ETS) Rolling Forecast & Evaluasi")
    df_train = st.session_state.get('train_df')
    df_test = st.session_state.get('test_df')
    if df_train is not None and df_test is not None:
        trend = st.selectbox("Tipe Trend", ["add", "mul", None], index=0)
        seasonal = st.selectbox("Tipe Seasonal", [None, "add", "mul"], index=0)
        st.info(f"Model ETS: trend={trend}, seasonal={seasonal}")

        train_series = df_train['Total_PNBP'].values
        test_series = df_test['Total_PNBP'].values

        history = list(train_series)
        predictions = []
        for t in range(len(test_series)):
            try:
                model = ExponentialSmoothing(history, trend=trend, seasonal=seasonal, seasonal_periods=None)
                model_fit = model.fit(optimized=True)
                yhat = model_fit.forecast(1)[0]
            except Exception as e:
                yhat = np.nan
            predictions.append(yhat)
            history.append(test_series[t])
            st.write(f"Step {t+1}: Tahun {df_test['Tahun'].iloc[t]}, predicted={yhat:.2f}, expected={test_series[t]:.2f}")

        # Evaluasi
        actuals = test_series
        mae = np.mean(np.abs(np.array(predictions) - actuals))
        rmse = np.sqrt(np.mean((np.array(predictions) - actuals) ** 2))
        mape = np.mean(np.abs((np.array(predictions) - actuals) / actuals)) * 100

        st.write("## Evaluasi Hasil Prediksi (ETS)")
        st.write(f"MAE: {mae:,.2f}")
        st.write(f"RMSE: {rmse:,.2f}")
        st.write(f"MAPE: {mape:.2f}%")
        
        df_eval = pd.DataFrame({
            "Tahun": df_test['Tahun'],
            "Actual": actuals,
            "Forecast": predictions
        })
        st.write(df_eval)

        st.session_state['rolling_eval_result_ets'] = df_eval

        # Plot rolling forecast (tanpa prediksi masa depan)
        fig, ax = plt.subplots()
        ax.plot(df_eval["Tahun"], df_eval["Actual"], label="Actual", marker="o")
        ax.plot(df_eval["Tahun"], df_eval["Forecast"], label="ETS Forecast", marker="o")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total PNBP")
        ax.set_title("ETS Forecast vs Actual (Rolling Forecast)")
        ax.legend()
        st.pyplot(fig)

        # ====== Prediksi 1 Tahun ke Depan (Tabel & Grafik Baru) ======
        try:
            full_series = np.concatenate([train_series, test_series])
            final_model = ExponentialSmoothing(full_series, trend=trend, seasonal=seasonal, seasonal_periods=None)
            final_fit = final_model.fit(optimized=True)
            forecast_future = final_fit.forecast(steps=1)
            future_year = [df_test['Tahun'].iloc[-1] + 1]
            df_future = pd.DataFrame({
                "Tahun": future_year,
                "Forecast": forecast_future
            })
        except Exception as e:
            df_future = pd.DataFrame({"Tahun": [], "Forecast": []})

        # Tampilkan tabel prediksi 1 tahun ke depan
        st.subheader("🔮 Forecast one-year periode a head")
        st.write("Prediksi ini menggunakan seluruh data historis hingga tahun terakhir (2024).")
        st.write(df_future)

        # Plot grafik husus prediksi 1 tahun ke depan
        if not df_future.empty:
            fig2, ax2 = plt.subplots()
            ax2.plot(df_future["Tahun"], df_future["Forecast"], color="red", marker="o", linestyle="None", markersize=12, label="Forecast one-year a head")
            # Tambah label angka di atas titik
            for x, y in zip(df_future["Tahun"], df_future["Forecast"]):
                ax2.annotate(f"{int(y):,}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=11, color="red")
            ax2.set_xlabel("Tahun")
            ax2.set_ylabel("Forecast PNBP")
            ax2.set_title(f"Forecast {df_future['Tahun'].iloc[0]}")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.info("Prediksi masa depan tidak tersedia.")

    else:
        st.info("Selesaikan step split train-test dulu.")

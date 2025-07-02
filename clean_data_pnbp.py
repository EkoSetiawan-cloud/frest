import streamlit as st
import pandas as pd

def clean_data_page():
    st.header("3️⃣ Bersihkan Data Total PNBP Tahunan")
    df_total = st.session_state.get('df_total')
    if df_total is not None:
        st.write("Data sebelum bersih:")
        st.dataframe(df_total)
        df_clean = df_total.drop_duplicates(subset='Tahun').dropna()
        st.write("Data setelah bersih:")
        st.dataframe(df_clean)
        st.session_state['df_clean'] = df_clean
        st.success("✅ Data sudah bersih. Lanjut ke split train-test.")
    else:
        st.info("Selesaikan step agregasi dulu.")

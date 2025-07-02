import streamlit as st
import pandas as pd

def input_data_page():
    st.header("1️⃣ Input Data PNBP (Excel)")
    uploaded_file = st.file_uploader("Upload file Dataset-PNBP.xlsx", type="xlsx", key="file1")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name='Tabel_Target_dan_Realisasi_PNBP')
        df = df.astype(str)
        st.dataframe(df.head())
        st.write(f"Jumlah baris: {df.shape[0]}")
        st.write("Kolom:", list(df.columns))
        st.write("Missing value tiap kolom:")
        st.write(df.isnull().sum())
        st.session_state['df_mentah'] = df
        st.success("✅ Data berhasil di-load! Lanjut ke step berikutnya.")
    else:
        st.info("Silakan upload file .xlsx")

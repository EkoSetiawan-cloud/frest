import streamlit as st
import pandas as pd

def aggregate_data_page():
    st.header("2️⃣ Agregasi Total PNBP per Tahun")
    df = st.session_state.get('df_mentah')
    if df is not None:
        year_columns = [col for col in df.columns if col != 'Jenis PNBP']
        for col in year_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        total_pnbp = df[year_columns].sum(axis=0)
        df_total = pd.DataFrame({
            'Tahun': total_pnbp.index.astype(int),
            'Total_PNBP': total_pnbp.values
        }).sort_values('Tahun').reset_index(drop=True)
        st.dataframe(df_total)
        st.session_state['df_total'] = df_total
        st.success("✅ Data agregasi berhasil. Lanjut ke step berikutnya.")
    else:
        st.info("Selesaikan input data dulu di step sebelumnya.")

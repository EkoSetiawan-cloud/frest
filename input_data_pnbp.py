import streamlit as st
import pandas as pd

def input_data_page():
    st.header("📥 1️⃣ Input Data PNBP (Excel)")

    st.markdown("""
    <div style="background-color:#E3F2FD; padding:14px; border-radius:10px; margin-bottom:10px;">
    <b>Petunjuk Upload:</b><br>
    - Pastikan file <b>Excel</b> (.xlsx) berisi <b>sheet 'Tabel_Target_dan_Realisasi_PNBP'</b><br>
    - Kolom utama: <i>Jenis PNBP, 2014, 2015, ... , 2024</i><br>
    - Jika gagal upload, cek format file dan nama sheet!
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("🔼 Upload file Dataset-PNBP.xlsx", type="xlsx", key="file1")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, sheet_name='Tabel_Target_dan_Realisasi_PNBP')
            df = df.astype(str)  # Preview awal as string, untuk menghindari error
            st.success("✅ Data berhasil di-load! Silakan cek detail di bawah.")

            # Preview data
            st.subheader("👁️ Preview 5 Baris Pertama")
            st.dataframe(df.head())
            
            # Info struktur data
            st.markdown("#### 📋 Info Kolom & Tipe Data")
            info_df = pd.DataFrame({
                "Kolom": df.columns,
                "Tipe Data": [df[col].dtype for col in df.columns],
                "Missing Value": df.isnull().sum().values
            })
            st.dataframe(info_df)

            # Summary
            st.markdown(f"""
                <b>Jumlah Baris:</b> {df.shape[0]} &nbsp; | &nbsp;
                <b>Jumlah Kolom:</b> {df.shape[1]}
            """, unsafe_allow_html=True)
            
            # Tabel missing value (per baris, jika ada)
            if df.isnull().sum().sum() > 0:
                st.warning("⚠️ Ada data kosong (missing value) pada beberapa kolom!")
                st.dataframe(df[df.isnull().any(axis=1)].head(10))
            
            # Simpan ke session
            st.session_state['df_mentah'] = df
            
        except Exception as e:
            st.error(f"❌ Gagal membaca file! Pesan error: {e}")
            st.markdown("""
            <div style="color:gray; font-size:14px;">
            <b>Troubleshooting:</b> 
            - Pastikan file bukan hasil 'Save As PDF'.
            - Cek penamaan sheet & header kolom.
            - Coba buka dan simpan ulang file dengan Excel versi terbaru.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Silakan upload file <b>Dataset-PNBP.xlsx</b> pada kotak di atas.", icon="ℹ️")
        # Dummy preview
        st.markdown("##### Contoh Tampilan Dataset yang Diterima:")
        contoh = pd.DataFrame({
            "Jenis PNBP": ["BHP Frekuensi", "Izin Spektrum", "Kompensasi"],
            "2014": [1000000000, 500000000, 200000000],
            "2015": [1100000000, 600000000, 250000000],
            "2016": [1200000000, 650000000, 275000000]
        })
        st.dataframe(contoh)

    st.markdown("""
    ---
    <span style="color:gray; font-size:14px;">
    <b>Tips:</b> Untuk hasil terbaik, pastikan format data sudah rapi dan bebas dari cell merge/hidden.
    </span>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd

def clean_data_page():
    st.header("🧹 3️⃣ Bersihkan Data Total PNBP Tahunan")
    st.markdown("""
    <div style="background-color:#FFF3E0; padding:14px; border-radius:10px; margin-bottom:10px;">
    <b>Deskripsi:</b><br>
    Pada tahap ini, data hasil agregasi akan dibersihkan dari duplikat dan nilai kosong (NaN) agar kualitas analisis dan prediksi terjaga.
    </div>
    """, unsafe_allow_html=True)

    df_total = st.session_state.get('df_total')
    if df_total is not None:
        # Sebelum bersih
        st.markdown("#### 🔍 Data Sebelum Dibersihkan")
        st.dataframe(df_total)
        baris_awal = df_total.shape[0]
        nan_awal = df_total.isnull().sum().sum()
        duplikat_awal = df_total.duplicated(subset='Tahun').sum()

        # Bersihkan data
        df_clean = df_total.drop_duplicates(subset='Tahun').dropna()
        baris_akhir = df_clean.shape[0]
        nan_akhir = df_clean.isnull().sum().sum()

        # Statistik perubahan
        st.markdown(f"""
            <div style="background-color:#E1F5FE; padding:10px; border-radius:8px; margin-bottom:8px;">
            <b>Jumlah Baris Sebelum:</b> {baris_awal} <br>
            <b>Jumlah Baris Setelah Bersih:</b> {baris_akhir} <br>
            <b>Duplikat Dibuang:</b> {duplikat_awal} <br>
            <b>Baris Mengandung NaN:</b> {nan_awal}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ✅ Data Setelah Dibersihkan")
        st.dataframe(df_clean)

        # Info jika masih ada masalah
        if nan_akhir > 0:
            st.warning(f"⚠️ Masih terdapat {nan_akhir} nilai kosong pada data. Cek kembali sumber data!")
        if df_clean.empty:
            st.error("❌ Semua data terbuang! Periksa proses agregasi/data sumber.")
        elif baris_akhir < baris_awal:
            st.info("ℹ️ Data sudah lebih bersih, siap digunakan ke tahap berikutnya.")

        # Simpan ke session
        st.session_state['df_clean'] = df_clean
        if not df_clean.empty:
            st.success("✅ Data sudah bersih. Lanjut ke *Split Data Train-Test*.")
    else:
        st.info("Belum ada data agregasi. Selesaikan langkah sebelumnya terlebih dahulu.")

    st.markdown("""
    ---
    <span style="color:gray; font-size:14px;">
    <b>Tips:</b> Jika seluruh data terbuang, cek lagi apakah kolom tahun dan nilainya sudah benar, serta tidak ada baris kosong di data asli.
    </span>
    """, unsafe_allow_html=True)

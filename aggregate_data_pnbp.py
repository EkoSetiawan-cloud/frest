import streamlit as st
import pandas as pd

def aggregate_data_page():
    st.header("📊 2️⃣ Agregasi Total PNBP per Tahun")
    st.markdown("""
    <div style="background-color:#FFFDE7; padding:14px; border-radius:10px; margin-bottom:10px;">
    <b>Deskripsi:</b><br>
    Tahap ini menjumlahkan seluruh PNBP per tahun (berbasis kolom tahun pada dataset). Hasil agregasi akan dipakai untuk prediksi tren dan evaluasi performa model.
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get('df_mentah')
    if df is not None:
        try:
            # Pastikan kolom tahun numerik, kecuali 'Jenis PNBP'
            year_columns = [col for col in df.columns if col != 'Jenis PNBP']
            for col in year_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            total_pnbp = df[year_columns].sum(axis=0)
            df_total = pd.DataFrame({
                'Tahun': total_pnbp.index.astype(int),
                'Total_PNBP': total_pnbp.values
            }).sort_values('Tahun').reset_index(drop=True)

            # Ringkasan hasil
            st.markdown(f"""
                <div style="background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:10px;">
                <b>Jumlah Tahun:</b> {df_total.shape[0]} <br>
                <b>Total PNBP Keseluruhan:</b> <span style="color:#1565C0"><b>Rp {df_total['Total_PNBP'].sum():,.0f}</b></span>
                </div>
            """, unsafe_allow_html=True)
            
            # Tabel agregasi (sortable, filterable)
            st.markdown("#### 🗃️ Tabel Agregasi Total PNBP per Tahun")
            st.dataframe(df_total, use_container_width=True)

            # Chart total PNBP per tahun
            st.markdown("#### 📈 Visualisasi Total PNBP per Tahun")
            st.line_chart(df_total.set_index("Tahun")["Total_PNBP"], use_container_width=True)

            # Cek duplikat/anomali
            if df_total.duplicated(subset='Tahun').sum() > 0:
                st.warning("⚠️ Ada tahun duplikat pada hasil agregasi!")
            if df_total['Tahun'].min() < 2000 or df_total['Tahun'].max() > 2050:
                st.warning("⚠️ Ditemukan tahun di luar rentang normal (2000-2050)!")

            # Simpan ke session state
            st.session_state['df_total'] = df_total
            st.success("✅ Data agregasi berhasil. Lanjut ke step berikutnya: *Bersihkan Data*.")
        except Exception as e:
            st.error(f"❌ Terjadi error saat agregasi data: {e}")

    else:
        st.info("Belum ada data! Selesaikan dulu proses *Input Data* di langkah sebelumnya.")

    st.markdown("""
    ---
    <span style="color:gray; font-size:14px;">
    <b>Tips:</b> Jika data tidak muncul dengan benar, cek urutan langkah dan pastikan format kolom tahun pada dataset sudah benar.
    </span>
    """, unsafe_allow_html=True)

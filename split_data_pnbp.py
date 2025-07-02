import streamlit as st
import pandas as pd

def split_data_page():
    st.header("🪓 4️⃣ Split Data Train-Test")
    st.markdown("""
    <div style="background-color:#F1F8E9; padding:14px; border-radius:10px; margin-bottom:10px;">
    <b>Deskripsi:</b><br>
    Data akan dibagi menjadi <b>Train</b> (latih) dan <b>Test</b> (uji) berdasarkan tahun, agar prediksi bisa dievaluasi secara objektif. Umumnya data test dipilih 3–5 tahun terakhir.
    </div>
    """, unsafe_allow_html=True)

    df_clean = st.session_state.get('df_clean')
    if df_clean is not None and not df_clean.empty:
        tahun_min = int(df_clean['Tahun'].min()) + 1
        tahun_max = int(df_clean['Tahun'].max())
        default_test_start = tahun_max-3 if tahun_max-3 >= tahun_min else tahun_min
        test_start_year = st.number_input(
            "Tahun awal data test (rekomendasi: 3–5 tahun terakhir):",
            min_value=tahun_min,
            max_value=tahun_max,
            value=default_test_start,
            step=1
        )

        train_df = df_clean[df_clean['Tahun'] < test_start_year].reset_index(drop=True)
        test_df = df_clean[df_clean['Tahun'] >= test_start_year].reset_index(drop=True)

        # Ringkasan
        st.markdown(f"""
            <div style="background-color:#E3F2FD; padding:10px; border-radius:8px; margin-bottom:8px;">
            <b>Train set:</b> {train_df['Tahun'].min()} – {train_df['Tahun'].max()} <br>
            <b>Test set:</b> {test_df['Tahun'].min()} – {test_df['Tahun'].max()} <br>
            <b>Jumlah Tahun Train:</b> {train_df.shape[0]} &nbsp; | &nbsp;
            <b>Jumlah Tahun Test:</b> {test_df.shape[0]}
            </div>
        """, unsafe_allow_html=True)

        # Warning proporsi timpang
        total = train_df.shape[0] + test_df.shape[0]
        prop_train = train_df.shape[0]/total*100 if total else 0
        prop_test = test_df.shape[0]/total*100 if total else 0
        if prop_train < 60 or prop_test < 10:
            st.warning("⚠️ Proporsi data train dan test kurang ideal. Cek kembali pemilihan tahun split.")

        # Visualisasi split (bar chart)
        st.markdown("#### 📊 Visualisasi Split Tahun")
        split_viz = pd.DataFrame({
            "Set": ["Train"]*train_df.shape[0] + ["Test"]*test_df.shape[0],
            "Tahun": list(train_df['Tahun']) + list(test_df['Tahun']),
            "Total_PNBP": list(train_df['Total_PNBP']) + list(test_df['Total_PNBP'])
        })
        chart_data = split_viz.pivot(index="Tahun", columns="Set", values="Total_PNBP").fillna(0)
        st.bar_chart(chart_data, use_container_width=True)

        # Tabel detail
        st.subheader("Tabel Data Train")
        st.dataframe(train_df)
        st.subheader("Tabel Data Test")
        st.dataframe(test_df)

        # Simpan ke session state
        st.session_state['train_df'] = train_df
        st.session_state['test_df'] = test_df

        st.success(f"✅ Split berhasil! Data siap untuk proses modelling dan evaluasi.")
    else:
        st.info("Belum ada data bersih. Selesaikan langkah sebelumnya terlebih dahulu.")

    st.markdown("""
    ---
    <span style="color:gray; font-size:14px;">
    <b>Tips:</b> Proporsi umum data train : test adalah 70:30. Jangan lupa lakukan split secara konsisten sebelum modelling.
    </span>
    """, unsafe_allow_html=True)

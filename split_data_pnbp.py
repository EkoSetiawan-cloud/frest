import streamlit as st
import pandas as pd

def split_data_page():
    st.header("4️⃣ Split Data Train-Test")
    df_clean = st.session_state.get('df_clean')
    if df_clean is not None:
        tahun_min = int(df_clean['Tahun'].min()) + 1
        tahun_max = int(df_clean['Tahun'].max())
        test_start_year = st.number_input("Tahun awal data test:", min_value=tahun_min, max_value=tahun_max, value=tahun_max-3, step=1)
        train_df = df_clean[df_clean['Tahun'] < test_start_year].reset_index(drop=True)
        test_df = df_clean[df_clean['Tahun'] >= test_start_year].reset_index(drop=True)
        st.subheader("Data Train")
        st.dataframe(train_df)
        st.subheader("Data Test")
        st.dataframe(test_df)
        st.session_state['train_df'] = train_df
        st.session_state['test_df'] = test_df
        st.success(f"✅ Train: {train_df['Tahun'].min()}–{train_df['Tahun'].max()}, Test: {test_df['Tahun'].min()}–{test_df['Tahun'].max()}")
    else:
        st.info("Selesaikan step bersih data dulu.")

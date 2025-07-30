import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="Upload Option Chain to DB", layout="centered")
st.title("📤 Upload Option Chain to PostgreSQL")

# --- DB Connection ---
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Option Chain Excel File", type=["xlsx"])

trade_date = st.date_input("Trade Date", value=datetime.today())
expiry_date = st.date_input("Expiry Date")

if uploaded_file:
    try:
        with engine.connect() as conn:
            # Check for duplicates
            check_query = text("""
                SELECT COUNT(*) FROM fact_option_chain
                WHERE trade_date = :trade_date AND expiry_date = :expiry_date
            """)
            result = conn.execute(check_query, {
                "trade_date": trade_date,
                "expiry_date": expiry_date
            }).scalar()

        if result > 0:
            st.warning(f"⚠️ Records for trade_date {trade_date} and expiry_date {expiry_date} already exist.")
            st.stop()

        df_raw = pd.read_excel(uploaded_file, skiprows=3)

        ce_cols = df_raw.iloc[:, [32, 28, 10, 9, 4]].copy()
        pe_cols = df_raw.iloc[:, [32, 40, 58, 59, 60]].copy()

        ce_cols.columns = ['strike', 'premium', 'delta', 'gamma', 'vega']
        pe_cols.columns = ['strike', 'premium', 'delta', 'gamma', 'vega']
        ce_cols['option_type'] = 'CE'
        pe_cols['option_type'] = 'PE'

        df = pd.concat([ce_cols, pe_cols], ignore_index=True)

        for col in ['strike', 'premium', 'delta', 'gamma', 'vega']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        df['trade_date'] = pd.to_datetime(trade_date)
        df['expiry_date'] = pd.to_datetime(expiry_date)

        # Reorder columns
        df = df[['trade_date', 'expiry_date', 'strike', 'option_type', 'premium', 'delta', 'gamma', 'vega']]

        with engine.connect() as conn:
            df.to_sql("fact_option_chain", con=conn, if_exists='append', index=False, method='multi')

        st.success(f"✅ Uploaded {len(df)} records to 'fact_option_chain' table.")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Error: {e}")

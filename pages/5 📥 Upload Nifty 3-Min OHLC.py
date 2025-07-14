import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote
from datetime import datetime

# ---------------------
# 🎯 Page Title
# ---------------------
st.title("📥 Upload Nifty 3-Min OHLC Data")
st.markdown("Upload your TradingView-exported CSV file containing Nifty Futures OHLC with 3-min interval.")

# ---------------------
# 🗄️ DB Connection
# ---------------------
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ---------------------
# 📁 File Upload
# ---------------------
uploaded_file = st.file_uploader("Choose CSV file with 3-min interval data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # 1️⃣ Parse & clean
        df['timestamp'] = pd.to_datetime(df['time'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df['trade_date'] = df['timestamp'].dt.date

        # 2️⃣ Rename columns to lowercase
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'Volume': 'volume',
            'VWAP': 'vwap'
        }, inplace=True)

        # 3️⃣ Final column selection & order
        df = df[['trade_date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

        # 4️⃣ Create table (if not exists)
        with engine.connect() as conn:

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fact_nifty_3min_ohlc (
                    trade_date DATE,
                    timestamp TIMESTAMP PRIMARY KEY,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    vwap NUMERIC
                );
            """))

        # 5️⃣ Upload to DB
        df.to_sql("nifty_3min_ohlc", con=engine, if_exists='append', index=False, method='multi')

        st.success(f"✅ Successfully inserted {len(df)} records into `nifty_3min_ohlc` table.")

        with st.expander("🔍 Preview Data"):
            st.dataframe(df.head(10))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# -------------------------
# 🔐 DB Connection
# -------------------------
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# -------------------------
# 🎯 Page Title
# -------------------------
st.title("📆 Multi-Day Strike Analysis with VWAP")
st.markdown("Analyze buildup patterns, VWAP, and Price/OI movement for a selected option strike.")

# -------------------------
# 🎛️ User Input
# -------------------------
with st.sidebar:
    st.header("Strike Filters")
    symbol = st.text_input("Symbol", value="NIFTY")
    strike_price = st.number_input("Strike Price", value=25100)
    option_type = st.selectbox("Option Type", ["CE", "PE"])
    expiry_date = st.date_input("Expiry Date")
    start_date = st.date_input("From Date")
    end_date = st.date_input("To Date")

# -------------------------
# 🔍 Data Load and Analysis
# -------------------------
if st.button("Analyze"):
    query = text("""
        SELECT * FROM option_3min_ohlc
        WHERE symbol = :symbol
          AND strike_price = :strike
          AND option_type = :otype
          AND expiry_date = :expiry
          AND trade_date BETWEEN :start AND :end
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "symbol": symbol,
            "strike": strike_price,
            "otype": option_type,
            "expiry": expiry_date,
            "start": start_date,
            "end": end_date
        })

    if df.empty:
        st.warning("No data found.")
    else:
        # 🧮 Buildup tagging
        df["oi_change"] = df["open_interest"].diff()
        df["price_change"] = df["close"].diff()

        def tag_buildup(row):
            if pd.isna(row["oi_change"]) or pd.isna(row["price_change"]):
                return None
            if row["price_change"] > 0 and row["oi_change"] > 0:
                return "Long Buildup"
            elif row["price_change"] < 0 and row["oi_change"] > 0:
                return "Short Buildup"
            elif row["price_change"] > 0 and row["oi_change"] < 0:
                return "Short Covering"
            elif row["price_change"] < 0 and row["oi_change"] < 0:
                return "Longs Unwinding"
            else:
                return "Neutral"

        df["buildup"] = df.apply(tag_buildup, axis=1)

        # 📊 VWAP Calculation
        df["vwap_numerator"] = df["close"] * df["volume"]
        df_vwap = df.groupby("trade_date").agg(
            vwap_numerator=("vwap_numerator", "sum"),
            total_volume=("volume", "sum")
        )
        df_vwap["VWAP"] = df_vwap["vwap_numerator"] / df_vwap["total_volume"]

        # 📋 Daily Buildup Summary
        buildup_summary = df.groupby(["trade_date", "buildup"]).size().unstack(fill_value=0)

        # 📈 Display
        st.subheader("🔎 Daily Buildup Summary")
        st.dataframe(buildup_summary)

        st.subheader("📊 Daily VWAP Levels")
        st.dataframe(df_vwap[["VWAP"]].round(2))

        st.subheader("📈 Intraday Price & OI Trend")
        st.line_chart(df.set_index("timestamp")[["close", "open_interest"]])

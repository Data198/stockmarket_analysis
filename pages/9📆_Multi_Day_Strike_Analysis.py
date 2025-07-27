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
st.title("📈 Multi-Day Strike Analysis with Entry Zones")

with st.sidebar:
    st.header("🔎 Strike Filters")
    symbol = st.text_input("Symbol", value="NIFTY")
    strike_price = st.number_input("Strike Price", value=25100)
    option_type = st.selectbox("Option Type", ["CE", "PE"])
    expiry_date = st.date_input("Expiry Date")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

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
            return "Neutral"

        df["buildup"] = df.apply(tag_buildup, axis=1)
        df["vwap_numerator"] = df["close"] * df["volume"]

        df_vwap = df.groupby("trade_date").agg(
            total_volume=("volume", "sum"),
            total_vwap_value=("vwap_numerator", "sum")
        )
        df_vwap["VWAP"] = df_vwap["total_vwap_value"] / df_vwap["total_volume"]

        st.subheader("📊 Daily VWAP Levels")
        st.dataframe(df_vwap[["VWAP"]].round(2))

        st.subheader("📋 Buildup Summary by Day")
        buildup_summary = df.groupby(["trade_date", "buildup"]).size().unstack(fill_value=0)
        st.dataframe(buildup_summary)

        st.subheader("📌 Entry Zones for Option Buyers")
        sc_df = df[df["buildup"] == "Short Covering"]
        lu_df = df[df["buildup"] == "Longs Unwinding"]

        if not sc_df.empty:
            st.markdown("### ✅ Potential Breakout Zones (SC)")
            for i, row in sc_df.iterrows():
                st.markdown(f"- {row['trade_date']} {row['timestamp']} → ₹{row['close']:.2f}")
        else:
            st.info("No Short Covering zones found.")

        if not lu_df.empty:
            st.markdown("### ⚠️ Avoidance / Breakdown Zones (LU)")
            for i, row in lu_df.iterrows():
                st.markdown(f"- {row['trade_date']} {row['timestamp']} → ₹{row['close']:.2f}")
        else:
            st.info("No Longs Unwinding zones found.")

        st.subheader("📈 Price vs Open Interest")
        st.line_chart(df.set_index("timestamp")[["close", "open_interest"]])

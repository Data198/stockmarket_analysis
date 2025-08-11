import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote
from datetime import datetime
import re
import numpy as np

# ------------------------------
# DB Connection
# ------------------------------
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

st.title("📂 Upload Option OHLC (3-min) Data")

def extract_metadata_from_filename(filename: str):
    name = filename.replace(".xlsx", "")
    parts = name.split("_")

    symbol = parts[0].upper()

    # Find CE or PE strike
    ce_part = next((p for p in parts if "CE" in p and not p.startswith("0")), None)
    pe_part = next((p for p in parts if "PE" in p and not p.startswith("0")), None)
    option_type = "CE" if ce_part else "PE"
    strike_part = ce_part if ce_part else pe_part

    if not strike_part:
        raise ValueError("❌ Could not identify strike price.")

    strike_price = int(re.sub(r"[^\d]", "", strike_part))

    # Find expiry date (e.g., 17JUL25)
    expiry_str = next((p for p in parts if re.match(r"\d{2}[A-Z]{3}\d{2}", p)), None)
    if not expiry_str:
        raise ValueError("❌ Expiry date not found in filename.")
    expiry = datetime.strptime(expiry_str, "%d%b%y").date()

    # Find trade date (e.g., 14_7_2025 or 14_07_2025)
    trade_match = re.search(r"(\d{1,2})_(\d{1,2})_(\d{4})", filename)
    if not trade_match:
        raise ValueError("❌ Trade date (e.g., 14_7_2025 or 14_07_2025) not found in filename.")
    day, month, year = trade_match.groups()
    trade_date = datetime.strptime(f"{day}_{month}_{year}", "%d_%m_%Y").date()

    return symbol, strike_price, option_type, expiry, trade_date

uploaded_files = st.file_uploader("Upload CE/PE OHLC Excel Files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            st.markdown(f"### Processing: `{uploaded_file.name}`")
            symbol, strike_price, option_type, expiry, trade_date = extract_metadata_from_filename(uploaded_file.name)
            st.markdown(
                f"**Symbol:** `{symbol}` | **Trade Date:** `{trade_date}` | **Expiry Date:** `{expiry}` | "
                f"**Strike:** `{strike_price}` | **Option Type:** `{option_type}`"
            )

            if st.button(f"Process and Upload: {uploaded_file.name}"):
                raw_preview = pd.read_excel(uploaded_file, header=None)

                header_row = None
                for i, row in raw_preview.iterrows():
                    if "Time" in row.values:
                        header_row = i
                        break

                if header_row is None:
                    raise ValueError("❌ Couldn't detect header row.")

                df_raw = pd.read_excel(uploaded_file, header=header_row)

                # Dynamic PE/CE mapping
                suffix = "" if option_type == "CE" else ".1"
                mapping = {
                    "open_interest": f"OI{suffix}",
                    "oi_change_pct": f"OI Chg %{suffix}",
                    "vwap": f"VWAP{suffix}",
                    "close": f"Price{suffix}",
                    "volume": f"Volume{suffix}",
                    "iv": f"IV{suffix}",
                    "price_change": f"Price Chg{suffix}"
                }

                def get_col(col_name):
                    return next((c for c in df_raw.columns if c.strip() == col_name.strip()), None)

                df_clean = pd.DataFrame()
                df_clean["timestamp"] = df_raw[get_col("Time")]
                for new_col, old_col in mapping.items():
                    col = get_col(old_col)
                    df_clean[new_col] = df_raw[col] if col else np.nan

                df_clean = df_clean.dropna(subset=["timestamp"])
                df_clean = df_clean[~df_clean["timestamp"].astype(str).str.lower().str.contains("time")]
                df_clean["timestamp"] = pd.to_datetime(trade_date.strftime("%Y-%m-%d") + " " + df_clean["timestamp"].astype(str))

                numeric_cols = ["open_interest", "oi_change_pct", "vwap", "close", "volume", "iv", "price_change"]
                for col in numeric_cols:
                    df_clean[col] = df_clean[col].astype(str).str.replace(",", "").str.strip()
                    df_clean[col] = df_clean[col].replace(["", "-", "None", "."], np.nan)
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                df_clean["symbol"] = symbol
                df_clean["strike_price"] = strike_price
                df_clean["option_type"] = option_type
                df_clean["expiry_date"] = expiry
                df_clean["trade_date"] = trade_date

                df_final = df_clean[[
                    "trade_date", "timestamp", "symbol", "strike_price", "option_type", "expiry_date",
                    "open_interest", "oi_change_pct", "vwap", "close", "volume", "iv", "price_change"
                ]]

                with engine.connect() as conn:
                    query = text("""
                        SELECT timestamp FROM option_3min_ohlc
                        WHERE symbol = :symbol AND trade_date = :trade_date AND strike_price = :strike_price
                        AND option_type = :option_type AND timestamp = ANY(:timestamps)
                    """)
                    existing_timestamps = conn.execute(query, {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "strike_price": int(strike_price),
                        "option_type": option_type,
                        "timestamps": [ts.to_pydatetime() for ts in df_final["timestamp"]]
                    }).scalars().all()

                df_final = df_final[~df_final["timestamp"].isin(existing_timestamps)]

                if df_final.empty:
                    st.warning(f"⚠️ All rows already exist in DB for {uploaded_file.name}.")
                else:
                    df_final.to_sql("option_3min_ohlc", engine, if_exists="append", index=False, method="multi")
                    st.success(f"✅ Inserted {len(df_final)} new rows for {uploaded_file.name}")
                    with st.expander(f"Preview Inserted Data: {uploaded_file.name}"):
                        st.dataframe(df_final)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine, text
from urllib.parse import quote
from fyers_apiv3 import fyersModel
import pytz

# --- Load DB secrets from .streamlit/secrets.toml ---
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
database = st.secrets["postgres"]["database"]

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# --- Fyers Auth ---
client_id = "NAKKSHA7SX-100"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb2VlbHlVT2UtbktrTm1SYWQyUUxjTnhyZGNsaXNoREh1RGdaRVg5dDk1MDBKREQ1NW15MVpqQ2RIZlROU0tBOVIwNmJjY0lRWDNqTTFwNlFYcGZnYnYyNE05cHZvY05hTHNROU0xelI1MWxWMmREST0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIyMGIxODc2MmM1MmVhYzNkYjE2OTEyMGYyNDliMzUwNzc1MGY4ZjNmNmFiMWU5ZDZjMmZiZTRkMyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiREQwMTA3NiIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzUyODg1MDAwLCJpYXQiOjE3NTI4MjAwODIsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1MjgyMDA4Miwic3ViIjoiYWNjZXNzX3Rva2VuIn0.LwIrxbR-5WsWgVlUgBLTvmtIbgxVYVlWurpwGI2oVWo"  # 🔐 Replace with your token

fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path=None)

# --- Utility: Construct Option Symbol ---
def construct_symbol(index: str, expiry: date, strike: int, option_type: str) -> str:
    if expiry.weekday() != 3:  # Ensure expiry is Thursday (weekly options)
        raise ValueError("Expiry must be on a Thursday for NIFTY/BANKNIFTY options.")
    
    month_map = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'O', 11: 'N', 12: 'D'}
    m = month_map[expiry.month]
    dd = expiry.strftime("%d").lstrip('0')  # Remove leading zero from day
    yy = expiry.strftime("%y")
    return f"NSE:{index.upper()}{yy}{m}{dd}{int(strike)}{option_type.upper()}"

# --- Fetch & Store OHLC for Given Day ---
def fetch_and_store_ohlc(symbol: str, for_date: date) -> str:
    data = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": for_date.strftime("%Y-%m-%d"),  # Ensure proper date format
        "range_to": for_date.strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }

    response = fyers.history(data)

    if "candles" in response:
        df = pd.DataFrame(response["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        df["trade_date"] = for_date
        df["symbol"] = symbol
        df = df[["trade_date", "symbol", "timestamp", "open", "high", "low", "close", "volume"]]

        with engine.connect() as conn:
            existing = pd.read_sql(
                text("SELECT timestamp FROM ohlc_options_live WHERE symbol = :sym AND trade_date = :dt"),
                conn, params={"sym": symbol, "dt": for_date}
            )

        new_df = df[~df["timestamp"].isin(existing["timestamp"])]
        if not new_df.empty:
            new_df.to_sql("ohlc_options_live", con=engine, index=False, if_exists="append", method="multi")
            return f"✅ {len(new_df)} inserted: {symbol} ({for_date})"
        else:
            return f"⚠️ No new data: {symbol} ({for_date})"
    else:
        return f"❌ Error for {symbol} ({for_date}): {response.get('message', 'Unknown')}"

# --- Streamlit UI ---
st.set_page_config(page_title="Option OHLC Fetcher", layout="wide")
st.title("📈 Option OHLC Fetcher (Strike Range + Date Range)")

col1, col2, col3 = st.columns(3)
with col1:
    index = st.selectbox("📌 Index", ["NIFTY", "BANKNIFTY"])
with col2:
    option_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)
with col3:
    expiry_date = st.date_input("🗓️ Expiry Date", value=date(2025, 7, 17))  # Set to a Thursday

start_strike = st.number_input("🔢 Strike Start", step=50, value=25000)
end_strike = st.number_input("🔢 Strike End", step=50, value=25200)
strike_step = st.selectbox("📐 Strike Step", [50, 100], index=0)

start_date = st.date_input("📆 Start Date", value=date.today() - timedelta(days=2))
end_date = st.date_input("📆 End Date", value=date.today())

if st.button("🚀 Fetch OHLC for All Symbols"):
    if start_strike > end_strike or start_date > end_date:
        st.error("Check your strike range or date range.")
    else:
        strike_range = range(start_strike, end_strike + 1, strike_step)
        date_range = pd.date_range(start_date, end_date)
        total_tasks = len(strike_range) * len(date_range)
        progress = st.progress(0)
        status_log = []

        task = 0
        for strike in strike_range:
            symbol = construct_symbol(index, expiry_date, strike, option_type)
            for d in date_range:
                msg = fetch_and_store_ohlc(symbol, d.date())
                status_log.append(msg)
                task += 1
                progress.progress(task / total_tasks)

        st.success("🎯 Completed fetching for all symbols!")
        for msg in status_log:
            st.write(msg)

# --- Show Results for Last Symbol on End Date ---
if st.checkbox("📊 Show Last Symbol Candles"):
    last_symbol = construct_symbol(index, expiry_date, end_strike, option_type)
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT * FROM ohlc_options_live
                WHERE symbol = :sym AND trade_date = :dt
                ORDER BY timestamp DESC
                LIMIT 10
            """),
            conn,
            params={"sym": last_symbol, "dt": end_date}
        )
    st.dataframe(df)
    if not df.empty:
        st.line_chart(df.set_index("timestamp")[["close"]])
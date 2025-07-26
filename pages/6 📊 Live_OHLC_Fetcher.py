import streamlit as st
import pandas as pd
from datetime import date
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.dialects.postgresql import insert
from fyers_apiv3 import fyersModel
from urllib.parse import quote
import pytz

# --- Load DB credentials from Streamlit secrets ---
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
database = st.secrets["postgres"]["database"]

# --- Initialize SQLAlchemy engine ---
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# --- Fyers credentials ---
client_id = "NAKKSHA7SX-100"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb2VlbHlVT2UtbktrTm1SYWQyUUxjTnhyZGNsaXNoREh1RGdaRVg5dDk1MDBKREQ1NW15MVpqQ2RIZlROU0tBOVIwNmJjY0lRWDNqTTFwNlFYcGZnYnYyNE05cHZvY05hTHNROU0xelI1MWxWMmREST0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIyMGIxODc2MmM1MmVhYzNkYjE2OTEyMGYyNDliMzUwNzc1MGY4ZjNmNmFiMWU5ZDZjMmZiZTRkMyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiREQwMTA3NiIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzUyODg1MDAwLCJpYXQiOjE3NTI4MjAwODIsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1MjgyMDA4Miwic3ViIjoiYWNjZXNzX3Rva2VuIn0.LwIrxbR-5WsWgVlUgBLTvmtIbgxVYVlWurpwGI2oVWo"  # Replace with full valid token

fyers = fyersModel.FyersModel(
    client_id=client_id,
    token=access_token,
    log_path=None
)

# --- Function to fetch and insert OHLC data ---
def fetch_and_store_ohlc(symbol: str, start_date: date, end_date: date) -> str:
    data = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": str(start_date),
        "range_to": str(end_date),
        "cont_flag": "1"
    }

    response = fyers.history(data)

    if "candles" not in response:
        return f"❌ Failed to fetch data: {response}"

    df = pd.DataFrame(response["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convert UTC timestamp to IST
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    
    # ✅ Derive trade_date from timestamp (not from input)
    df["trade_date"] = df["timestamp"].dt.date
    df["symbol"] = symbol
    df = df[["trade_date", "symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    # Insert into DB with deduplication
    meta = MetaData()
    table = Table("ohlc_equity_live", meta, autoload_with=engine)

    inserted = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            stmt = insert(table).values(**row.to_dict()).on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
            result = conn.execute(stmt)
            inserted += result.rowcount

    return f"✅ {inserted} new candles inserted for {symbol} ({start_date} to {end_date})"


# ----------------------------
# 📊 Streamlit UI Starts Here
# ----------------------------
st.title("📈 Fetch Historical 1-min OHLC (Fyers API)")

symbol = st.text_input("🔎 Enter symbol (e.g., NSE:HAVELLS-EQ)", value="NSE:HAVELLS-EQ")
start_date = st.date_input("📅 From Date", value=date.today())
end_date = st.date_input("📅 To Date", value=date.today())

if st.button("🚀 Fetch & Save to PostgreSQL"):
    if start_date > end_date:
        st.error("⚠️ Start date must be before end date.")
    else:
        status = fetch_and_store_ohlc(symbol, start_date, end_date)
        st.success(status)

# 📉 Optional: Show Last 10 Candles
if st.checkbox("📊 Show last 10 candles from DB"):
    query = f"""
    SELECT * FROM ohlc_equity_live
    WHERE symbol = '{symbol}' AND trade_date = '{end_date}'
    ORDER BY timestamp DESC
    LIMIT 10
    """
    df_view = pd.read_sql(query, engine)
    st.dataframe(df_view)
    st.line_chart(df_view.set_index("timestamp")[["close"]])

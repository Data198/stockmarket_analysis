import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# ------------------------------
# DB Connection
# ------------------------------
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# ------------------------------
# Function: Generate 40% Threshold Trade Signals (One Long & One Short per Strike)
# ------------------------------
def generate_40pct_trade_signals(df, price_col='close', time_col='timestamp', strike_col='strike_price'):
    df = df.sort_values(by=[strike_col, time_col]).copy()

    market_open_time = pd.to_datetime(df[time_col].dt.date.min().strftime('%Y-%m-%d') + ' 09:15:00')
    first_candle_close_time = market_open_time + pd.Timedelta(minutes=3)  # 9:18 AM

    signals = []

    for strike, group in df.groupby(strike_col):
        group = group.reset_index(drop=True)

        # Find first candle close price after first candle close time
        first_candle = group[group[time_col] >= first_candle_close_time].head(1)
        if first_candle.empty:
            continue

        first_close = first_candle.iloc[0][price_col]

        upper_threshold = first_close + 0.40 * first_close
        lower_threshold = first_close - 0.40 * first_close

        short_taken = False
        long_taken = False

        for i, row in group.iterrows():
            current_time = row[time_col]
            if current_time < first_candle_close_time:
                continue

            price = row[price_col]

            # Check upper threshold hit for short trade
            if (not short_taken) and (price >= upper_threshold):
                entry_price = price
                stop_loss = entry_price + 0.20 * first_close
                signals.append({
                    'timestamp': current_time,
                    'strike_price': strike,
                    'entry_price': entry_price,
                    'direction': 'SHORT',
                    'stop_loss': stop_loss
                })
                short_taken = True

            # Check lower threshold hit for long trade
            if (not long_taken) and (price <= lower_threshold):
                entry_price = price
                stop_loss = entry_price - 0.20 * first_close
                signals.append({
                    'timestamp': current_time,
                    'strike_price': strike,
                    'entry_price': entry_price,
                    'direction': 'LONG',
                    'stop_loss': stop_loss
                })
                long_taken = True

            # If both trades taken, break early
            if short_taken and long_taken:
                break

    return pd.DataFrame(signals)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Intraday 40% Premium Trade Signal Generator")

trade_date = st.date_input("Select Trade Date", value=pd.to_datetime("2025-08-01"))
symbol = st.text_input("Enter Symbol", value="NIFTY")
expiry_date = st.date_input("Select Expiry Date", value=pd.to_datetime("2025-08-07"))

# Format dates for SQL parameters
trade_date_str = trade_date.strftime('%Y-%m-%d') if hasattr(trade_date, 'strftime') else str(trade_date)
expiry_date_str = expiry_date.strftime('%Y-%m-%d') if hasattr(expiry_date, 'strftime') else str(expiry_date)

params = {
    "trade_date": trade_date_str,
    "symbol": symbol,
    "expiry_date": expiry_date_str
}

st.write("Query Parameters:", params)

# ------------------------------
# Load data from DB
# ------------------------------
query = """
SELECT trade_date, timestamp, symbol, strike_price, option_type, expiry_date,
       open_interest, oi_change_pct, vwap, close, volume, iv, price_change
FROM option_3min_ohlc
WHERE trade_date = :trade_date
AND symbol = :symbol
AND expiry_date = :expiry_date
ORDER BY strike_price, timestamp
"""

try:
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
except Exception as e:
    st.error(f"Error loading data from database: {e}")
    st.stop()

if df.empty:
    st.warning("No data found for the selected parameters.")
    st.stop()

# Generate trade signals
df_signals = generate_40pct_trade_signals(df)

if df_signals.empty:
    st.warning("No 40% threshold trades generated for the selected parameters.")
else:
    st.markdown("### Generated Trade Signals")
    st.dataframe(df_signals)

    st.markdown(f"Total trades generated: {len(df_signals)}")

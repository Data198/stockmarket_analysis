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
# Function: Test 40% Intraday Reversal Using First Candle Close as Base (After First Candle Close)
# ------------------------------
def test_40pct_reversal_post_first_candle(df, price_col='close', time_col='timestamp', strike_col='strike_price'):
    df = df.sort_values(by=[strike_col, time_col]).copy()
    df['hit_upper_threshold'] = False
    df['hit_lower_threshold'] = False
    df['reversal_after_threshold'] = False

    reversal_window = 5  # 5 bars = 15 minutes for 3-min intervals

    # Define market open and first candle close time (assuming timestamps are datetime)
    market_open_time = pd.to_datetime(df[time_col].dt.date.min().strftime('%Y-%m-%d') + ' 09:15:00')
    first_candle_close_time = market_open_time + pd.Timedelta(minutes=3)  # 9:18 AM

    results = []

    for strike, group in df.groupby(strike_col):
        group = group.reset_index(drop=True)

        if group.empty or len(group) < 2:
            results.append(group)
            continue

        # Use first candle's close as base price for thresholds
        # But only consider the first candle that closes at or after first_candle_close_time
        first_candle = group[group[time_col] >= first_candle_close_time].head(1)
        if first_candle.empty:
            # No data after first candle close, skip
            results.append(group)
            continue

        first_close = first_candle.iloc[0][price_col]

        upper_threshold = first_close + 0.40 * first_close
        lower_threshold = first_close - 0.40 * first_close

        for i, row in group.iterrows():
            if i == 0:
                continue  # skip first candle

            current_time = row[time_col]

            # Only apply threshold logic after first candle close time
            if current_time < first_candle_close_time:
                continue

            price = row[price_col]

            # Check upward threshold hit
            if price >= upper_threshold:
                group.at[i, 'hit_upper_threshold'] = True
                # Check reversal: price falls back by 10% of first_close within next 5 intervals
                window = group.iloc[i+1:i+1+reversal_window]
                if (window[price_col] <= (price - 0.10 * first_close)).any():
                    group.at[i, 'reversal_after_threshold'] = True

            # Check downward threshold hit
            elif price <= lower_threshold:
                group.at[i, 'hit_lower_threshold'] = True
                # Check reversal: price rises by 10% of first_close within next 5 intervals
                window = group.iloc[i+1:i+1+reversal_window]
                if (window[price_col] >= (price + 0.10 * first_close)).any():
                    group.at[i, 'reversal_after_threshold'] = True

        results.append(group)

    return pd.concat(results, ignore_index=True)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Intraday 40% Premium Reversal Tester (First Candle Close Base)")

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

# Run the 40% reversal test
df_result = test_40pct_reversal_post_first_candle(df)

# Filter to show only rows where threshold was hit (upper or lower)
filtered_df = df_result[(df_result['hit_upper_threshold'] | df_result['hit_lower_threshold'])]

st.markdown("### Sample Test Results (Only Threshold Hits)")
st.dataframe(filtered_df[['timestamp', 'strike_price', 'close',
                          'hit_upper_threshold', 'hit_lower_threshold', 'reversal_after_threshold']].head(50))

# Optional: Uncomment below to show only reversals after threshold hits
# reversal_df = df_result[df_result['reversal_after_threshold']]
# st.markdown("### Sample Test Results (Only Reversals After Threshold)")
# st.dataframe(reversal_df[['timestamp', 'strike_price', 'close',
#                          'hit_upper_threshold', 'hit_lower_threshold', 'reversal_after_threshold']].head(50))

# Summary statistics
total_hits = filtered_df.shape[0]
total_reversals = filtered_df[filtered_df['reversal_after_threshold']].shape[0]

st.markdown(f"**Total 40% threshold hits:** {total_hits}")
st.markdown(f"**Reversals after threshold hits:** {total_reversals}")
st.markdown(f"**Reversal percentage:** {round(100 * total_reversals / total_hits, 2) if total_hits else 0} %")

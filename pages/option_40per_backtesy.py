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
# Function: Test 40% Intraday Reversal
# ------------------------------
def test_40pct_reversal(df, premium_col='premium', price_col='close', time_col='timestamp', strike_col='strike_price'):
    df = df.sort_values(by=[strike_col, time_col]).copy()
    df['threshold_40pct'] = df[premium_col] * 0.40

    results = []
    for strike, group in df.groupby(strike_col):
        group = group.reset_index(drop=True)
        entry_premium = group.loc[0, premium_col]
        threshold = entry_premium * 0.40
        group['price_change'] = group[price_col] - entry_premium
        group['hit_upper_threshold'] = group['price_change'] >= threshold
        group['hit_lower_threshold'] = group['price_change'] <= -threshold

        reversal_flags = []
        reversal_window = 5  # 5 bars = 15 mins for 3-min intervals

        for i, row in group.iterrows():
            if row['hit_upper_threshold']:
                window = group.iloc[i+1:i+1+reversal_window]
                reversal = (window[price_col] <= (row[price_col] - 0.10 * entry_premium)).any()
                reversal_flags.append(reversal)
            elif row['hit_lower_threshold']:
                window = group.iloc[i+1:i+1+reversal_window]
                reversal = (window[price_col] >= (row[price_col] + 0.10 * entry_premium)).any()
                reversal_flags.append(reversal)
            else:
                reversal_flags.append(False)

        group['reversal_after_threshold'] = reversal_flags
        results.append(group)

    return pd.concat(results, ignore_index=True)

# ------------------------------
# Streamlit UI: Input Selection
# ------------------------------
st.title("Intraday 40% Premium Reversal Tester")

trade_date = st.date_input("Select Trade Date", value=pd.to_datetime("2025-08-01"))
symbol = st.text_input("Enter Symbol", value="NIFTY")
expiry_date = st.date_input("Select Expiry Date", value=pd.to_datetime("2025-08-07"))

# ------------------------------
# Load Data from DB
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

params = {
    "trade_date": trade_date,
    "symbol": symbol,
    "expiry_date": expiry_date
}

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn, params=params)

if df.empty:
    st.warning("No data found for the selected parameters.")
    st.stop()

# Use 'close' as premium for this test
df['premium'] = df['close']

# ------------------------------
# Run Test & Show Results
# ------------------------------
df_result = test_40pct_reversal(df)

st.markdown("### Sample of Test Results")
st.dataframe(df_result[['timestamp', 'strike_price', 'premium', 'price_change', 
                        'hit_upper_threshold', 'hit_lower_threshold', 'reversal_after_threshold']].head(50))

# Summary statistics
total_hits = df_result[(df_result['hit_upper_threshold'] | df_result['hit_lower_threshold'])].shape[0]
total_reversals = df_result[df_result['reversal_after_threshold']].shape[0]

st.markdown(f"**Total 40% threshold hits:** {total_hits}")
st.markdown(f"**Reversals after threshold hits:** {total_reversals}")
st.markdown(f"**Reversal percentage:** {round(100 * total_reversals / total_hits, 2) if total_hits else 0} %")

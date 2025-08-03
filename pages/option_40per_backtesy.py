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

def backtest_40pct_reversal(df):
    df = df.copy()
    df['trade_date'] = df['timestamp'].dt.date
    df.sort_values(['tradingsymbol', 'trade_date', 'timestamp'], inplace=True)
    df['hit_upper_threshold'] = False
    df['hit_lower_threshold'] = False
    df['reversal_after_threshold'] = False

    grouped = df.groupby(['tradingsymbol', 'trade_date'])
    results = []

    for (symbol, day), group in grouped:
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        first_close = group.loc[0, 'close']
        upper_threshold = first_close * 1.40
        lower_threshold = first_close * 0.60

        for i, row in group.iterrows():
            if i == 0:
                continue

            price = row['close']

            if price >= upper_threshold:
                group.at[i, 'hit_upper_threshold'] = True
                window = group.iloc[i+1:i+6]
                if (window['close'] <= price * 0.90).any():
                    group.at[i, 'reversal_after_threshold'] = True

            elif price <= lower_threshold:
                group.at[i, 'hit_lower_threshold'] = True
                window = group.iloc[i+1:i+6]
                if (window['close'] >= price * 1.10).any():
                    group.at[i, 'reversal_after_threshold'] = True

        results.append(group)

    return pd.concat(results, ignore_index=True)

st.title("40% Premium Intraday Reversal Backtest")

with st.sidebar:
    symbol = st.text_input("Symbol", value="NIFTY")
    option_type = st.selectbox("Option Type", ["CE", "PE"])
    start_date = st.date_input("Start Date", value=pd.to_datetime("2025-07-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-07-31"))

    strikes_query = """
    SELECT DISTINCT strike_price
    FROM option_3min_ohlc_kite
    WHERE tradingsymbol LIKE :pattern
      AND timestamp::date BETWEEN :start_date AND :end_date
    ORDER BY strike_price
    """
    pattern = f"{symbol}%{option_type}"

    with engine.connect() as conn:
        strikes_df = pd.read_sql(text(strikes_query), conn, params={
            "pattern": pattern,
            "start_date": start_date,
            "end_date": end_date,
        })

    strikes_list = strikes_df['strike_price'].dropna().sort_values().astype(int).astype(str).tolist()

    selected_strikes = st.multiselect(
        "Select Strikes",
        options=strikes_list,
        default=strikes_list[:5]
    )

if not selected_strikes:
    st.warning("Please select at least one strike.")
    st.stop()

# Prepare parameters for the IN clause dynamically
strike_params = {f"strike_{i}": strike for i, strike in enumerate(selected_strikes)}
placeholders = ", ".join(f":strike_{i}" for i in range(len(selected_strikes)))

query = f"""
SELECT tradingsymbol, timestamp, close
FROM option_3min_ohlc_kite
WHERE tradingsymbol LIKE :symbol_pattern
AND CAST(strike_price AS TEXT) IN ({placeholders})
AND timestamp::date BETWEEN :start_date AND :end_date
ORDER BY tradingsymbol, timestamp
"""

params = {
    "symbol_pattern": f"{symbol}%{option_type}",
    "start_date": start_date,
    "end_date": end_date,
    **strike_params
}

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn, params=params)

if df.empty:
    st.warning("No data found for given parameters")
    st.stop()

df['timestamp'] = pd.to_datetime(df['timestamp'])

result_df = backtest_40pct_reversal(df)

st.markdown("### Sample Results")
st.dataframe(result_df.head(50))

total_hits = result_df[(result_df['hit_upper_threshold'] | result_df['hit_lower_threshold'])].shape[0]
total_reversals = result_df[result_df['reversal_after_threshold']].shape[0]
reversal_pct = (total_reversals / total_hits * 100) if total_hits else 0

st.markdown(f"**Total Threshold Hits:** {total_hits}")
st.markdown(f"**Reversals After Hits:** {total_reversals}")
st.markdown(f"**Reversal Percentage:** {reversal_pct:.2f}%")

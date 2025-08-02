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
# Function: Generate Complete Trades with Entry, SL, Exit, and PnL
# ------------------------------
def generate_complete_trades(df, price_col='close', time_col='timestamp', strike_col='strike_price'):
    df = df.sort_values(by=[strike_col, time_col]).copy()

    market_open_time = pd.to_datetime(df[time_col].dt.date.min().strftime('%Y-%m-%d') + ' 09:15:00')
    first_candle_close_time = market_open_time + pd.Timedelta(minutes=3)  # 9:18 AM

    trades = []

    for strike, group in df.groupby(strike_col):
        group = group.reset_index(drop=True)

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

            # SHORT trade entry
            if (not short_taken) and (price >= upper_threshold):
                entry_price = price
                stop_loss = entry_price + 0.20 * first_close
                target = first_close  # example target: back to first_close price

                # Scan forward for exit
                exit_price = None
                exit_time = None
                for j in range(i+1, len(group)):
                    future_price = group.loc[j, price_col]
                    future_time = group.loc[j, time_col]

                    # Check SL hit
                    if future_price >= stop_loss:
                        exit_price = stop_loss
                        exit_time = future_time
                        break
                    # Check target hit
                    elif future_price <= target:
                        exit_price = target
                        exit_time = future_time
                        break

                # If no exit found, exit at last available price/time
                if exit_price is None:
                    exit_price = group.iloc[-1][price_col]
                    exit_time = group.iloc[-1][time_col]

                pnl = entry_price - exit_price  # Short profit = entry - exit

                trades.append({
                    'strike_price': strike,
                    'direction': 'SHORT',
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                short_taken = True

            # LONG trade entry
            if (not long_taken) and (price <= lower_threshold):
                entry_price = price
                stop_loss = entry_price - 0.20 * first_close
                target = first_close  # example target: back to first_close price

                # Scan forward for exit
                exit_price = None
                exit_time = None
                for j in range(i+1, len(group)):
                    future_price = group.loc[j, price_col]
                    future_time = group.loc[j, time_col]

                    # Check SL hit
                    if future_price <= stop_loss:
                        exit_price = stop_loss
                        exit_time = future_time
                        break
                    # Check target hit
                    elif future_price >= target:
                        exit_price = target
                        exit_time = future_time
                        break

                # If no exit found, exit at last available price/time
                if exit_price is None:
                    exit_price = group.iloc[-1][price_col]
                    exit_time = group.iloc[-1][time_col]

                pnl = exit_price - entry_price  # Long profit = exit - entry

                trades.append({
                    'strike_price': strike,
                    'direction': 'LONG',
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                long_taken = True

            if short_taken and long_taken:
                break

    return pd.DataFrame(trades)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Intraday 40% Premium Complete Trade Generator")

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

# Generate complete trades
df_trades = generate_complete_trades(df)

if df_trades.empty:
    st.warning("No trades generated for the selected parameters.")
else:
    st.markdown("### Complete Trade Details")
    st.dataframe(df_trades)

    st.markdown(f"Total trades generated: {len(df_trades)}")

    # ------------------------------
    # Monthly Consolidated Summary
    # ------------------------------
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['year_month'] = df_trades['entry_time'].dt.to_period('M')

    monthly_summary = df_trades.groupby('year_month').agg(
        total_trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        winning_trades=('pnl', lambda x: (x > 0).sum())
    ).reset_index()

    monthly_summary['win_rate_pct'] = 100 * monthly_summary['winning_trades'] / monthly_summary['total_trades']

    st.markdown("### Monthly Consolidated Trade Summary")
    st.dataframe(monthly_summary)

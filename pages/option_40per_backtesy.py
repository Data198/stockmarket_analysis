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
# Function: Generate Intraday Trades Based on 40% Move from First Candle Close
# ------------------------------
def generate_intraday_trades(df):
    df = df.sort_values(by=['strike_price', 'option_type', 'timestamp']).copy()
    trades = []

    market_open_time = pd.to_datetime(df['timestamp'].dt.date.min().strftime('%Y-%m-%d') + ' 09:15:00')
    first_candle_close_time = market_open_time + pd.Timedelta(minutes=3)  # 9:18 AM
    market_close_time = pd.to_datetime(df['timestamp'].dt.date.min().strftime('%Y-%m-%d') + ' 15:15:00')

    pe_trade_taken = False
    ce_trade_taken = False

    for (strike, opt_type), group in df.groupby(['strike_price', 'option_type']):
        group = group.reset_index(drop=True)

        first_candle = group[group['timestamp'] >= first_candle_close_time].head(1)
        if first_candle.empty:
            continue

        first_close = first_candle.iloc[0]['close']

        long_pe_trigger = first_close * 1.40  # 40% above first close
        long_ce_trigger = first_close * 0.60  # 40% below first close

        if opt_type == 'PE' and pe_trade_taken:
            continue
        if opt_type == 'CE' and ce_trade_taken:
            continue

        for i, row in group.iterrows():
            current_time = row['timestamp']
            if current_time < first_candle_close_time or current_time > market_close_time:
                continue

            price = row['close']

            if opt_type == 'PE':
                if price >= long_pe_trigger:
                    entry_price = price
                    stop_loss = entry_price * 0.80  # 20% below entry
                    target = entry_price * 1.35     # 35% above entry

                    exit_price, exit_time = None, None
                    for j in range(i+1, len(group)):
                        future_price = group.loc[j, 'close']
                        future_time = group.loc[j, 'timestamp']

                        if future_time > market_close_time:
                            break

                        if future_price <= stop_loss:
                            exit_price = stop_loss
                            exit_time = future_time
                            break
                        elif future_price >= target:
                            exit_price = target
                            exit_time = future_time
                            break

                    if exit_price is None:
                        exit_price = group[group['timestamp'] <= market_close_time].iloc[-1]['close']
                        exit_time = market_close_time

                    pnl = exit_price - entry_price

                    trades.append({
                        'strike_price': strike,
                        'option_type': opt_type,
                        'direction': 'LONG',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    pe_trade_taken = True
                    break

            elif opt_type == 'CE':
                if price <= long_ce_trigger:
                    entry_price = price
                    stop_loss = entry_price * 0.80  # 20% below entry
                    target = entry_price * 1.35     # 35% above entry

                    exit_price, exit_time = None, None
                    for j in range(i+1, len(group)):
                        future_price = group.loc[j, 'close']
                        future_time = group.loc[j, 'timestamp']

                        if future_time > market_close_time:
                            break

                        if future_price <= stop_loss:
                            exit_price = stop_loss
                            exit_time = future_time
                            break
                        elif future_price >= target:
                            exit_price = target
                            exit_time = future_time
                            break

                    if exit_price is None:
                        exit_price = group[group['timestamp'] <= market_close_time].iloc[-1]['close']
                        exit_time = market_close_time

                    pnl = exit_price - entry_price

                    trades.append({
                        'strike_price': strike,
                        'option_type': opt_type,
                        'direction': 'LONG',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    ce_trade_taken = True
                    break

        if pe_trade_taken and ce_trade_taken:
            break

    return pd.DataFrame(trades)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Intraday 40% Premium Trade Generator (PE & CE)")

trade_date_range = st.date_input("Select Trade Date Range",
                                value=(pd.to_datetime("2025-07-25"), pd.to_datetime("2025-07-31")),
                                help="Select start and end date")

symbol = st.text_input("Enter Symbol", value="NIFTY")
expiry_date = st.date_input("Select Expiry Date", value=pd.to_datetime("2025-07-31"))

start_date_str = trade_date_range[0].strftime('%Y-%m-%d')
end_date_str = trade_date_range[1].strftime('%Y-%m-%d')
expiry_date_str = expiry_date.strftime('%Y-%m-%d')

params = {
    "start_date": start_date_str,
    "end_date": end_date_str,
    "symbol": symbol,
    "expiry_date": expiry_date_str
}

st.write("Query Parameters:", params)

query = """
SELECT trade_date, timestamp, symbol, strike_price, option_type, expiry_date,
       open_interest, oi_change_pct, vwap, close, volume, iv, price_change
FROM option_3min_ohlc
WHERE trade_date BETWEEN :start_date AND :end_date
AND symbol = :symbol
AND expiry_date = :expiry_date
ORDER BY strike_price, option_type, timestamp
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

df['timestamp'] = pd.to_datetime(df['timestamp'])

df_trades = generate_intraday_trades(df)

if df_trades.empty:
    st.warning("No trades generated for the selected parameters.")
else:
    st.markdown("### Generated Trades")
    st.dataframe(df_trades)

    # Monthly summary
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

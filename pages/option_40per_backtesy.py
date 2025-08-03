import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# ------------------------------
# DB Connection using Streamlit secrets
# ------------------------------
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

def backtest_40pct_strategy(start_date, end_date, symbol, expiry_date):
    query = """
    SELECT timestamp, tradingsymbol, strike_price, option_type, expiry_date,
           open, high, low, close, volume, oi
    FROM option_3min_ohlc_kite
    WHERE timestamp::date BETWEEN :start_date AND :end_date
      AND tradingsymbol LIKE :symbol_pattern
      AND expiry_date = :expiry_date
    ORDER BY strike_price, option_type, timestamp
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "symbol_pattern": f"{symbol}%",
        "expiry_date": expiry_date
    }

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        st.warning("No data found for given parameters.")
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['trade_date'] = df['timestamp'].dt.date

    trades = []

    grouped = df.groupby(['strike_price', 'option_type', 'trade_date'])

    for (strike, opt_type, trade_date), group in grouped:
        group = group.reset_index(drop=True)

        market_open = pd.to_datetime(f"{trade_date} 09:15:00")
        first_candle_close_time = market_open + pd.Timedelta(minutes=3)
        market_close_time = pd.to_datetime(f"{trade_date} 15:15:00")

        first_candle = group[group['timestamp'] >= first_candle_close_time].head(1)
        if first_candle.empty:
            continue

        high = first_candle.iloc[0]['high']
        low = first_candle.iloc[0]['low']

        long_pe_trigger = low + 0.40 * (high - low)
        long_ce_trigger = high - 0.40 * (high - low)

        for i, row in group.iterrows():
            current_time = row['timestamp']
            if current_time < first_candle_close_time or current_time > market_close_time:
                continue

            price = row['close']

            if opt_type == 'PE' and price >= long_pe_trigger:
                entry_price = price
                stop_loss = entry_price * 0.80
                target = entry_price * 1.35

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

            elif opt_type == 'CE' and price <= long_ce_trigger:
                entry_price = price
                stop_loss = entry_price * 0.80
                target = entry_price * 1.35

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

    return pd.DataFrame(trades)

# Streamlit UI
st.title("40% Premium Intraday Reversal Backtest")

with st.sidebar:
    symbol = st.text_input("Symbol", value="NIFTY")
    expiry_date = st.date_input("Expiry Date", value=pd.to_datetime("2025-08-07"))
    start_date = st.date_input("Start Date", value=pd.to_datetime("2025-07-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-07-31"))

    if start_date > end_date:
        st.error("Start Date must be before End Date")
        st.stop()

if st.button("Run Backtest"):
    trades_df = backtest_40pct_strategy(start_date, end_date, symbol, expiry_date)

    if trades_df.empty:
        st.warning("No trades generated for selected parameters.")
    else:
        st.markdown("### Trade Details")
        st.dataframe(trades_df)

        total_trades = len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df['pnl'] > 0).mean() * 100

        st.markdown(f"**Total Trades:** {total_trades}")
        st.markdown(f"**Total P&L:** {total_pnl:.2f}")
        st.markdown(f"**Win Rate:** {win_rate:.2f}%")

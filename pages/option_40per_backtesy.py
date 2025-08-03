import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from urllib.parse import quote

# DB Connection - your existing secrets usage
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

def backtest_40pct_strategy(start_date, end_date, symbol, expiry_date):
    query = """
    SELECT trade_date, timestamp, symbol, strike_price, option_type, expiry_date,
           open, high, low, close, volume, oi
    FROM option_3min_ohlc_kite
    WHERE trade_date BETWEEN :start_date AND :end_date
      AND symbol = :symbol
      AND expiry_date = :expiry_date
    ORDER BY strike_price, option_type, timestamp
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "symbol": symbol,
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

    # Group by strike, option type, and trade_date for correct intraday analysis
    grouped = df.groupby(['strike_price', 'option_type', 'trade_date'])

    for (strike, opt_type, trade_date), group in grouped:
        group = group.reset_index(drop=True)

        # Define market times per trade_date
        market_open = pd.to_datetime(f"{trade_date} 09:15:00")
        first_candle_close_time = market_open + pd.Timedelta(minutes=3)
        market_close_time = pd.to_datetime(f"{trade_date} 15:15:00")

        # Find first candle after market open + 3 min
        first_candle = group[group['timestamp'] >= first_candle_close_time].head(1)
        if first_candle.empty:
            st.write(f"No first candle for strike {strike}, {opt_type} on {trade_date}")
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

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
    market_open = pd.to_datetime(df['timestamp'].dt.date.min().strftime('%Y-%m-%d') + ' 09:15:00')
    first_candle_close_time = market_open + pd.Timedelta(minutes=3)
    market_close_time = pd.to_datetime(df['timestamp'].dt.date.min().strftime('%Y-%m-%d') + ' 15:15:00')

    trades = []
    pe_trade_taken = False
    ce_trade_taken = False

    for (strike, opt_type), group in df.groupby(['strike_price', 'option_type']):
        group = group.reset_index(drop=True)

        # Fix: select first candle at or after first candle close time
        first_candle = group[group['timestamp'] >= first_candle_close_time].head(1)
        if first_candle.empty:
            print(f"No first candle found for strike {strike} {opt_type}")
            continue

        high = first_candle.iloc[0]['high']
        low = first_candle.iloc[0]['low']

        long_pe_trigger = low + 0.40 * (high - low)
        long_ce_trigger = high - 0.40 * (high - low)

        print(f"Strike: {strike}, Option: {opt_type}, First candle high: {high}, low: {low}")
        print(f"PE trigger: {long_pe_trigger}, CE trigger: {long_ce_trigger}")

        if opt_type == 'PE' and pe_trade_taken:
            continue
        if opt_type == 'CE' and ce_trade_taken:
            continue

        for i, row in group.iterrows():
            current_time = row['timestamp']
            if current_time < first_candle_close_time or current_time > market_close_time:
                continue

            price = row['close']
            print(f"Time: {current_time}, Price: {price}")

            if opt_type == 'PE':
                if price >= long_pe_trigger:
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
                    pe_trade_taken = True
                    break

            elif opt_type == 'CE':
                if price <= long_ce_trigger:
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
                    ce_trade_taken = True
                    break

        if pe_trade_taken and ce_trade_taken:
            break

    return pd.DataFrame(trades)

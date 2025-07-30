import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# Database connection setup
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

st.title("📈 Strategy-1 Backtest: Breakout with Volume & SL/TP")

# Fixed SL and TP combinations to test
sl_values = [20, 40, 60, 80, 100]
tp_values = [20, 40, 60, 80, 100]

if st.button("Run Backtest"):

    # Load data
    level_query = text("SELECT * FROM fact_intraday_levels")
    futures_query = text("SELECT * FROM nifty_3min_ohlc")

    with engine.connect() as conn:
        levels_df = pd.read_sql(level_query, conn)
        futures_df = pd.read_sql(futures_query, conn)

    levels_df["trade_date"] = pd.to_datetime(levels_df["trade_date"])
    futures_df["trade_date"] = pd.to_datetime(futures_df["trade_date"])
    futures_df["timestamp"] = pd.to_datetime(futures_df["timestamp"])

    trade_days = sorted(futures_df["trade_date"].drop_duplicates())
    level_to_apply_map = {
        trade_days[i]: trade_days[i + 1]
        for i in range(len(trade_days) - 1)
    }
    levels_df["apply_date"] = levels_df["trade_date"].map(level_to_apply_map)

    levels_filtered = levels_df[
        (levels_df["apply_date"] >= trade_days[0]) &
        (levels_df["apply_date"] <= trade_days[-1])
    ]

    futures_filtered = futures_df[
        (futures_df["trade_date"] >= trade_days[0]) &
        (futures_df["trade_date"] <= trade_days[-1])
    ]

    joined_df = futures_filtered.merge(
        levels_filtered,
        left_on="trade_date",
        right_on="apply_date",
        how="left",
        suffixes=('', '_level')
    ).sort_values("timestamp")

    if "volume" not in joined_df.columns:
        st.error("Error: 'volume' column not found in merged data. Please check your data source.")
        st.stop()

    joined_df = joined_df.dropna(subset=["volume"])
    joined_df["volume_sma"] = joined_df.groupby("trade_date")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )

    results_matrix = pd.DataFrame(index=sl_values, columns=tp_values)
    best_combo = None
    best_avg_pnl = float('-inf')

    def backtest_for_sl_tp(sl_points, tp_points):
        trades = []
        for trade_date, group in joined_df.groupby("trade_date"):
            triggered = False
            for i in range(len(group)):
                row = group.iloc[i]
                if pd.isna(row.get("r1")) or pd.isna(row.get("s1")):
                    continue
                if triggered or pd.isna(row["volume_sma"]):
                    continue
                if row["volume"] > row["volume_sma"]:
                    entry_price = row["close"]
                    if row["close"] > row["r1"]:
                        direction = "Long"
                        sl = entry_price - sl_points
                        tp = entry_price + tp_points
                    elif row["close"] < row["s1"]:
                        direction = "Short"
                        sl = entry_price + sl_points
                        tp = entry_price - tp_points
                    else:
                        continue
                    entry_time = row["timestamp"]
                    sub_df = group.iloc[i + 1:]
                    for _, exit_row in sub_df.iterrows():
                        price = exit_row["close"]
                        if (direction == "Long" and price >= tp) or (direction == "Short" and price <= tp):
                            trades.append([trade_date, direction, entry_time, entry_price, exit_row["timestamp"], price, "Target"])
                            triggered = True
                            break
                        elif (direction == "Long" and price <= sl) or (direction == "Short" and price >= sl):
                            trades.append([trade_date, direction, entry_time, entry_price, exit_row["timestamp"], price, "Stop"])
                            triggered = True
                            break
                    if not triggered and not sub_df.empty:
                        trades.append([trade_date, direction, entry_time, entry_price, sub_df.iloc[-1]["timestamp"], sub_df.iloc[-1]["close"], "EOD"])
                        triggered = True

        results_df = pd.DataFrame(trades, columns=[
            "Date", "Direction", "Entry Time", "Entry Price", "Exit Time", "Exit Price", "Exit Reason"
        ])
        if results_df.empty:
            return 0, 0, 0

        results_df["PnL"] = results_df.apply(
            lambda x: x["Exit Price"] - x["Entry Price"] if x["Direction"] == "Long"
            else x["Entry Price"] - x["Exit Price"], axis=1
        )
        total = len(results_df)
        wins = (results_df["PnL"] > 0).sum()
        avg_pnl = results_df["PnL"].mean()
        win_rate = (wins / total) * 100
        return total, win_rate, avg_pnl

    progress_bar = st.progress(0)
    total_combinations = len(sl_values) * len(tp_values)
    count = 0

    for sl in sl_values:
        for tp in tp_values:
            total, win_rate, avg_pnl = backtest_for_sl_tp(sl, tp)
            results_matrix.at[sl, tp] = f"Trades:{total}\nWin%:{win_rate:.1f}\nAvgPnL:{avg_pnl:.2f}"
            if avg_pnl > best_avg_pnl:
                best_avg_pnl = avg_pnl
                best_combo = (sl, tp)
            count += 1
            progress_bar.progress(count / total_combinations)

    st.subheader("Backtest Results Matrix (Stop Loss vs Target)")
    st.dataframe(results_matrix)

    if best_combo:
        st.success(f"Best combination: Stop Loss = {best_combo[0]} pts, Target = {best_combo[1]} pts with Average PnL = {best_avg_pnl:.2f}")

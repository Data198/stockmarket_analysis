import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# -------------------------
# 🔐 Database Connection
# -------------------------
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -------------------------
# 🎯 Page Title
# -------------------------
st.title("📈 Strategy-1 Backtest: Breakout with Volume & SL/TP")

# -------------------------
# Sidebar Filters
# -------------------------
with st.sidebar:
    st.header("🔎 Filters")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    volume_window = st.slider("Volume SMA Period", 5, 50, 20)
    sl_points = st.number_input("Stop Loss (pts)", value=60)
    tp_points = st.number_input("Target (pts)", value=60)

# -------------------------
# Main Analysis Trigger
# -------------------------
if st.button("Run Backtest"):
    # Load level data and futures data
    level_query = text("SELECT * FROM fact_intraday_levels")
    futures_query = text("SELECT * FROM nifty_3min_ohlc")

    with engine.connect() as conn:
        levels_df = pd.read_sql(level_query, conn)
        futures_df = pd.read_sql(futures_query, conn)

    # Preprocess dates
    levels_df["trade_date"] = pd.to_datetime(levels_df["trade_date"])
    futures_df["trade_date"] = pd.to_datetime(futures_df["trade_date"])
    futures_df["timestamp"] = pd.to_datetime(futures_df["timestamp"])

    # Match level dates to next valid futures day
    trade_days = sorted(futures_df["trade_date"].drop_duplicates())
    level_to_apply_map = {
        trade_days[i]: trade_days[i + 1]
        for i in range(len(trade_days) - 1)
    }
    levels_df["apply_date"] = levels_df["trade_date"].map(level_to_apply_map)

    # Filter by date
    levels_filtered = levels_df[
        (levels_df["apply_date"] >= pd.to_datetime(start_date)) &
        (levels_df["apply_date"] <= pd.to_datetime(end_date))
    ]

    futures_filtered = futures_df[
        (futures_df["trade_date"] >= pd.to_datetime(start_date)) &
        (futures_df["trade_date"] <= pd.to_datetime(end_date))
    ]

    # Join levels with futures
    joined_df = futures_filtered.merge(
        levels_filtered,
        left_on="trade_date",
        right_on="apply_date",
        how="left",
        suffixes=('', '_level')
    ).sort_values("timestamp")

    # Debug: Check columns to ensure 'volume' exists
    st.write("Columns in joined_df:", joined_df.columns.tolist())

    if "volume" not in joined_df.columns:
        st.error("Error: 'volume' column not found in merged data. Please check your data source.")
    else:
        # Drop rows where volume is NaN to avoid errors in rolling calculation
        joined_df = joined_df.dropna(subset=["volume"])

        # Calculate rolling volume SMA grouped by trade_date
        joined_df["volume_sma"] = joined_df.groupby("trade_date")["volume"].transform(
            lambda x: x.rolling(volume_window, min_periods=1).mean()
        )

        # Proceed with your backtest logic here...
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

        # Create trade log
        results_df = pd.DataFrame(trades, columns=[
            "Date", "Direction", "Entry Time", "Entry Price", "Exit Time", "Exit Price", "Exit Reason"
        ])
        results_df["PnL"] = results_df.apply(
            lambda x: x["Exit Price"] - x["Entry Price"] if x["Direction"] == "Long"
            else x["Entry Price"] - x["Exit Price"], axis=1
        )

        # Show results
        st.subheader("📋 Trade Log")
        st.dataframe(results_df)

        # Summary
        total = len(results_df)
        wins = (results_df["PnL"] > 0).sum()
        avg_pnl = results_df["PnL"].mean() if total else 0
        win_rate = (wins / total) * 100 if total else 0

        st.subheader("📈 Backtest Summary")
        st.markdown(f"- **Total Trades**: {total}")
        st.markdown(f"- **Win Rate**: {win_rate:.2f}%")
        st.markdown(f"- **Average PnL**: {avg_pnl:.2f} points")

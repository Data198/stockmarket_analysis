import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote
from login_page import login
from datetime import date, timedelta

# ------------------------------
# 🔐 Enforce login
# ------------------------------
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

if not login(engine):
    st.stop()

st.title("🧠 Market Sentiment Tagging")

# ------------------------------
# 🗓️ Trade Date & Symbol Input
# ------------------------------
with engine.connect() as conn:
    dates = conn.execute(text("SELECT DISTINCT trade_date FROM fact_oi_bhavcopy ORDER BY trade_date DESC")).fetchall()
    symbols = conn.execute(text("SELECT DISTINCT symbol FROM fact_oi_bhavcopy ORDER BY symbol")).fetchall()

available_dates = sorted([d[0] for d in dates])
default_date = available_dates[0] if available_dates else date.today()

trade_date = st.sidebar.date_input("Select Trade Date", default_date,
                                   min_value=min(available_dates), max_value=max(available_dates),
                                   key="sentiment_trade_date")

symbol = st.sidebar.selectbox("Select Symbol", [s[0] for s in symbols], key="sentiment_symbol")

# ------------------------------
# 📆 Expiry Selector
# ------------------------------
with engine.connect() as conn:
    expiries = conn.execute(text("""
        SELECT DISTINCT expiry_date 
        FROM fact_oi_bhavcopy 
        WHERE trade_date = :date AND symbol = :symbol 
        ORDER BY expiry_date
    """), {"date": trade_date, "symbol": symbol}).fetchall()

expiry_options = sorted([e[0] for e in expiries])

if expiry_options:
    default_expiry = expiry_options[0]
    expiry_date = st.sidebar.date_input("Select Expiry Date", default_expiry,
                                        min_value=min(expiry_options), max_value=max(expiry_options),
                                        key="sentiment_expiry_date")
else:
    st.sidebar.warning("No expiry dates found.")
    st.stop()

# ------------------------------
# 📊 Load data
# ------------------------------
query = """
SELECT * FROM fact_oi_bhavcopy
WHERE trade_date = :date AND symbol = :symbol AND expiry_date = :expiry
"""

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn, params={
        "date": trade_date,
        "symbol": symbol,
        "expiry": expiry_date
    })

if df.empty:
    st.warning("No data available for the selected combination.")
    st.stop()

# ------------------------------
# 🔢 Concentration Calculations
# ------------------------------
top5_ce = df[(df['option_type'] == 'CE') & (df['ce_oi_change'] > 0)].nlargest(5, 'ce_oi_change')
top5_pe = df[(df['option_type'] == 'PE') & (df['pe_oi_change'] > 0)].nlargest(5, 'pe_oi_change')

ce_total = df[(df['option_type'] == 'CE') & (df['ce_oi_change'] > 0)]['ce_oi_change'].sum()
pe_total = df[(df['option_type'] == 'PE') & (df['pe_oi_change'] > 0)]['pe_oi_change'].sum()

ce_concentration_pct_5 = round((top5_ce['ce_oi_change'].sum() / ce_total) * 100, 2) if ce_total > 0 else 0
pe_concentration_pct_5 = round((top5_pe['pe_oi_change'].sum() / pe_total) * 100, 2) if pe_total > 0 else 0

# ------------------------------
# 🧠 AI-based Sentiment Logic
# ------------------------------
diff = ce_concentration_pct_5 - pe_concentration_pct_5

if ce_concentration_pct_5 > 60 and pe_concentration_pct_5 < 40:
    sentiment_tag = "🔴 High Call Pressure (Bearish Sentiment)"
elif pe_concentration_pct_5 > 60 and ce_concentration_pct_5 < 40:
    sentiment_tag = "🟢 High Put Pressure (Bullish Sentiment)"
elif abs(diff) <= 15 and ce_concentration_pct_5 > 40 and pe_concentration_pct_5 > 40:
    sentiment_tag = "⚖️ Balanced Pressure – Tug of War"
elif diff >= 25:
    sentiment_tag = "📉 Bearish Bias – Dominant Call Writers"
elif diff <= -25:
    sentiment_tag = "📈 Bullish Bias – Dominant Put Writers"
else:
    sentiment_tag = "🤔 Neutral or Unclear Bias"

# ------------------------------
# 💬 Display Sentiment
# ------------------------------
st.markdown(
    f"""
    <div style="background-color:#fff9e6;padding:15px;border-left:6px solid #ffcc00;
                border-radius:8px;margin-top:20px;font-size:16px;">
        <strong>Market Sentiment:</strong><br>
        {sentiment_tag}
    </div>
    """,
    unsafe_allow_html=True
)

# Get previous trade date
prev_date_query = """
SELECT MAX(trade_date) 
FROM fact_oi_bhavcopy 
WHERE trade_date < :curr_date
"""
with engine.connect() as conn:
    prev_trade_date = conn.execute(text(prev_date_query), {"curr_date": trade_date}).scalar()

if prev_trade_date:
    prev_query = """
    SELECT * FROM fact_oi_bhavcopy
    WHERE trade_date = :date AND symbol = :symbol AND expiry_date = :expiry
    """
    with engine.connect() as conn:
        prev_df = pd.read_sql(text(prev_query), conn, params={
            "date": prev_trade_date,
            "symbol": symbol,
            "expiry": expiry_date
        })

    if not prev_df.empty:
        prev_ce_df = prev_df[(prev_df['option_type'] == 'CE') & (prev_df['ce_oi_change'] > 0)]
        prev_pe_df = prev_df[(prev_df['option_type'] == 'PE') & (prev_df['pe_oi_change'] > 0)]

        prev_top5_ce = prev_ce_df.nlargest(5, 'ce_oi_change')
        prev_top5_pe = prev_pe_df.nlargest(5, 'pe_oi_change')

        prev_ce_total = prev_ce_df['ce_oi_change'].sum()
        prev_pe_total = prev_pe_df['pe_oi_change'].sum()

        prev_ce_pct = round((prev_top5_ce['ce_oi_change'].sum() / prev_ce_total) * 100, 2) if prev_ce_total > 0 else 0
        prev_pe_pct = round((prev_top5_pe['pe_oi_change'].sum() / prev_pe_total) * 100, 2) if prev_pe_total > 0 else 0

        # Display trend table
        st.markdown("### 📈 Trend Comparison with Previous Day")
        curr_date_str = trade_date.strftime('%Y-%m-%d')
        prev_date_str = prev_trade_date.strftime('%Y-%m-%d')

        st.markdown(
            f"""
            <table style='width:70%; border-collapse: collapse; text-align: center; margin-top: 10px;'>
                <tr style='background-color:#1f4e79; color:white; font-weight:bold;'>
                    <td>Date</td><td>Top 5 CE OI %</td><td>Top 5 PE OI %</td>
                </tr>
                <tr style='background-color:#e6f3ff;'>
                    <td>{curr_date_str}</td><td>{ce_concentration_pct_5} %</td><td>{pe_concentration_pct_5} %</td>
                </tr>
                <tr style='background-color:#f4f4f4;'>
                    <td>{prev_date_str}</td><td>{prev_ce_pct} %</td><td>{prev_pe_pct} %</td>
                </tr>
            </table>
            """, unsafe_allow_html=True
        )
    else:
        st.info(f"No data available for previous date: {prev_trade_date}")
else:
    st.info("Previous trade date not found.")

# Optional debugging info
with st.expander("📊 Show Calculations"):
    st.write(f"Top 5 CE OI Concentration: {ce_concentration_pct_5} %")
    st.write(f"Top 5 PE OI Concentration: {pe_concentration_pct_5} %")
    st.write(f"Difference (CE - PE): {diff} %")

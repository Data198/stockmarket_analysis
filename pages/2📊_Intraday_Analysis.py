import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote
from login_page import login
import matplotlib.pyplot as plt
from datetime import date

# Load secrets from .streamlit/secrets.toml
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# 🔐 Enforce login
if not login(engine):
    st.stop()

# 🌟 Personalized welcome
formatted_name = st.session_state.username.replace("_", " ").title()
st.sidebar.markdown(f"**Welcome, {formatted_name} 👋**")

st.title("Intraday Parameters Dashboard")

# ------------------------------
# 🗓️ Trade Date & Symbol Selectors
# ------------------------------
with engine.connect() as conn:
    dates = conn.execute(text("SELECT DISTINCT trade_date FROM fact_oi_bhavcopy ORDER BY trade_date DESC")).fetchall()
    symbols = conn.execute(text("SELECT DISTINCT symbol FROM fact_oi_bhavcopy ORDER BY symbol")).fetchall()

available_dates = sorted([d[0] for d in dates])

if not available_dates:
    st.warning("No trade dates found in the database.")
    st.stop()

default_date = available_dates[0]

trade_date = st.sidebar.date_input("Select Trade Date", default_date,
                                   min_value=min(available_dates), max_value=max(available_dates),
                                   key="trade_date_picker")

symbol = st.sidebar.selectbox("Select Symbol", [s[0] for s in symbols], key="symbol_selector")

# ------------------------------
# 📆 Expiry Selector (with fallback)
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
                                        key="expiry_date_picker")
else:
    st.sidebar.warning("No expiry dates found for the selected symbol and date.")
    st.stop()

# ------------------------------
# 📊 Load Data for Selected Inputs
# ------------------------------
query = """
SELECT * FROM fact_oi_bhavcopy
WHERE trade_date = :date AND symbol = :symbol AND expiry_date = :expiry
ORDER BY strike_price
"""
# 📉 Get previous trade date
previous_date_query = """
    SELECT MAX(trade_date) FROM fact_oi_bhavcopy 
    WHERE trade_date < :date AND symbol = :symbol
"""
with engine.connect() as conn:
    prev_date_result = conn.execute(text(previous_date_query), {
        "date": trade_date,
        "symbol": symbol
    }).fetchone()

prev_trade_date = prev_date_result[0] if prev_date_result else None

# 📊 Load both current and previous day data within the same connection block
df_prev = pd.DataFrame()
df = pd.DataFrame()

with engine.connect() as conn:
    if prev_trade_date:
        df_prev = pd.read_sql(text(query), conn, params={
            "date": prev_trade_date,
            "symbol": symbol,
            "expiry": expiry_date
        })
    df = pd.read_sql(text(query), conn, params={
        "date": trade_date,
        "symbol": symbol,
        "expiry": expiry_date
    })

if df.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# 📌 Resistance Levels from Top CE OI Changes
top5_ce = df[(df['option_type'] == 'CE') & (df['ce_oi_change'].notnull())].nlargest(5, 'ce_oi_change')
top3_ce = top5_ce.head(3)
top2_ce = top5_ce.head(2)

r1 = round(top5_ce['resistance'].mean())
r2 = round(top3_ce['resistance'].mean())
r3 = round(top2_ce['resistance'].mean())

# 📌 Support Levels from Top PE OI Changes
top5_pe = df[(df['option_type'] == 'PE') & (df['pe_oi_change'].notnull())].nlargest(5, 'pe_oi_change')
top3_pe = top5_pe.head(3)
top2_pe = top5_pe.head(2)

s1 = round(top5_pe['support'].mean())
s2 = round(top3_pe['support'].mean())
s3 = round(top2_pe['support'].mean())

# 🧩 Relevant Strike Prices
ce1, ce2, ce3 = [int(x) for x in top5_ce['strike_price'].head(3)]
pe1, pe2, pe3 = [int(x) for x in top5_pe['strike_price'].head(3)]

# ✅ Auto-insert to fact_intraday_levels
if st.button("✅ Save Intraday Levels to Database"):
    insert_query = text("""
    INSERT INTO fact_intraday_levels (
        trade_date, symbol, r1, r2, r3, s1, s2, s3,
        ce_1, ce_2, ce_3, pe_1, pe_2, pe_3
    )
    VALUES (
        :trade_date, :symbol, :r1, :r2, :r3, :s1, :s2, :s3,
        :ce1, :ce2, :ce3, :pe1, :pe2, :pe3
    )
    ON CONFLICT (trade_date, symbol) DO UPDATE SET
        r1 = EXCLUDED.r1, r2 = EXCLUDED.r2, r3 = EXCLUDED.r3,
        s1 = EXCLUDED.s1, s2 = EXCLUDED.s2, s3 = EXCLUDED.s3,
        ce_1 = EXCLUDED.ce_1, ce_2 = EXCLUDED.ce_2, ce_3 = EXCLUDED.ce_3,
        pe_1 = EXCLUDED.pe_1, pe_2 = EXCLUDED.pe_2, pe_3 = EXCLUDED.pe_3;
    """)

    with engine.begin() as conn:
        conn.execute(insert_query, {
            "trade_date": trade_date,
            "symbol": symbol,
            "r1": r1, "r2": r2, "r3": r3,
            "s1": s1, "s2": s2, "s3": s3,
            "ce1": ce1, "ce2": ce2, "ce3": ce3,
            "pe1": pe1, "pe2": pe2, "pe3": pe3
        })

    st.success(f"✅ Intraday levels for {symbol} on {trade_date} saved to database.")


# ✅ CE OI Concentration Calculations
positive_ce_oi = df[(df['option_type'] == 'CE') & (df['ce_oi_change'] > 0)]
ce_concentration_pct_5 = ce_concentration_pct_3 = ce_concentration_pct_2 = 0

if not positive_ce_oi.empty:
    ce_concentration_pct_5 = round((top5_ce['ce_oi_change'].sum() / positive_ce_oi['ce_oi_change'].sum()) * 100, 2)
    ce_concentration_pct_3 = round((top3_ce['ce_oi_change'].sum() / positive_ce_oi['ce_oi_change'].sum()) * 100, 2)
    ce_concentration_pct_2 = round((top2_ce['ce_oi_change'].sum() / positive_ce_oi['ce_oi_change'].sum()) * 100, 2)

# ✅ PE OI Concentration Calculations
positive_pe_oi = df[(df['option_type'] == 'PE') & (df['pe_oi_change'] > 0)]
pe_concentration_pct_5 = pe_concentration_pct_3 = pe_concentration_pct_2 = 0

if not positive_pe_oi.empty:
    pe_concentration_pct_5 = round((top5_pe['pe_oi_change'].sum() / positive_pe_oi['pe_oi_change'].sum()) * 100, 2)
    pe_concentration_pct_3 = round((top3_pe['pe_oi_change'].sum() / positive_pe_oi['pe_oi_change'].sum()) * 100, 2)
    pe_concentration_pct_2 = round((top2_pe['pe_oi_change'].sum() / positive_pe_oi['pe_oi_change'].sum()) * 100, 2)


# Card-style visual for CE OI Concentration %

st.markdown("### CE OI Concentration (%)")
st.markdown(f"""
<div style="display: flex; justify-content: flex-start; gap: 20px;">
    <div style="background-color:#e6f3ff;padding:15px 10px;border-radius:12px;text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);width:200px;">
        <h5 style="color:#004080;margin:5px 0;">Top 5 CE OI %</h5>
        <h3 style="color:#003399;">{ce_concentration_pct_5} %</h3>
    </div>
    <div style="background-color:#e6f3ff;padding:15px 10px;border-radius:12px;text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);width:200px;">
        <h5 style="color:#004080;margin:5px 0;">Top 3 CE OI %</h5>
        <h3 style="color:#003399;">{ce_concentration_pct_3} %</h3>
    </div>
    <div style="background-color:#e6f3ff;padding:15px 10px;border-radius:12px;text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);width:200px;">
        <h5 style="color:#004080;margin:5px 0;">Top 2 CE OI %</h5>
        <h3 style="color:#003399;">{ce_concentration_pct_2} %</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# Card-style visual for PE OI Concentration %

st.markdown("### PE OI Concentration (%)")
st.markdown(f"""
<div style="display: flex; justify-content: flex-start; gap: 20px;">
    <div style="background-color:#fff2e6;padding:15px 10px;border-radius:12px;text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);width:200px;">
        <h5 style="color:#994d00;margin:5px 0;">Top 5 PE OI %</h5>
        <h3 style="color:#cc5200;">{pe_concentration_pct_5} %</h3>
    </div>
    <div style="background-color:#fff2e6;padding:15px 10px;border-radius:12px;text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);width:200px;">
        <h5 style="color:#994d00;margin:5px 0;">Top 3 PE OI %</h5>
        <h3 style="color:#cc5200;">{pe_concentration_pct_3} %</h3>
    </div>
    <div style="background-color:#fff2e6;padding:15px 10px;border-radius:12px;text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);width:200px;">
        <h5 style="color:#994d00;margin:5px 0;">Top 2 PE OI %</h5>
        <h3 style="color:#cc5200;">{pe_concentration_pct_2} %</h3>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------
# 🔷 Display Cards
# ------------------------------
st.markdown("### Key Resistance Levels")
st.markdown(
    f"""
    <table style='width:80%; border-collapse: collapse; text-align: center;'>
        <tr style='background-color:#1f4e79; color:white; font-weight:bold;'>
            <td>R1</td><td>R2</td><td>R3</td>
        </tr>
        <tr style='background-color:#e9edf6; color:#000; font-size:18px;'>
            <td>{r1}</td><td>{r2}</td><td>{r3}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True
)

st.markdown("### Key Support Levels")
st.markdown(
    f"""
    <table style='width:80%; border-collapse: collapse; text-align: center;'>
        <tr style='background-color:#a9d18e; color:black; font-weight:bold;'>
            <td>S1</td><td>S2</td><td>S3</td>
        </tr>
        <tr style='background-color:#f4f9f4; color:#000; font-size:18px;'>
            <td>{s1}</td><td>{s2}</td><td>{s3}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True
)

# 🎯 Relevant Strikes
st.markdown("### Relevant Strike Prices")
st.markdown(
    f"""
    <table style='width:80%; border-collapse: collapse; text-align: center;'>
        <tr style='background-color:#1f4e79; color:white; font-weight:bold;'>
            <td>CE-1</td><td>CE-2</td><td>CE-3</td>
        </tr>
        <tr style='background-color:#e9edf6; color:#000; font-size:18px;'>
            <td>{ce1}</td><td>{ce2}</td><td>{ce3}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True
)

st.markdown(
    f"""
    <table style='width:80%; border-collapse: collapse; text-align: center;'>
        <tr style='background-color:#a9d18e; color:black; font-weight:bold;'>
            <td>PE-1</td><td>PE-2</td><td>PE-3</td>
        </tr>
        <tr style='background-color:#f4f9f4; color:#000; font-size:18px;'>
            <td>{pe1}</td><td>{pe2}</td><td>{pe3}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True
)

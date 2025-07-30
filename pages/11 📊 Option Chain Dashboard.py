import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# --- DB Setup ---
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- UI Layout ---
st.set_page_config(page_title="Option Chain Dashboard", layout="wide")
st.title("📊 Option Chain Dashboard")

# --- Date Filter ---
with st.sidebar:
    st.header("📅 Select Filters")
    with engine.connect() as conn:
        dates = conn.execute(text("SELECT DISTINCT trade_date FROM fact_option_chain ORDER BY trade_date DESC")).fetchall()
        expiries = conn.execute(text("SELECT DISTINCT expiry_date FROM fact_option_chain ORDER BY expiry_date DESC")).fetchall()
    date_options = [r[0] for r in dates]
    expiry_options = [r[0] for r in expiries]
    trade_date = st.selectbox("Trade Date", date_options)
    expiry_date = st.selectbox("Expiry Date", expiry_options)

# --- Fetch Data ---
query = text("""
    SELECT * FROM fact_option_chain
    WHERE trade_date = :trade_date AND expiry_date = :expiry_date
    ORDER BY strike
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn, params={"trade_date": trade_date, "expiry_date": expiry_date})

if df.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# --- Tabs for Analysis ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Strike Overview", "📉 VaR Analysis", "📊 Pinning Zones", "🤖 Readiness Tag"])

# --- Tab 1: Strike Overview ---
with tab1:
    st.subheader("🔍 Strike-wise Sensitivities")
    st.dataframe(df[['strike', 'option_type', 'premium', 'delta', 'gamma', 'vega']])

# --- Tab 2: VaR Analysis ---
with tab2:
    st.subheader("📉 Sensitivity-Based VaR")
    spot_price = st.number_input("Spot Price", value=24843)
    delta_s = spot_price * 0.01
    vol_shock = 0.02

    df['VaR'] = df['delta'] * delta_s + 0.5 * df['gamma'] * delta_s**2 + df['vega'] * vol_shock
    df['VaR_pct'] = df['VaR'] / df['premium'] * 100
    df['readiness_score'] = df.apply(lambda row: int(0.5 <= abs(row['delta']) <= 0.7) + int(row['VaR_pct'] <= 80), axis=1)

    st.dataframe(df[['strike', 'option_type', 'premium', 'delta', 'VaR', 'VaR_pct', 'readiness_score']])

# --- Tab 3: Pinning Zones ---
with tab3:
    st.subheader("📊 Expiry Pinning Bias")
    pin_df = df[(df['option_type'] == 'CE') & (df['delta'].between(0.45, 0.55))]
    st.write("🔵 CE strikes near delta 0.5 (likely pin zone):")
    st.dataframe(pin_df[['strike', 'premium', 'delta', 'VaR']])

# --- Tab 4: Readiness Tag ---
with tab4:
    st.subheader("🤖 AI-Style Readiness Tags")
    def tag_action(row):
        if row['readiness_score'] == 2:
            return "✅ BUY"
        elif row['readiness_score'] == 1:
            return "⚠️ WATCH"
        else:
            return "🚫 AVOID"

    df['readiness_tag'] = df.apply(tag_action, axis=1)
    st.dataframe(df[['strike', 'option_type', 'premium', 'VaR_pct', 'readiness_score', 'readiness_tag']])

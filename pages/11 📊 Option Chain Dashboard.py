import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
st.set_page_config(page_title="Option Chain Trading Plan Dashboard", layout="wide")
st.title("📊 Option Chain Trading Plan Dashboard")

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

# --- Input for Spot Price ---
spot_price = st.sidebar.number_input("Spot Price", value=24843)

# --- Calculate VaR and Readiness Score ---
delta_s = spot_price * 0.01
vol_shock = 0.02

df['VaR'] = df['delta'] * delta_s + 0.5 * df['gamma'] * delta_s**2 + df['vega'] * vol_shock
df['VaR_pct'] = df['VaR'] / df['premium'] * 100

# Readiness score based on delta and VaR_pct
df['readiness_score'] = df.apply(
    lambda row: int(0.5 <= abs(row['delta']) <= 0.7) + int(row['VaR_pct'] <= 80), axis=1
)

# --- Trade Strategy Suggestion ---
def suggest_strategy(row):
    if row['readiness_score'] == 2 and row['delta'] > 0:
        return "Long Call"
    elif row['readiness_score'] == 2 and row['delta'] < 0:
        return "Long Put"
    elif row['readiness_score'] == 1 and row['VaR_pct'] > 50:
        return "Watch - Possible Short"
    else:
        return "Avoid"

df['trade_strategy'] = df.apply(suggest_strategy, axis=1)

# --- Risk Management: Stop Loss and Target ---
# Example: Stop loss = 30% of premium, Target = 60% of premium (can be adjusted)
df['stop_loss'] = df['premium'] * 0.3
df['target'] = df['premium'] * 0.6

# --- Color coding readiness tags ---
def readiness_tag(row):
    if row['readiness_score'] == 2:
        return "✅ BUY"
    elif row['readiness_score'] == 1:
        return "⚠️ WATCH"
    else:
        return "🚫 AVOID"

df['readiness_tag'] = df.apply(readiness_tag, axis=1)

# --- Sidebar Filters for Option Type, OI, Volume ---
st.sidebar.header("🔎 Additional Filters")
option_types = df['option_type'].unique().tolist()
selected_option_types = st.sidebar.multiselect("Option Type", option_types, default=option_types)

min_oi = int(df['open_interest'].min()) if 'open_interest' in df.columns else 0
max_oi = int(df['open_interest'].max()) if 'open_interest' in df.columns else 0
oi_filter = st.sidebar.slider("Open Interest Range", min_oi, max_oi, (min_oi, max_oi)) if 'open_interest' in df.columns else None

min_vol = int(df['volume'].min()) if 'volume' in df.columns else 0
max_vol = int(df['volume'].max()) if 'volume' in df.columns else 0
vol_filter = st.sidebar.slider("Volume Range", min_vol, max_vol, (min_vol, max_vol)) if 'volume' in df.columns else None

# Apply filters
filtered_df = df[df['option_type'].isin(selected_option_types)]
if oi_filter:
    filtered_df = filtered_df[(filtered_df['open_interest'] >= oi_filter[0]) & (filtered_df['open_interest'] <= oi_filter[1])]
if vol_filter:
    filtered_df = filtered_df[(filtered_df['volume'] >= vol_filter[0]) & (filtered_df['volume'] <= vol_filter[1])]

# --- Tabs for Analysis ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Strike Overview",
    "📉 VaR Analysis",
    "📊 Pinning Zones",
    "🤖 Readiness & Strategy",
    "📈 Visualizations"
])

# --- Tab 1: Strike Overview ---
with tab1:
    st.subheader("🔍 Strike-wise Sensitivities & Metrics")
    st.dataframe(filtered_df[['strike', 'option_type', 'premium', 'delta', 'gamma', 'vega', 'open_interest', 'volume']])

# --- Tab 2: VaR Analysis ---
with tab2:
    st.subheader("📉 Sensitivity-Based VaR & Risk Metrics")
    st.dataframe(filtered_df[['strike', 'option_type', 'premium', 'delta', 'VaR', 'VaR_pct', 'stop_loss', 'target']])

# --- Tab 3: Pinning Zones ---
with tab3:
    st.subheader("📊 Expiry Pinning Bias (CE strikes near delta 0.5)")
    pin_df = filtered_df[(filtered_df['option_type'] == 'CE') & (filtered_df['delta'].between(0.45, 0.55))]
    if pin_df.empty:
        st.info("No pinning zone strikes found for selected filters.")
    else:
        st.dataframe(pin_df[['strike', 'premium', 'delta', 'VaR', 'open_interest', 'volume']])

# --- Tab 4: Readiness & Strategy ---
with tab4:
    st.subheader("🤖 AI-Style Readiness Tags & Trade Strategies")
    # Color code readiness_tag column
    def color_readiness(val):
        if val == "✅ BUY":
            color = 'green'
        elif val == "⚠️ WATCH":
            color = 'orange'
        else:
            color = 'red'
        return f'color: {color}'

    st.dataframe(
        filtered_df[['strike', 'option_type', 'premium', 'VaR_pct', 'readiness_score', 'readiness_tag', 'trade_strategy']]
        .style.applymap(color_readiness, subset=['readiness_tag'])
    )

# --- Tab 5: Visualizations ---
with tab5:
    st.subheader("📈 Visualizations")

    # Greeks vs Strike
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=filtered_df, x='strike', y='delta', hue='option_type', ax=ax)
    ax.set_title("Delta vs Strike")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=filtered_df, x='strike', y='gamma', hue='option_type', ax=ax2)
    ax2.set_title("Gamma vs Strike")
    st.pyplot(fig2)

    # VaR distribution
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df['VaR_pct'], bins=30, kde=True, ax=ax3)
    ax3.set_title("VaR % Distribution")
    st.pyplot(fig3)

    # Open Interest Heatmap (if available)
    if 'open_interest' in filtered_df.columns:
        oi_pivot = filtered_df.pivot_table(index='option_type', columns='strike', values='open_interest', fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        sns.heatmap(oi_pivot, cmap="YlGnBu", ax=ax4)
        ax4.set_title("Open Interest Heatmap")
        st.pyplot(fig4)

    # Volume Heatmap (if available)
    if 'volume' in filtered_df.columns:
        vol_pivot = filtered_df.pivot_table(index='option_type', columns='strike', values='volume', fill_value=0)
        fig5, ax5 = plt.subplots(figsize=(12, 4))
        sns.heatmap(vol_pivot, cmap="YlOrRd", ax=ax5)
        ax5.set_title("Volume Heatmap")
        st.pyplot(fig5)

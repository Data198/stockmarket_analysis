import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from urllib.parse import quote
import plotly.graph_objects as go

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
st.title("📈 Multi-Day OI & Price Trend with Auto Support/Resistance and Signals")

# -------------------------
# Sidebar Filters
# -------------------------
with st.sidebar:
    st.header("🔎 Filters")
    symbol = st.text_input("Symbol", value="NIFTY")
    strike_price = st.number_input("Strike Price", value=25100)
    option_type = st.selectbox("Option Type", ["CE", "PE"])
    expiry_date = st.date_input("Expiry Date")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

# -------------------------
# Helper Functions
# -------------------------

def detect_oi_support_resistance(symbol, option_type, expiry_date, start_date, end_date, top_n=3):
    """
    Fetch OI data for all strikes on expiry date and find top N strikes by OI.
    These strikes act as key support/resistance zones.
    """
    query = text(
        """
        SELECT strike_price, SUM(open_interest) as total_oi
        FROM option_3min_ohlc
        WHERE symbol = :symbol
          AND option_type = :otype
          AND expiry_date = :expiry
          AND trade_date BETWEEN :start AND :end
        GROUP BY strike_price
        ORDER BY total_oi DESC
        LIMIT :top_n
        """
    )
    with engine.connect() as conn:
        oi_df = pd.read_sql(
            query,
            conn,
            params={
                "symbol": symbol,
                "otype": option_type,
                "expiry": expiry_date,
                "start": start_date,
                "end": end_date,
                "top_n": top_n,
            },
        )
    return oi_df

def detect_volume_clusters(df, n=3):
    """
    Detect top N price levels by volume (volume clusters).
    """
    # Aggregate volume by close price rounded to nearest integer
    vol_by_price = df.groupby(df['close'].round()).agg({'volume': 'sum'}).reset_index()
    vol_by_price = vol_by_price.sort_values('volume', ascending=False).head(n)
    return vol_by_price

def generate_trade_signals(price_trend, oi_trend, current_price, support_zones, resistance_zones):
    """
    Generate simple entry/exit signals based on trend and price vs zones.
    """
    signals = []

    # Bullish scenario
    if price_trend == "uptrend" and oi_trend == "rising":
        # Entry near support zones
        for s in support_zones:
            if current_price <= s * 1.01:  # within 1% above support
                signals.append(f"Buy near support zone ₹{s:.2f}")
        # Exit near resistance zones
        for r in resistance_zones:
            if current_price >= r * 0.99:  # within 1% below resistance
                signals.append(f"Consider taking profits near resistance ₹{r:.2f}")

    # Bearish scenario
    elif price_trend == "downtrend" and oi_trend == "rising":
        # Entry near resistance zones
        for r in resistance_zones:
            if current_price >= r * 0.99:
                signals.append(f"Short near resistance zone ₹{r:.2f}")
        # Exit near support zones
        for s in support_zones:
            if current_price <= s * 1.01:
                signals.append(f"Cover shorts near support ₹{s:.2f}")

    # Short covering rally (price up, OI down)
    elif price_trend == "uptrend" and oi_trend == "falling":
        signals.append("Caution: Rally may be short covering. Wait for confirmation.")

    # Long unwinding (price down, OI down)
    elif price_trend == "downtrend" and oi_trend == "falling":
        signals.append("Caution: Downtrend may be losing strength. Watch for reversal.")

    else:
        signals.append("Mixed signals. Wait for clearer trend.")

    if not signals:
        signals.append("No clear trade signals at current price.")

    return signals

# -------------------------
# Main Analysis Trigger
# -------------------------
if st.button("Analyze"):

    # Fetch multi-day 3-min data for selected strike
    query = text(
        """
        SELECT trade_date, timestamp, open_interest, close, volume
        FROM option_3min_ohlc
        WHERE symbol = :symbol
          AND strike_price = :strike
          AND option_type = :otype
          AND expiry_date = :expiry
          AND trade_date BETWEEN :start AND :end
        ORDER BY trade_date, timestamp
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "symbol": symbol,
                "strike": strike_price,
                "otype": option_type,
                "expiry": expiry_date,
                "start": start_date,
                "end": end_date,
            },
        )

    if df.empty:
        st.warning("No data found for the selected filters.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # Aggregate daily data
        daily_data = df.groupby('trade_date').agg(
            daily_close=('close', 'last'),
            daily_oi=('open_interest', 'last'),
            daily_volume=('volume', 'sum')
        ).reset_index()

        # Calculate daily and cumulative changes
        daily_data['price_change'] = daily_data['daily_close'].diff()
        daily_data['price_change_pct'] = daily_data['daily_close'].pct_change() * 100
        daily_data['oi_change'] = daily_data['daily_oi'].diff()
        daily_data['oi_change_pct'] = daily_data['daily_oi'].pct_change() * 100
        daily_data['cum_price_change'] = daily_data['daily_close'] - daily_data['daily_close'].iloc[0]
        daily_data['cum_oi_change'] = daily_data['daily_oi'] - daily_data['daily_oi'].iloc[0]

        # Determine trends
        price_trend = "uptrend" if daily_data['cum_price_change'].iloc[-1] > 0 else "downtrend"
        oi_trend = "rising" if daily_data['cum_oi_change'].iloc[-1] > 0 else "falling"

        # Detect OI support/resistance zones (top 3 strikes by OI)
        oi_zones_df = detect_oi_support_resistance(symbol, option_type, expiry_date, start_date, end_date, top_n=3)
        oi_support_resistance = oi_zones_df['strike_price'].sort_values().values
        # For simplicity, treat lower strikes as support, higher as resistance
        support_zones = oi_support_resistance[:len(oi_support_resistance)//2 + 1]
        resistance_zones = oi_support_resistance[len(oi_support_resistance)//2 + 1:]

        # Detect volume clusters (top 3 price levels by volume)
        vol_clusters_df = detect_volume_clusters(df, n=3)
        vol_price_levels = vol_clusters_df['close'].values

        # Combine zones (OI strikes + volume clusters) for support/resistance
        combined_support = np.unique(np.concatenate([support_zones, vol_price_levels]))
        combined_resistance = np.unique(np.concatenate([resistance_zones, vol_price_levels]))

        # Current price (last close)
        current_price = daily_data['daily_close'].iloc[-1]

        # Generate trade signals
        signals = generate_trade_signals(price_trend, oi_trend, current_price, combined_support, combined_resistance)

        # Display daily summary
        st.subheader("📅 Daily Summary")
        st.dataframe(daily_data.style.format({
            'daily_close': '₹{:.2f}',
            'daily_oi': '{:,.0f}',
            'daily_volume': '{:,.0f}',
            'price_change': '₹{:.2f}',
            'price_change_pct': '{:.2f}%',
            'oi_change': '{:,.0f}',
            'oi_change_pct': '{:.2f}%',
            'cum_price_change': '₹{:.2f}',
            'cum_oi_change': '{:,.0f}'
        }))

        # Plot price and OI trends with support/resistance zones
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_data['trade_date'], y=daily_data['daily_close'],
            mode='lines+markers', name='Daily Close Price',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=daily_data['trade_date'], y=daily_data['daily_oi'],
            mode='lines+markers', name='Daily Open Interest',
            line=dict(color='orange'), yaxis='y2'
        ))

        # Plot support zones as horizontal lines
        for s in combined_support:
            fig.add_hline(y=s, line_dash="dot", line_color="green", annotation_text=f"Support ₹{s:.0f}", annotation_position="bottom left")

        # Plot resistance zones as horizontal lines
        for r in combined_resistance:
            fig.add_hline(y=r, line_dash="dot", line_color="red", annotation_text=f"Resistance ₹{r:.0f}", annotation_position="top left")

        fig.update_layout(
            title=f"Price and Open Interest Trend for {symbol} {strike_price} {option_type}",
            xaxis_title="Trade Date",
            yaxis=dict(
                title="Price (₹)",
                side='left'
            ),
            yaxis2=dict(
                title="Open Interest",
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.01, y=0.99),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display trend interpretation
        st.subheader("📋 Multi-Day Trend Interpretation")
        st.markdown(f"""
        - **Price Trend:** {price_trend} ({daily_data['cum_price_change'].iloc[-1]:.2f} ₹ change)
        - **Open Interest Trend:** {oi_trend} ({daily_data['cum_oi_change'].iloc[-1]:,.0f} contracts change)
        """)

        # Display detected support and resistance zones
        st.subheader("🛠️ Detected Support Zones")
        st.write([f"₹{s:.2f}" for s in combined_support])

        st.subheader("🛑 Detected Resistance Zones")
        st.write([f"₹{r:.2f}" for r in combined_resistance])

        # Display trade signals
        st.subheader("🎯 Trade Signals")
        for sig in signals:
            st.markdown(f"- {sig}")

import streamlit as st
import pandas as pd
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
st.title("📈 Multi-Day OI & Price Trend Analysis")

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
# Main Analysis Trigger
# -------------------------
if st.button("Analyze"):

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
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # Aggregate data per day: get daily close price and last OI
        daily_data = df.groupby('trade_date').agg(
            daily_close=('close', 'last'),
            daily_oi=('open_interest', 'last'),
            daily_volume=('volume', 'sum')
        ).reset_index()

        # Calculate daily changes
        daily_data['price_change'] = daily_data['daily_close'].diff()
        daily_data['price_change_pct'] = daily_data['daily_close'].pct_change() * 100
        daily_data['oi_change'] = daily_data['daily_oi'].diff()
        daily_data['oi_change_pct'] = daily_data['daily_oi'].pct_change() * 100

        # Calculate cumulative changes from start date
        daily_data['cum_price_change'] = daily_data['daily_close'] - daily_data['daily_close'].iloc[0]
        daily_data['cum_oi_change'] = daily_data['daily_oi'] - daily_data['daily_oi'].iloc[0]

        # Display daily summary table
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

        # Plot multi-day price and OI trends
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

        # Layout with secondary y-axis
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
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interpretation / Summary
        st.subheader("📋 Multi-Day Trend Interpretation")

        price_trend = "uptrend" if daily_data['cum_price_change'].iloc[-1] > 0 else "downtrend"
        oi_trend = "rising" if daily_data['cum_oi_change'].iloc[-1] > 0 else "falling"

        st.markdown(f"""
        - **Price Trend:** {price_trend} ({daily_data['cum_price_change'].iloc[-1]:.2f} ₹ change)
        - **Open Interest Trend:** {oi_trend} ({daily_data['cum_oi_change'].iloc[-1]:,.0f} contracts change)
        """)

        # Basic combined interpretation
        if price_trend == "uptrend" and oi_trend == "rising":
            st.success("Bullish trend confirmed: price and OI both rising, indicating fresh long positions.")
        elif price_trend == "downtrend" and oi_trend == "rising":
            st.warning("Bearish trend with rising OI: new short positions entering the market.")
        elif price_trend == "uptrend" and oi_trend == "falling":
            st.info("Price rising but OI falling: likely short covering, trend may be weaker.")
        elif price_trend == "downtrend" and oi_trend == "falling":
            st.info("Price falling and OI falling: long unwinding, trend may be losing strength.")
        else:
            st.info("Mixed signals, further analysis recommended.")

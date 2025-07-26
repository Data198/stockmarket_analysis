import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote
from datetime import datetime, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# DB Connection using st.secrets
# ------------------------------
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# ------------------------------
# Helper Functions
# ------------------------------

def calculate_vwap(df):
    # VWAP = sum(price * volume) / sum(volume)
    vwap = (df['close'] * df['volume']).sum() / df['volume'].sum() if df['volume'].sum() > 0 else np.nan
    return vwap

def detect_volume_spikes(df, multiplier=2):
    avg_vol = df['volume'].mean()
    spikes = df[df['volume'] > avg_vol * multiplier]
    return spikes

def get_session_blocks(df):
    # Define session blocks (customize as needed)
    blocks = [
        ("09:15", "10:00"),
        ("10:00", "12:00"),
        ("12:00", "13:00"),
        ("13:00", "15:30"),
    ]
    sessions = []
    for start, end in blocks:
        mask = (df['timestamp'].dt.time >= time.fromisoformat(start)) & (df['timestamp'].dt.time < time.fromisoformat(end))
        sessions.append((start + "-" + end, df[mask]))
    return sessions

def buildup_type(row):
    if pd.isna(row['oi_change']) or pd.isna(row['price_change']):
        return None
    if row['oi_change'] > 0 and row['price_change'] > 0:
        return "Long Buildup"
    elif row['oi_change'] > 0 and row['price_change'] < 0:
        return "Short Buildup"
    elif row['oi_change'] < 0 and row['price_change'] > 0:
        return "Short Covering"
    elif row['oi_change'] < 0 and row['price_change'] < 0:
        return "Long Unwinding"
    else:
        return "Neutral"

def get_buildup_stats(df):
    df['oi_change'] = df['open_interest'].diff()
    df['price_change'] = df['close'].diff()
    df['buildup'] = df.apply(buildup_type, axis=1)
    stats = df['buildup'].value_counts(normalize=True) * 100
    return stats.to_dict(), df

def get_volume_clusters(df, n=3):
    # Find top n bars by volume
    return df.nlargest(n, 'volume')

def get_psychological_levels(df):
    # Round numbers near high/low
    day_low = df['close'].min()
    day_high = df['close'].max()
    levels = {
        "Support": round(day_low, 2),
        "Resistance": round(day_high, 2),
        "Breakdown Level": round(np.floor(day_low / 5) * 5, 2),
        "Major Resistance": round(np.ceil(day_high / 5) * 5, 2)
    }
    return levels

def session_summary(session_name, session_df):
    if session_df.empty:
        return f"• {session_name}: No data"
    open_p = session_df['close'].iloc[0]
    close_p = session_df['close'].iloc[-1]
    move = close_p - open_p
    move_pct = (move / open_p) * 100 if open_p else 0
    oi_open = session_df['open_interest'].iloc[0]
    oi_close = session_df['open_interest'].iloc[-1]
    oi_move = oi_close - oi_open
    oi_move_pct = (oi_move / oi_open) * 100 if oi_open else 0
    buildup_stats, _ = get_buildup_stats(session_df)
    dom_buildup = max(buildup_stats, key=buildup_stats.get) if buildup_stats else "N/A"
    return (f"• {session_name}: {move:+.2f} ({move_pct:+.2f}%), "
            f"OI {oi_move:+,.0f} ({oi_move_pct:+.2f}%), "
            f"Dominant: {dom_buildup}")

def generate_narrative(df, metrics, buildup_stats, vwap, volume_spikes, volume_clusters, levels, session_blocks, iv_open, iv_close):
    lines = []
    # OVERALL SENTIMENT
    lines.append("📋 DETAILED MARKET SENTIMENT SUMMARY")
    lines.append("="*60)
    lines.append(f"\n{'🔴' if metrics['overall_sentiment'].startswith('Strongly Bearish') else '🟢' if metrics['overall_sentiment'].startswith('Strongly Bullish') else '🟡'} OVERALL SENTIMENT: {metrics['overall_sentiment'].upper()}")
    lines.append("-"*40)
    lines.append(f"• Price {'crashed' if metrics['price_change_pct'] < -10 else 'rose' if metrics['price_change_pct'] > 10 else 'moved'} {metrics['price_change_pct']:+.2f}% from ₹{metrics['opening_price']:.2f} to ₹{metrics['closing_price']:.2f}")
    lines.append(f"• OI {'rose' if metrics['oi_change'] > 0 else 'fell'} {metrics['oi_change_pct']:+.2f}% ({metrics['opening_oi']:,} → {metrics['closing_oi']:,})")
    lines.append(f"• VWAP: ₹{vwap:.2f} | Current vs VWAP: {metrics['closing_price']-vwap:+.2f}")
    lines.append(f"• IV: {iv_open:.2f}% → {iv_close:.2f}% ({iv_close-iv_open:+.2f}%)")
    lines.append(f"• Volume: {metrics['total_volume'] / 1e6:.1f}M | Avg per 3-min: {metrics['avg_volume'] / 1e3:.0f}K")
    lines.append("")

    # BUILDUP
    lines.append("🏗️ BUILDUP ANALYSIS:")
    lines.append("-"*20)
    for k, v in buildup_stats.items():
        lines.append(f"• {k}: {v:.1f}%")
    lines.append("")

    # SESSION TIMELINE
    lines.append("⏰ SESSION TIMELINE:")
    lines.append("-"*20)
    for name, s_df in session_blocks:
        lines.append(session_summary(name, s_df))
    lines.append("")

    # LEVELS
    lines.append("1️⃣ CRITICAL LEVELS TO MONITOR:")
    lines.append("-"*20)
    for k, v in levels.items():
        lines.append(f"• {k}: ₹{v}")
    for i, row in volume_clusters.iterrows():
        lines.append(f"• High Volume Bar: {row['timestamp'].strftime('%H:%M')} @ ₹{row['close']:.2f} (Vol: {row['volume']/1e6:.2f}M)")
    lines.append("")

    # VOLUME SPIKES
    lines.append("2️⃣ VOLUME SPIKES:")
    lines.append("-"*20)
    if not volume_spikes.empty:
        for i, row in volume_spikes.iterrows():
            lines.append(f"• {row['timestamp'].strftime('%H:%M')}: Vol {row['volume']/1e6:.2f}M, Price: ₹{row['close']:.2f}")
    else:
        lines.append("• No significant volume spikes detected.")
    lines.append("")

    # TIME DECAY
    lines.append("3️⃣ TIME DECAY FACTOR:")
    lines.append("-"*20)
    lines.append("• Check expiry proximity and theta risk.")
    lines.append("")

    # SENTIMENT INDICATORS
    lines.append("4️⃣ SENTIMENT INDICATORS:")
    lines.append("-"*20)
    lines.append(f"• {metrics['overall_sentiment']} sentiment")
    lines.append("")

    # TRADING STRATEGY
    lines.append("7️⃣ TRADING STRATEGY IMPLICATIONS:")
    lines.append("-"*20)
    if metrics['overall_sentiment'].startswith("Strongly Bearish") or metrics['overall_sentiment'] == "Bearish":
        lines.append("🔴 FOR BEARS:")
        lines.append("  • Look for rallies to resistance to add shorts")
        lines.append("  • Target: Support/Breakdown levels")
        lines.append("  • Stop Loss: Above major resistance")
        lines.append("")
        lines.append("🟡 FOR BULLS:")
        lines.append("  • Wait for clear reversal signals")
        lines.append("  • Need volume expansion above resistance")
        lines.append("  • Risk is HIGH due to time decay")
    elif metrics['overall_sentiment'].startswith("Strongly Bullish") or metrics['overall_sentiment'] == "Bullish":
        lines.append("🟢 FOR BULLS:")
        lines.append("  • Buy on dips to support")
        lines.append("  • Target: Resistance levels")
        lines.append("  • Stop Loss: Below support")
        lines.append("")
        lines.append("🔴 FOR BEARS:")
        lines.append("  • Wait for reversal or volume breakdown")
    else:
        lines.append("🟡 NEUTRAL:")
        lines.append("  • Wait for breakout or breakdown")
    lines.append("")
    lines.append("⚠️  RISK FACTORS:")
    lines.append("-"*20)
    lines.append("• Extreme time decay near expiry")
    lines.append("• High OI suggests strong positions")
    lines.append("• Any adverse news can accelerate move")
    lines.append("• Liquidity may reduce as expiry approaches")
    lines.append("")
    lines.append("🎯 FINAL RECOMMENDATION:")
    lines.append("-"*20)
    lines.append(f"{metrics['overall_sentiment'].upper()} BIAS with caution on bounces")
    return "\n".join(lines)

# ------------------------------
# OptionsAnalyzer Class
# ------------------------------
class OptionsAnalyzer:
    def __init__(self, engine):
        self.engine = engine

    def get_available_strikes(self, date_filter=None):
        query = """
        SELECT DISTINCT symbol, strike_price, expiry_date, option_type
        FROM option_3min_ohlc
        WHERE 1=1
        """
        if date_filter:
            query += f" AND trade_date = '{date_filter}'"
        query += " ORDER BY symbol, strike_price, expiry_date"
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            st.error(f"Error fetching strikes: {e}")
            return pd.DataFrame()

    def fetch_options_data(self, symbol, strike_price, expiry_date, option_type, date_filter=None):
        query = """
        SELECT timestamp, open_interest, close, volume, iv
        FROM option_3min_ohlc
        WHERE symbol = %(symbol)s AND strike_price = %(strike_price)s AND expiry_date = %(expiry_date)s AND option_type = %(option_type)s
        """
        params = {
            "symbol": symbol,
            "strike_price": strike_price,
            "expiry_date": expiry_date,
            "option_type": option_type
        }
        if date_filter:
            query += " AND trade_date = %(trade_date)s"
            params["trade_date"] = date_filter
        query += " ORDER BY timestamp"
        try:
            df = pd.read_sql(query, self.engine, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def calculate_sentiment_metrics(self, df):
        if df.empty or len(df) < 2:
            return {}
        opening_price = df['close'].iloc[0]
        closing_price = df['close'].iloc[-1]
        price_change = closing_price - opening_price
        price_change_pct = (price_change / opening_price) * 100 if opening_price else 0

        opening_oi = df['open_interest'].iloc[0]
        closing_oi = df['open_interest'].iloc[-1]
        oi_change = closing_oi - opening_oi
        oi_change_pct = (oi_change / opening_oi) * 100 if opening_oi else 0

        total_volume = df['volume'].sum()
        avg_volume = df['volume'].mean()
        max_volume = df['volume'].max()
        volume_spikes = len(df[df['volume'] > avg_volume * 2])

        opening_iv = df['iv'].iloc[0] if 'iv' in df.columns and not df['iv'].isna().all() else 0
        closing_iv = df['iv'].iloc[-1] if 'iv' in df.columns and not df['iv'].isna().all() else 0

        sentiment_score = 0
        sentiment_factors = []

        if price_change_pct > 5:
            sentiment_score += 2
            sentiment_factors.append("Strong price rally")
        elif price_change_pct > 0:
            sentiment_score += 1
            sentiment_factors.append("Price increase")
        elif price_change_pct < -5:
            sentiment_score -= 2
            sentiment_factors.append("Sharp price decline")
        elif price_change_pct < 0:
            sentiment_score -= 1
            sentiment_factors.append("Price decline")

        if oi_change_pct > 20:
            if price_change_pct > 0:
                sentiment_score += 1
                sentiment_factors.append("Strong long buildup")
            else:
                sentiment_score -= 1
                sentiment_factors.append("Strong short buildup")

        if total_volume > avg_volume * len(df) * 1.5:
            sentiment_factors.append("High volume activity")

        if sentiment_score >= 2:
            overall_sentiment = "Strongly Bullish"
            sentiment_color = "#2ca02c"
        elif sentiment_score == 1:
            overall_sentiment = "Bullish"
            sentiment_color = "#90EE90"
        elif sentiment_score == -1:
            overall_sentiment = "Bearish"
            sentiment_color = "#FFA07A"
        elif sentiment_score <= -2:
            overall_sentiment = "Strongly Bearish"
            sentiment_color = "#d62728"
        else:
            overall_sentiment = "Neutral"
            sentiment_color = "#ff7f0e"

        return {
            'opening_price': opening_price,
            'closing_price': closing_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'opening_oi': opening_oi,
            'closing_oi': closing_oi,
            'oi_change': oi_change,
            'oi_change_pct': oi_change_pct,
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'max_volume': max_volume,
            'volume_spikes': volume_spikes,
            'opening_iv': opening_iv,
            'closing_iv': closing_iv,
            'overall_sentiment': overall_sentiment,
            'sentiment_color': sentiment_color,
            'sentiment_factors': sentiment_factors,
            'sentiment_score': sentiment_score
        }

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.title("📈 Options Sentiment Analysis Dashboard")
    st.markdown("---")

    analyzer = OptionsAnalyzer(engine)

    st.sidebar.header("📅 Filters")
    date_filter = st.sidebar.date_input("Select Date", value=datetime.now().date())

    strikes_df = analyzer.get_available_strikes(date_filter)
    if strikes_df.empty:
        st.warning("No data available for the selected date.")
        return

    symbols = strikes_df['symbol'].unique()
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)
    symbol_strikes = strikes_df[strikes_df['symbol'] == selected_symbol]
    strike_options = [f"{row['strike_price']} {row['option_type']} (Exp: {row['expiry_date']})"
                      for _, row in symbol_strikes.iterrows()]
    selected_strike_option = st.sidebar.selectbox("Select Strike & Type", strike_options)

    if selected_strike_option:
        parts = selected_strike_option.split()
        strike_price = float(parts[0])
        option_type = parts[1]
        expiry_date = selected_strike_option.split("Exp: ")[1].rstrip(")")

        with st.spinner("Fetching and analyzing data..."):
            df = analyzer.fetch_options_data(selected_symbol, strike_price, expiry_date, option_type, date_filter)
            if df.empty:
                st.error("No data found for the selected option.")
                return
            metrics = analyzer.calculate_sentiment_metrics(df)
            buildup_stats, df = get_buildup_stats(df)
            vwap = calculate_vwap(df)
            volume_spikes = detect_volume_spikes(df)
            volume_clusters = get_volume_clusters(df)
            levels = get_psychological_levels(df)
            session_blocks = get_session_blocks(df)
            iv_open = metrics['opening_iv']
            iv_close = metrics['closing_iv']

        # Main Dashboard
        st.header(f"Analysis for {selected_symbol} {strike_price} {option_type}")
        st.markdown(f"**Expiry:** {expiry_date} | **Date:** {date_filter}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid {metrics['sentiment_color']};">
                <h3 style="color: {metrics['sentiment_color']};">Overall Sentiment</h3>
                <h2 style="color: {metrics['sentiment_color']};">{metrics['overall_sentiment']}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.metric("Price Change", f"₹{metrics['price_change']:.2f}", f"{metrics['price_change_pct']:.2f}%")
        with col3:
            st.metric("OI Change", f"{metrics['oi_change']:,.0f}", f"{metrics['oi_change_pct']:.2f}%")

        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Opening Price", f"₹{metrics['opening_price']:.2f}")
            st.metric("Closing Price", f"₹{metrics['closing_price']:.2f}")
        with col2:
            st.metric("Opening OI", f"{metrics['opening_oi']:,.0f}")
            st.metric("Closing OI", f"{metrics['closing_oi']:,.0f}")
        with col3:
            st.metric("Total Volume", f"{metrics['total_volume']:,.0f}")
            st.metric("Volume Spikes", metrics['volume_spikes'])
        with col4:
            st.metric("Opening IV", f"{metrics['opening_iv']:.2f}%")
            st.metric("IV Change", f"{iv_close - iv_open:+.2f}%")
            st.metric("VWAP", f"₹{vwap:.2f}")

        st.subheader("🏗️ Buildup Analysis")
        buildup_df = pd.DataFrame(list(buildup_stats.items()), columns=['Pattern', 'Percentage'])
        st.bar_chart(buildup_df.set_index('Pattern'))

        st.subheader("📈 Price, OI & Volume (with VWAP & Spikes)")
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & VWAP', 'Open Interest', 'Volume'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        # Price & VWAP
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Close Price', line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=[vwap]*len(df), name='VWAP', line=dict(color='purple', dash='dash')), row=1, col=1)
        # Volume spikes
        if not volume_spikes.empty:
            fig.add_trace(go.Scatter(
                x=volume_spikes['timestamp'], y=volume_spikes['close'],
                mode='markers', name='Volume Spike', marker=dict(color='red', size=10, symbol='star')
            ), row=1, col=1)
        # OI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['open_interest'], name='Open Interest', line=dict(color='orange', width=2)), row=2, col=1)
        # Volume
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='green'), row=3, col=1)
        # Volume clusters
        if not volume_clusters.empty:
            fig.add_trace(go.Bar(
                x=volume_clusters['timestamp'], y=volume_clusters['volume'],
                name='High Volume Bar', marker_color='red'
            ), row=3, col=1)
        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📝 Market Sentiment Summary")
        summary = generate_narrative(df, metrics, buildup_stats, vwap, volume_spikes, volume_clusters, levels, session_blocks, iv_open, iv_close)
        st.code(summary, language="markdown")

if __name__ == "__main__":
    main()
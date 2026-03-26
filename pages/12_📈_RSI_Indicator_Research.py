import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from urllib.parse import quote

st.set_page_config(page_title="RSI Indicator Research", page_icon="📈", layout="wide")

st.title("📈 RSI Indicator Research & Recommendations")
st.markdown("Deep dive into RSI variants — formulas, strengths, and best use cases for **intraday options trading on NSE**.")

# ──────────────────────────────────────────────
# DB Connection
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine():
    user = st.secrets["postgres"]["user"]
    password = quote(st.secrets["postgres"]["password"])
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"]["port"]
    db = st.secrets["postgres"]["database"]
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

engine = get_engine()

# ──────────────────────────────────────────────
# RSI Calculation Functions
# ──────────────────────────────────────────────

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder Smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_cutler(close: pd.Series, period: int = 14) -> pd.Series:
    """Cutler's RSI — uses Simple MA instead of Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
              smooth_k: int = 3, smooth_d: int = 3):
    """Stochastic RSI — applies Stochastic formula on RSI values."""
    r = rsi_wilder(close, rsi_period)
    r_min = r.rolling(stoch_period).min()
    r_max = r.rolling(stoch_period).max()
    k = 100 * (r - r_min) / (r_max - r_min).replace(0, np.nan)
    k_smooth = k.rolling(smooth_k).mean()
    d_smooth = k_smooth.rolling(smooth_d).mean()
    return k_smooth, d_smooth


def connors_rsi(close: pd.Series, rsi_short: int = 3, rsi_streak: int = 2,
                roc_period: int = 100):
    """
    Connors RSI — combines:
      1. RSI(3) of price
      2. RSI(2) of the up/down streak length
      3. Percentile rank of 1-day ROC over last `roc_period` bars
    """
    # Component 1: RSI(3) on close
    c1 = rsi_wilder(close, rsi_short)

    # Component 2: streak RSI
    streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            streak.iloc[i] = max(streak.iloc[i - 1], 0) + 1
        elif close.iloc[i] < close.iloc[i - 1]:
            streak.iloc[i] = min(streak.iloc[i - 1], 0) - 1
        else:
            streak.iloc[i] = 0
    c2 = rsi_wilder(streak, rsi_streak)

    # Component 3: percentile rank of 1-bar ROC
    roc = close.pct_change() * 100
    c3 = roc.rolling(roc_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    return (c1 + c2 + c3) / 3


def laguerre_rsi(close: pd.Series, gamma: float = 0.5) -> pd.Series:
    """
    Laguerre RSI — uses a 4-element Laguerre filter.
    gamma (0–1): higher = smoother but more lag. Typical: 0.5.
    """
    L0 = pd.Series(0.0, index=close.index)
    L1 = pd.Series(0.0, index=close.index)
    L2 = pd.Series(0.0, index=close.index)
    L3 = pd.Series(0.0, index=close.index)

    for i in range(1, len(close)):
        L0.iloc[i] = (1 - gamma) * close.iloc[i] + gamma * L0.iloc[i - 1]
        L1.iloc[i] = -gamma * L0.iloc[i] + L0.iloc[i - 1] + gamma * L1.iloc[i - 1]
        L2.iloc[i] = -gamma * L1.iloc[i] + L1.iloc[i - 1] + gamma * L2.iloc[i - 1]
        L3.iloc[i] = -gamma * L2.iloc[i] + L2.iloc[i - 1] + gamma * L3.iloc[i - 1]

    cu = pd.Series(0.0, index=close.index)
    cd = pd.Series(0.0, index=close.index)

    for i in range(len(close)):
        _cu, _cd = 0.0, 0.0
        if L0.iloc[i] >= L1.iloc[i]:
            _cu += L0.iloc[i] - L1.iloc[i]
        else:
            _cd += L1.iloc[i] - L0.iloc[i]
        if L1.iloc[i] >= L2.iloc[i]:
            _cu += L1.iloc[i] - L2.iloc[i]
        else:
            _cd += L2.iloc[i] - L1.iloc[i]
        if L2.iloc[i] >= L3.iloc[i]:
            _cu += L2.iloc[i] - L3.iloc[i]
        else:
            _cd += L3.iloc[i] - L2.iloc[i]
        cu.iloc[i] = _cu
        cd.iloc[i] = _cd

    denom = cu + cd
    return pd.Series(
        np.where(denom == 0, 0, 100 * cu / denom),
        index=close.index
    )


def volume_weighted_rsi(close: pd.Series, volume: pd.Series,
                        period: int = 14) -> pd.Series:
    """Volume-Weighted RSI — weights each period's gain/loss by its volume."""
    delta = close.diff()
    gain = delta.clip(lower=0) * volume
    loss = (-delta.clip(upper=0)) * volume
    avg_gain = gain.rolling(period).sum() / volume.rolling(period).sum()
    avg_loss = loss.rolling(period).sum() / volume.rolling(period).sum()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def oi_weighted_rsi(close: pd.Series, oi: pd.Series,
                    period: int = 14) -> pd.Series:
    """
    OI-Weighted RSI (custom) — weights each bar's price move by its
    Open Interest change magnitude. Useful for options markets where
    OI buildup amplifies momentum signals.
    """
    oi_change = oi.diff().abs().fillna(1).replace(0, 1)
    delta = close.diff()
    gain = delta.clip(lower=0) * oi_change
    loss = (-delta.clip(upper=0)) * oi_change
    avg_gain = gain.rolling(period).sum() / oi_change.rolling(period).sum()
    avg_loss = loss.rolling(period).sum() / oi_change.rolling(period).sum()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_divergence(close: pd.Series, rsi: pd.Series,
                   lookback: int = 5) -> pd.Series:
    """
    Detects RSI divergence:
      +1 = Bullish divergence (price lower low, RSI higher low)
      -1 = Bearish divergence (price higher high, RSI lower high)
       0 = No divergence
    """
    signal = pd.Series(0, index=close.index)
    for i in range(lookback, len(close)):
        price_window = close.iloc[i - lookback:i + 1]
        rsi_window = rsi.iloc[i - lookback:i + 1]
        # Bearish: price makes higher high but RSI doesn't
        if price_window.iloc[-1] == price_window.max() and rsi_window.iloc[-1] < rsi_window.max():
            signal.iloc[i] = -1
        # Bullish: price makes lower low but RSI doesn't
        elif price_window.iloc[-1] == price_window.min() and rsi_window.iloc[-1] > rsi_window.min():
            signal.iloc[i] = 1
    return signal


# ──────────────────────────────────────────────
# Research Summary Section
# ──────────────────────────────────────────────
st.markdown("---")
st.header("1. RSI Variants — Overview")

variants = {
    "Classic Wilder RSI": {
        "formula": "RS = Avg Gain / Avg Loss  (Wilder EMA)\nRSI = 100 - 100/(1+RS)",
        "best_period": "14 (swing) · 9 (intraday) · 5 (scalping)",
        "pros": "Industry standard. Works on all timeframes. Great for identifying overbought/oversold.",
        "cons": "Lags on fast moves. Gives many false signals in sideways markets.",
        "nse_use": "Use RSI(9) on 3-min charts for Nifty/BankNifty intraday momentum.",
        "signal": "Buy < 30 · Sell > 70 (trending); 40/60 in strong trends",
    },
    "Cutler's RSI": {
        "formula": "Same as Wilder but uses Simple MA (SMA) instead of EMA",
        "best_period": "14",
        "pros": "More consistent across different starting points. No seed-value dependency.",
        "cons": "Slightly less responsive than Wilder. Less widely supported in libraries.",
        "nse_use": "Good for end-of-day analysis. Less useful on 3-min charts.",
        "signal": "Buy < 30 · Sell > 70",
    },
    "Stochastic RSI": {
        "formula": "StochRSI = (RSI - RSI_min) / (RSI_max - RSI_min)\nK = SMA(StochRSI, 3)  ·  D = SMA(K, 3)",
        "best_period": "RSI(14), Stoch(14), K(3), D(3)",
        "pros": "More sensitive than RSI. Catches momentum reversals early. K/D crossover is clean.",
        "cons": "Very noisy. Many false signals. Requires trend filter.",
        "nse_use": "Best on 3-min charts for quick entry timing. Confirm with RSI(14) trend direction.",
        "signal": "Buy: K crosses D above from <20 · Sell: K crosses D below from >80",
    },
    "Connors RSI": {
        "formula": "CRSI = (RSI(3) + RSI_streak(2) + Percentile_ROC(100)) / 3",
        "best_period": "RSI(3), Streak RSI(2), Percentile(100)",
        "pros": "Multi-dimensional. Reduces false signals by combining 3 independent measures.",
        "cons": "Complex. Needs 100+ bars. Streak RSI adds extra computation.",
        "nse_use": "Excellent for mean-reversion plays on liquid options (ATM strikes). CRSI < 10 = extreme oversold.",
        "signal": "Buy < 10 (strong) / < 25 (moderate) · Sell > 90 (strong) / > 75 (moderate)",
    },
    "Laguerre RSI": {
        "formula": "4-element Laguerre filter on price → RSI of filtered values\ngamma controls smoothness (0.5 recommended)",
        "best_period": "No period — controlled by gamma (0.2 fast · 0.5 balanced · 0.8 slow)",
        "pros": "Very smooth with minimal lag vs traditional MA smoothing. Fewer whipsaws.",
        "cons": "Less intuitive to tune. gamma must be optimised per asset/timeframe.",
        "nse_use": "Best for BankNifty scalping on 1–3 min. gamma=0.4 works well for volatile sessions.",
        "signal": "Buy: crosses up from <20 · Sell: crosses down from >80",
    },
    "Volume-Weighted RSI": {
        "formula": "Gain/Loss weighted by bar volume before averaging",
        "best_period": "14",
        "pros": "High-volume moves get more weight — aligns RSI with real market conviction.",
        "cons": "Requires clean volume data. Less effective on options (volume ≠ OI).",
        "nse_use": "Use on the underlying (NIFTY/BANKNIFTY spot/futures) to confirm RSI signals from options.",
        "signal": "Same as classic RSI but signals are higher conviction",
    },
    "OI-Weighted RSI (Custom)": {
        "formula": "Gain/Loss weighted by |OI change| before averaging\n(Custom indicator designed for F&O markets)",
        "best_period": "14",
        "pros": "Unique to F&O: OI buildup/unwinding amplifies the RSI signal. High-OI moves treated as more significant.",
        "cons": "Requires OI data per bar. Must normalise OI changes. Novel — needs backtesting.",
        "nse_use": "PURPOSE-BUILT for this app. Apply on 3-min options OHLC with OI data. Strong OI buildup + oversold RSI = high-probability reversal.",
        "signal": "Buy < 35 with large OI buildup · Sell > 65 with large OI buildup",
    },
}

for name, info in variants.items():
    highlight = "🏆 " if "Custom" in name or "Connors" in name or "Stochastic" in name else ""
    with st.expander(f"{highlight}{name}", expanded=("Custom" in name)):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Formula:**\n```\n{info['formula']}\n```")
            st.markdown(f"**Best Period/Setting:** `{info['best_period']}`")
            st.markdown(f"**Signal Levels:** {info['signal']}")
        with col2:
            st.success(f"**Pros:** {info['pros']}")
            st.error(f"**Cons:** {info['cons']}")
            st.info(f"**NSE Intraday Use:** {info['nse_use']}")

# ──────────────────────────────────────────────
# Live Demo Section
# ──────────────────────────────────────────────
st.markdown("---")
st.header("2. Live RSI Demo on Your OHLC Data")

@st.cache_data(show_spinner=False, ttl=300)
def load_symbols():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT symbol FROM ohlc_equity_live ORDER BY symbol"
        )).fetchall()
    return [r[0] for r in rows]

@st.cache_data(show_spinner=False, ttl=300)
def load_ohlc(symbol: str, limit: int = 500):
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlc_equity_live
            WHERE symbol = :sym
            ORDER BY timestamp DESC
            LIMIT :lim
        """), conn, params={"sym": symbol, "lim": limit})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

try:
    symbols = load_symbols()
    if not symbols:
        st.warning("No data found in `ohlc_equity_live`. Upload OHLC data first.")
        st.stop()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_symbol = st.selectbox("Select Symbol", symbols)
    with col2:
        rsi_period = st.slider("RSI Period", 3, 30, 14)
    with col3:
        gamma = st.slider("Laguerre Gamma", 0.1, 0.9, 0.5, step=0.05)

    indicators = st.multiselect(
        "Select RSI Indicators to Plot",
        ["Wilder RSI", "Cutler RSI", "StochRSI %K", "StochRSI %D",
         "Connors RSI", "Laguerre RSI", "Volume-Weighted RSI"],
        default=["Wilder RSI", "StochRSI %K", "Laguerre RSI"]
    )

    df = load_ohlc(selected_symbol)

    if df.empty:
        st.warning(f"No OHLC data for {selected_symbol}.")
        st.stop()

    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    # Compute all indicators
    df["wilder_rsi"] = rsi_wilder(close, rsi_period)
    df["cutler_rsi"] = rsi_cutler(close, rsi_period)
    k, d = stoch_rsi(close)
    df["stoch_k"] = k
    df["stoch_d"] = d
    df["laguerre_rsi"] = laguerre_rsi(close, gamma)
    df["vw_rsi"] = volume_weighted_rsi(close, volume, rsi_period)

    if len(df) >= 110:
        df["connors_rsi"] = connors_rsi(close)
    else:
        df["connors_rsi"] = np.nan

    df["rsi_div"] = rsi_divergence(close, df["wilder_rsi"])

    # Plot
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.30, 0.25],
                        subplot_titles=[f"{selected_symbol} Price", "RSI Indicators", "StochRSI"],
                        vertical_spacing=0.06)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Divergence markers
    bull_div = df[df["rsi_div"] == 1]
    bear_div = df[df["rsi_div"] == -1]
    if not bull_div.empty:
        fig.add_trace(go.Scatter(
            x=bull_div["timestamp"], y=bull_div["low"] * 0.999,
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="Bullish Div"
        ), row=1, col=1)
    if not bear_div.empty:
        fig.add_trace(go.Scatter(
            x=bear_div["timestamp"], y=bear_div["high"] * 1.001,
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Bearish Div"
        ), row=1, col=1)

    # RSI panel
    color_map = {
        "Wilder RSI": ("#2196F3", "wilder_rsi"),
        "Cutler RSI": ("#FF9800", "cutler_rsi"),
        "Connors RSI": ("#9C27B0", "connors_rsi"),
        "Laguerre RSI": ("#00BCD4", "laguerre_rsi"),
        "Volume-Weighted RSI": ("#4CAF50", "vw_rsi"),
    }
    for label, (color, col_name) in color_map.items():
        if label in indicators and col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col_name],
                name=label, line=dict(color=color, width=1.5)
            ), row=2, col=1)

    # OB/OS lines
    for level, color, label in [(70, "red", "OB 70"), (30, "green", "OS 30"), (50, "gray", "Mid")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      annotation_text=label, row=2, col=1)

    # StochRSI panel
    if "StochRSI %K" in indicators:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["stoch_k"],
            name="StochRSI %K", line=dict(color="#E91E63", width=1.5)
        ), row=3, col=1)
    if "StochRSI %D" in indicators:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["stoch_d"],
            name="StochRSI %D", line=dict(color="#FF5722", width=1.5, dash="dot")
        ), row=3, col=1)
    for level, color in [(80, "red"), (20, "green")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, row=3, col=1)

    fig.update_layout(
        height=750,
        showlegend=True,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=-0.05)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    st.subheader("Current RSI Readings")
    last = df.iloc[-1]
    stats = {
        "Wilder RSI": round(last["wilder_rsi"], 2),
        "Cutler RSI": round(last["cutler_rsi"], 2),
        "StochRSI %K": round(last["stoch_k"], 2),
        "StochRSI %D": round(last["stoch_d"], 2),
        "Laguerre RSI": round(last["laguerre_rsi"], 2),
        "Vol-Weighted RSI": round(last["vw_rsi"], 2),
        "Connors RSI": round(last["connors_rsi"], 2) if not np.isnan(last["connors_rsi"]) else "Need 110+ bars",
    }

    def rsi_badge(val):
        if isinstance(val, str):
            return val
        if val >= 70:
            return f"🔴 {val} (Overbought)"
        elif val <= 30:
            return f"🟢 {val} (Oversold)"
        elif val >= 60:
            return f"🟠 {val} (Bullish)"
        elif val <= 40:
            return f"🔵 {val} (Bearish)"
        return f"⚪ {val} (Neutral)"

    stat_df = pd.DataFrame([{"Indicator": k, "Value": rsi_badge(v)} for k, v in stats.items()])
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not load OHLC data: {e}")
    st.info("Make sure your `ohlc_equity_live` table has data. The RSI formulas above are still available for review.")

# ──────────────────────────────────────────────
# Recommendations Section
# ──────────────────────────────────────────────
st.markdown("---")
st.header("3. Recommendations for NSE Intraday Options Trading")

st.markdown("""
### Overall Rankings (tailored to 3-min F&O charts)

| Rank | Indicator | Best For | Timeframe | Key Setting |
|------|-----------|----------|-----------|-------------|
| 🥇 1 | **OI-Weighted RSI** | Options momentum confirmation | 3-min | Period 14, OI from fact_oi_bhavcopy |
| 🥈 2 | **Connors RSI** | Mean-reversion entries on ATM strikes | 3–5 min | RSI(3), Streak(2), ROC(100) |
| 🥉 3 | **StochRSI** | Fast entry timing, crossover signals | 1–3 min | RSI(14), Stoch(14), K(3), D(3) |
| 4 | **Laguerre RSI** | Smooth trend following, fewer whipsaws | 3-min | gamma = 0.4–0.5 |
| 5 | **Wilder RSI(9)** | Core trend bias filter | 5–15 min | Period 9 for intraday |
| 6 | **Volume-Weighted RSI** | Underlying spot/futures confirmation | 5-min | Period 14 |
| 7 | **Wilder RSI(14)** | Swing trade levels, support/resistance | Daily | Period 14 |
| 8 | **Cutler RSI** | End-of-day analysis | Daily | Period 14 |

---

### Recommended Combination Strategy for This App

```
Step 1 — Trend Filter (5-min):
  Wilder RSI(9) > 50  →  Bullish bias  (only buy CE / sell PE)
  Wilder RSI(9) < 50  →  Bearish bias  (only buy PE / sell CE)

Step 2 — Entry Signal (3-min):
  StochRSI %K crosses %D upward from <20  →  Long entry
  StochRSI %K crosses %D downward from >80 →  Short entry

Step 3 — Conviction Filter (your OI data):
  OI-Weighted RSI aligns with signal direction  →  High conviction
  CE OI building + RSI oversold  →  Market expects resistance = short CE
  PE OI building + RSI overbought →  Market expects support = short PE

Step 4 — Divergence Check:
  Bullish divergence (price lower low, RSI higher low)  →  Reversal buy
  Bearish divergence (price higher high, RSI lower high) →  Reversal sell

Step 5 — Exit:
  Laguerre RSI crosses back through 50  →  Exit position
  OR Connors RSI > 90 (extreme overbought)  →  Book profits
```

---

### Key RSI Levels for NSE F&O

| Scenario | RSI Level | Action |
|----------|-----------|--------|
| Extreme Oversold (gap-down open) | < 20 | Look for reversal CE buy |
| Oversold | 20–35 | CE buying opportunity |
| Neutral-Bearish | 35–45 | Wait or light CE |
| Neutral | 45–55 | No trade / reduce size |
| Neutral-Bullish | 55–65 | PE buying or CE hold |
| Overbought | 65–80 | Partial profit / PE entry |
| Extreme Overbought | > 80 | Strong PE buy / CE exit |

---

### Notes on Connors RSI (Top Pick for Mean Reversion)

Connors RSI is particularly powerful for **NSE options** because:
- CRSI < 10 on Nifty 3-min → extreme selling → buy ATM CE
- CRSI > 90 on BankNifty 3-min → extreme buying → buy ATM PE
- The streak component penalises assets that have moved in one direction too long —
  ideal for detecting exhaustion moves before option premium decay accelerates

---

### OI-Weighted RSI — Purpose-Built for This App

Since this app already stores OI data (`fact_oi_bhavcopy`) and 3-min OHLC
(`ohlc_equity_live`), the OI-Weighted RSI is uniquely valuable:

- **High OI buildup** = institutional positioning → moves are more meaningful
- When large OI builds at a strike AND price RSI is oversold, the probability
  of a bounce is higher than price-only RSI suggests
- Implementation: join `ohlc_equity_live` with `fact_oi_bhavcopy` on date + symbol
  and feed `oi_change` as the weight into the RSI formula above
""")

st.markdown("---")
st.caption("Research compiled for NSE F&O intraday trading | Branch: claude/rsi-indicator-research-JcNUM")

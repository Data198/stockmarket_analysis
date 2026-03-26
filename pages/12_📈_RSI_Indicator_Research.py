import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="RSI Indicator Research", page_icon="📈", layout="wide")

st.title("📈 RSI Indicator Research — Live on Real Nifty Data")
st.markdown("All RSI variants applied on **real 1-min NSE F&O data** with OI context.")

# ──────────────────────────────────────────────
# Load Feather Data
# ──────────────────────────────────────────────
FEATHER_FILE = "2026-03-23-index-nfo-data.feather"

@st.cache_data(show_spinner="Loading market data...")
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, FEATHER_FILE)
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

df_all = load_data()

# ──────────────────────────────────────────────
# RSI Functions
# ──────────────────────────────────────────────

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("wilder_rsi")


def stoch_rsi(close: pd.Series, rsi_period=14, stoch_period=14, k=3, d=3):
    r = rsi_wilder(close, rsi_period)
    r_min = r.rolling(stoch_period).min()
    r_max = r.rolling(stoch_period).max()
    stoch = 100 * (r - r_min) / (r_max - r_min).replace(0, np.nan)
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line.rename("stoch_k"), d_line.rename("stoch_d")


def connors_rsi(close: pd.Series, rsi_short=3, rsi_streak=2, roc_period=100):
    c1 = rsi_wilder(close, rsi_short)
    streak = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            streak.iloc[i] = max(streak.iloc[i - 1], 0) + 1
        elif close.iloc[i] < close.iloc[i - 1]:
            streak.iloc[i] = min(streak.iloc[i - 1], 0) - 1
    c2 = rsi_wilder(streak, rsi_streak)
    roc = close.pct_change() * 100
    c3 = roc.rolling(roc_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    return ((c1 + c2 + c3) / 3).rename("connors_rsi")


def laguerre_rsi(close: pd.Series, gamma=0.5) -> pd.Series:
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
        pairs = [(L0, L1), (L1, L2), (L2, L3)]
        for a, b in pairs:
            if a.iloc[i] >= b.iloc[i]:
                _cu += a.iloc[i] - b.iloc[i]
            else:
                _cd += b.iloc[i] - a.iloc[i]
        cu.iloc[i] = _cu
        cd.iloc[i] = _cd
    denom = cu + cd
    return pd.Series(np.where(denom == 0, 0, 100 * cu / denom), index=close.index).rename("laguerre_rsi")


def oi_weighted_rsi(close: pd.Series, oi: pd.Series, period=14) -> pd.Series:
    oi_chg = oi.diff().abs().fillna(1).replace(0, 1)
    delta = close.diff()
    gain = delta.clip(lower=0) * oi_chg
    loss = (-delta.clip(upper=0)) * oi_chg
    w_sum = oi_chg.rolling(period).sum()
    avg_gain = gain.rolling(period).sum() / w_sum
    avg_loss = loss.rolling(period).sum() / w_sum
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("oi_rsi")


def detect_divergence(close: pd.Series, rsi: pd.Series, lookback=10) -> pd.Series:
    signal = pd.Series(0, index=close.index)
    for i in range(lookback, len(close)):
        pw = close.iloc[i - lookback:i + 1]
        rw = rsi.iloc[i - lookback:i + 1]
        if pw.iloc[-1] == pw.min() and rw.iloc[-1] > rw.min():
            signal.iloc[i] = 1   # bullish
        elif pw.iloc[-1] == pw.max() and rw.iloc[-1] < rw.max():
            signal.iloc[i] = -1  # bearish
    return signal


def failure_swing_bull(rsi: pd.Series) -> pd.Series:
    """Bullish failure swing: RSI dips <30, bounces, holds >30, breaks prior high."""
    signal = pd.Series(0, index=rsi.index)
    state = "watching"
    low_val = None
    bounce_high = None
    for i in range(1, len(rsi)):
        v = rsi.iloc[i]
        if state == "watching" and v < 30:
            state = "dipped"
            low_val = v
        elif state == "dipped" and v > 30:
            state = "bounced"
            bounce_high = v
        elif state == "bounced":
            if v > bounce_high:
                bounce_high = v
            if v < 30:
                state = "dipped"
                low_val = v
                bounce_high = None
            elif bounce_high is not None and v > bounce_high:
                signal.iloc[i] = 1
                state = "watching"
    return signal


def failure_swing_bear(rsi: pd.Series) -> pd.Series:
    """Bearish failure swing: RSI >70, pulls back, fails to regain 70, breaks prior low."""
    signal = pd.Series(0, index=rsi.index)
    state = "watching"
    bounce_low = None
    for i in range(1, len(rsi)):
        v = rsi.iloc[i]
        if state == "watching" and v > 70:
            state = "peaked"
        elif state == "peaked" and v < 70:
            state = "pulled_back"
            bounce_low = v
        elif state == "pulled_back":
            if v < bounce_low:
                bounce_low = v
            if v > 70:
                state = "peaked"
                bounce_low = None
            elif bounce_low is not None and v < bounce_low:
                signal.iloc[i] = -1
                state = "watching"
    return signal


# ──────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────
st.sidebar.header("Chart Controls")

names = sorted(df_all["name"].unique())
selected_name = st.sidebar.selectbox("Index", names, index=names.index("NIFTY") if "NIFTY" in names else 0)

inst_types = sorted(df_all[df_all["name"] == selected_name]["instrument_type"].unique())
selected_inst = st.sidebar.selectbox("Instrument Type", inst_types, index=inst_types.index("FUT") if "FUT" in inst_types else 0)

df_filtered = df_all[(df_all["name"] == selected_name) & (df_all["instrument_type"] == selected_inst)]

if selected_inst != "FUT":
    expiries = sorted(df_filtered["expiry"].unique())
    selected_expiry = st.sidebar.selectbox("Expiry", expiries)
    df_filtered = df_filtered[df_filtered["expiry"] == selected_expiry]

    # Sort strikes by volume descending so ATM appears first
    strike_vol = df_filtered.groupby("strike")["volume"].sum().sort_values(ascending=False)
    strikes = strike_vol.index.tolist()
    selected_strike = st.sidebar.selectbox(
        "Strike (sorted by volume)",
        strikes,
        format_func=lambda x: f"{int(x)} ({selected_inst})"
    )
    df_filtered = df_filtered[df_filtered["strike"] == selected_strike]
else:
    # For FUT, pick the most active contract
    expiries = sorted(df_filtered["expiry"].unique())
    selected_expiry = st.sidebar.selectbox("Expiry", expiries)
    df_filtered = df_filtered[df_filtered["expiry"] == selected_expiry]

df_chart = df_filtered.sort_values("date").reset_index(drop=True)

resample = st.sidebar.selectbox("Resample To", ["1-min (raw)", "3-min", "5-min", "15-min"], index=1)

if resample != "1-min (raw)":
    rule = {"3-min": "3min", "5-min": "5min", "15-min": "15min"}[resample]
    df_chart = (
        df_chart.resample(rule, on="date")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum", "oi": "last"})
        .dropna(subset=["close"])
        .reset_index()
    )

st.sidebar.markdown("---")
rsi_period = st.sidebar.slider("RSI Period", 3, 30, 14)
gamma = st.sidebar.slider("Laguerre γ (gamma)", 0.1, 0.9, 0.5, step=0.05)

indicators = st.sidebar.multiselect(
    "RSI Indicators",
    ["Wilder RSI", "StochRSI %K", "StochRSI %D", "Connors RSI",
     "Laguerre RSI", "OI-Weighted RSI"],
    default=["Wilder RSI", "StochRSI %K", "OI-Weighted RSI"]
)

show_divergence = st.sidebar.checkbox("Show Divergence Markers", value=True)
show_failure_swings = st.sidebar.checkbox("Show Failure Swings", value=True)
show_oi_panel = st.sidebar.checkbox("Show OI Panel", value=True)

# ──────────────────────────────────────────────
# Guard — need enough bars
# ──────────────────────────────────────────────
if len(df_chart) < 20:
    st.warning(f"Not enough bars ({len(df_chart)}) for RSI. Try a different strike or lower timeframe.")
    st.stop()

close = df_chart["close"]
oi = df_chart["oi"]
volume = df_chart["volume"]

# ── Compute indicators ──
df_chart["wilder_rsi"] = rsi_wilder(close, rsi_period)
k_line, d_line = stoch_rsi(close)
df_chart["stoch_k"] = k_line
df_chart["stoch_d"] = d_line
df_chart["laguerre_rsi"] = laguerre_rsi(close, gamma)
df_chart["oi_rsi"] = oi_weighted_rsi(close, oi, rsi_period)

if len(df_chart) >= 110:
    df_chart["connors_rsi"] = connors_rsi(close)
else:
    df_chart["connors_rsi"] = np.nan

df_chart["div"] = detect_divergence(close, df_chart["wilder_rsi"])
df_chart["fs_bull"] = failure_swing_bull(df_chart["wilder_rsi"])
df_chart["fs_bear"] = failure_swing_bear(df_chart["wilder_rsi"])

# ──────────────────────────────────────────────
# Build Chart
# ──────────────────────────────────────────────
n_rows = 3 + (1 if show_oi_panel else 0)
row_heights = [0.40, 0.25, 0.20]
if show_oi_panel:
    row_heights.append(0.15)

subplot_titles = ["Price + Signals", "RSI Indicators", "StochRSI"]
if show_oi_panel:
    subplot_titles.append("Open Interest")

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
    vertical_spacing=0.04
)

# ── Row 1: Candlestick ──
fig.add_trace(go.Candlestick(
    x=df_chart["date"],
    open=df_chart["open"], high=df_chart["high"],
    low=df_chart["low"], close=df_chart["close"],
    name="Price",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350"
), row=1, col=1)

# ── Divergence markers ──
if show_divergence:
    bull_pts = df_chart[df_chart["div"] == 1]
    bear_pts = df_chart[df_chart["div"] == -1]
    if not bull_pts.empty:
        fig.add_trace(go.Scatter(
            x=bull_pts["date"], y=bull_pts["low"] * 0.9985,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=12, color="lime"),
            text=["BD"] * len(bull_pts), textposition="bottom center",
            textfont=dict(size=9, color="lime"),
            name="Bullish Divergence", showlegend=True
        ), row=1, col=1)
    if not bear_pts.empty:
        fig.add_trace(go.Scatter(
            x=bear_pts["date"], y=bear_pts["high"] * 1.0015,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color="tomato"),
            text=["RD"] * len(bear_pts), textposition="top center",
            textfont=dict(size=9, color="tomato"),
            name="Bearish Divergence", showlegend=True
        ), row=1, col=1)

# ── Failure swing markers on price ──
if show_failure_swings:
    fs_bull_pts = df_chart[df_chart["fs_bull"] == 1]
    fs_bear_pts = df_chart[df_chart["fs_bear"] == -1]
    if not fs_bull_pts.empty:
        fig.add_trace(go.Scatter(
            x=fs_bull_pts["date"], y=fs_bull_pts["low"] * 0.998,
            mode="markers",
            marker=dict(symbol="star", size=14, color="#00E5FF"),
            name="Bullish Failure Swing"
        ), row=1, col=1)
    if not fs_bear_pts.empty:
        fig.add_trace(go.Scatter(
            x=fs_bear_pts["date"], y=fs_bear_pts["high"] * 1.002,
            mode="markers",
            marker=dict(symbol="star", size=14, color="#FF6D00"),
            name="Bearish Failure Swing"
        ), row=1, col=1)

# ── Row 2: RSI Indicators ──
color_map = {
    "Wilder RSI":      ("#2196F3", "wilder_rsi"),
    "Connors RSI":     ("#9C27B0", "connors_rsi"),
    "Laguerre RSI":    ("#00BCD4", "laguerre_rsi"),
    "OI-Weighted RSI": ("#FF9800", "oi_rsi"),
}
for label, (color, col_name) in color_map.items():
    if label in indicators:
        fig.add_trace(go.Scatter(
            x=df_chart["date"], y=df_chart[col_name],
            name=label, line=dict(color=color, width=1.8)
        ), row=2, col=1)

for level, color, lbl in [(70, "rgba(239,83,80,0.6)", "OB"), (50, "rgba(150,150,150,0.4)", "Mid"), (30, "rgba(38,166,154,0.6)", "OS")]:
    fig.add_hline(y=level, line_dash="dot", line_color=color,
                  annotation_text=lbl, annotation_font_size=10, row=2, col=1)

# ── Row 3: StochRSI ──
if "StochRSI %K" in indicators:
    fig.add_trace(go.Scatter(
        x=df_chart["date"], y=df_chart["stoch_k"],
        name="StochRSI %K", line=dict(color="#E91E63", width=1.8)
    ), row=3, col=1)
if "StochRSI %D" in indicators:
    fig.add_trace(go.Scatter(
        x=df_chart["date"], y=df_chart["stoch_d"],
        name="StochRSI %D", line=dict(color="#FF5722", width=1.5, dash="dot")
    ), row=3, col=1)

# K/D crossover fills
k = df_chart["stoch_k"].fillna(50)
d = df_chart["stoch_d"].fillna(50)
fig.add_trace(go.Scatter(
    x=df_chart["date"], y=k, fill=None, mode="lines",
    line=dict(color="rgba(233,30,99,0)"), showlegend=False
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df_chart["date"], y=d, fill="tonexty", mode="lines",
    line=dict(color="rgba(255,87,34,0)"),
    fillcolor="rgba(233,30,99,0.08)", showlegend=False
), row=3, col=1)

for level, color in [(80, "rgba(239,83,80,0.5)"), (20, "rgba(38,166,154,0.5)")]:
    fig.add_hline(y=level, line_dash="dot", line_color=color, row=3, col=1)

# ── Row 4: OI Panel ──
if show_oi_panel:
    oi_chg = df_chart["oi"].diff().fillna(0)
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in oi_chg]
    fig.add_trace(go.Bar(
        x=df_chart["date"], y=df_chart["oi"],
        name="Open Interest", marker_color=colors, opacity=0.7
    ), row=4, col=1)

fig.update_layout(
    height=820,
    template="plotly_dark",
    showlegend=True,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=-0.04, font=dict(size=11)),
    margin=dict(l=40, r=40, t=60, b=40),
    title=dict(
        text=f"{selected_name} · {selected_inst} · {resample} · RSI({rsi_period})",
        font=dict(size=15)
    )
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="StochRSI", row=3, col=1, range=[0, 100])
if show_oi_panel:
    fig.update_yaxes(title_text="OI", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Current Readings Dashboard
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Current Readings")

last = df_chart.iloc[-1]

def badge(val, name):
    if pd.isna(val):
        return "⚫ N/A (need more bars)"
    val = round(val, 1)
    if val >= 80:
        return f"🔴 **{val}** — Extreme Overbought"
    elif val >= 70:
        return f"🟠 **{val}** — Overbought"
    elif val >= 55:
        return f"🟡 **{val}** — Bullish"
    elif val >= 45:
        return f"⚪ **{val}** — Neutral"
    elif val >= 30:
        return f"🔵 **{val}** — Bearish"
    elif val >= 20:
        return f"🟢 **{val}** — Oversold"
    else:
        return f"💚 **{val}** — Extreme Oversold"

cols = st.columns(3)
readings = [
    ("Wilder RSI", last["wilder_rsi"]),
    ("StochRSI %K", last["stoch_k"]),
    ("StochRSI %D", last["stoch_d"]),
    ("Laguerre RSI", last["laguerre_rsi"]),
    ("OI-Weighted RSI", last["oi_rsi"]),
    ("Connors RSI", last["connors_rsi"]),
]
for i, (name, val) in enumerate(readings):
    cols[i % 3].markdown(f"**{name}**  \n{badge(val, name)}")

# ──────────────────────────────────────────────
# Signal Interpretation
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Signal Interpretation")

wrsi = last["wilder_rsi"]
sk = last["stoch_k"]
sd = last["stoch_d"]
oi_rsi_val = last["oi_rsi"]

signals = []

# Trend bias
if not pd.isna(wrsi):
    if wrsi > 55:
        signals.append(("Trend Bias", "Bullish", "Wilder RSI above 55 — bulls in control", "green"))
    elif wrsi < 45:
        signals.append(("Trend Bias", "Bearish", "Wilder RSI below 45 — bears in control", "red"))
    else:
        signals.append(("Trend Bias", "Neutral", "Wilder RSI between 45–55 — no clear bias", "gray"))

# Entry signal from StochRSI
if not (pd.isna(sk) or pd.isna(sd)):
    if sk > sd and sk < 20:
        signals.append(("Entry Signal", "Strong Buy", "StochRSI %K crossed above %D from oversold zone", "green"))
    elif sk < sd and sk > 80:
        signals.append(("Entry Signal", "Strong Sell", "StochRSI %K crossed below %D from overbought zone", "red"))
    elif sk > sd:
        signals.append(("Entry Signal", "Mild Buy", "StochRSI %K above %D", "lightgreen"))
    else:
        signals.append(("Entry Signal", "Mild Sell", "StochRSI %K below %D", "salmon"))

# OI conviction
if not pd.isna(oi_rsi_val):
    if oi_rsi_val < 30:
        signals.append(("OI Conviction", "High — Oversold", "Large OI change with selling — potential reversal", "green"))
    elif oi_rsi_val > 70:
        signals.append(("OI Conviction", "High — Overbought", "Large OI change with buying — potential reversal", "red"))
    else:
        signals.append(("OI Conviction", "Moderate", "OI-weighted RSI in neutral zone", "gray"))

# Divergence
recent_div = df_chart["div"].tail(10)
if (recent_div == 1).any():
    signals.append(("Divergence", "Bullish Divergence Detected", "Price lower low, RSI higher low — reversal warning", "green"))
elif (recent_div == -1).any():
    signals.append(("Divergence", "Bearish Divergence Detected", "Price higher high, RSI lower high — reversal warning", "red"))
else:
    signals.append(("Divergence", "None", "No divergence in last 10 bars", "gray"))

# Failure swings
recent_fs_bull = df_chart["fs_bull"].tail(10)
recent_fs_bear = df_chart["fs_bear"].tail(10)
if (recent_fs_bull == 1).any():
    signals.append(("Failure Swing", "Bullish Failure Swing", "RSI dipped below 30, bounced, held, then broke prior high", "cyan"))
elif (recent_fs_bear == -1).any():
    signals.append(("Failure Swing", "Bearish Failure Swing", "RSI peaked above 70, pulled back, failed to recover, broke low", "orange"))

sig_df = pd.DataFrame(signals, columns=["Signal Type", "Reading", "Explanation", "Color"])
st.dataframe(sig_df[["Signal Type", "Reading", "Explanation"]], use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# RSI Education Quick Reference
# ──────────────────────────────────────────────
with st.expander("📖 Quick Reference — RSI Concepts"):
    st.markdown("""
| Concept | What to look for |
|---------|-----------------|
| **Wilder RSI > 50** | Bullish momentum — use as trend filter |
| **Wilder RSI < 50** | Bearish momentum |
| **StochRSI K crosses D < 20** | Buy signal (oversold crossover) |
| **StochRSI K crosses D > 80** | Sell signal (overbought crossover) |
| **Bullish Divergence** | Price lower low, RSI higher low → reversal up |
| **Bearish Divergence** | Price higher high, RSI lower high → reversal down |
| **Hidden Bullish Div** | Price higher low, RSI lower low → trend continuation up |
| **Bullish Failure Swing** | RSI: dip <30 → bounce → hold >30 → break prior high |
| **Bearish Failure Swing** | RSI: spike >70 → pullback → fail to recover → break prior low |
| **OI-Weighted RSI** | Same as Wilder but high-OI bars get more weight — stronger signals |
| **Connors RSI < 10** | Extreme oversold (mean reversion buy) |
| **Connors RSI > 90** | Extreme overbought (mean reversion sell) |
| **Laguerre RSI γ=0.5** | Smooth, low-lag RSI — good for exits |
    """)

st.caption("Data: 2026-03-23 · NSE F&O · 1-min raw data resampled | Branch: claude/rsi-indicator-research-JcNUM")

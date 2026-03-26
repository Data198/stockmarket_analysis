import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import os

st.set_page_config(page_title="RSI Backtest", page_icon="🔬", layout="wide")

st.title("🔬 RSI Backtest — NIFTY Options")
st.markdown(
    "Backtesting **two RSI strategies** on real 1-min NIFTY F&O data · **23 March 2026** · "
    "Signals on NIFTY FUT → Trades on ATM CE/PE options."
)

# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────
NIFTY_LOT   = 75
FUT_EXPIRY  = date(2026, 3, 30)
NEAR_EXP    = date(2026, 3, 24)   # near-weekly (intraday)
WEEKLY_EXP  = date(2026, 3, 30)   # weekly expiry (positional)
EOD_TIME    = pd.Timestamp("2026-03-23 15:15:00")
FEATHER     = "2026-03-23-index-nfo-data.feather"

# ─────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market data…")
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_feather(os.path.join(base, FEATHER))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

df_all = load_data()

# ─────────────────────────────────────────────────────
# INDICATOR HELPERS
# ─────────────────────────────────────────────────────
def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def stoch_rsi(close: pd.Series, rsi_p=14, stoch_p=14, k_s=3, d_s=3):
    r     = rsi_wilder(close, rsi_p)
    lo    = r.rolling(stoch_p).min()
    hi    = r.rolling(stoch_p).max()
    stoch = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    k     = stoch.rolling(k_s).mean()
    d     = k.rolling(d_s).mean()
    return k, d

def oi_weighted_rsi(close: pd.Series, oi: pd.Series, period: int) -> pd.Series:
    w      = oi.diff().abs().fillna(1).replace(0, 1)
    delta  = close.diff()
    g      = delta.clip(lower=0) * w
    l      = (-delta.clip(upper=0)) * w
    ws     = w.rolling(period).sum()
    rs     = (g.rolling(period).sum() / ws) / (l.rolling(period).sum() / ws).replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule, on="date")
          .agg({"open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum", "oi": "last"})
          .dropna(subset=["close"])
          .reset_index()
    )

# ─────────────────────────────────────────────────────
# OPTION PRICE LOOKUP (vectorised with searchsorted)
# ─────────────────────────────────────────────────────
def build_opt_index(opt_df: pd.DataFrame, tf_rule: str) -> dict:
    """
    Pre-build a dict: (strike, inst_type) -> resampled DataFrame
    for fast price lookups during backtest.
    """
    idx = {}
    for (strike, inst), grp in opt_df.groupby(["strike", "instrument_type"]):
        r = resample(grp.copy(), tf_rule)
        idx[(strike, inst)] = r
    return idx

def opt_price_at(opt_idx: dict, ts: pd.Timestamp,
                 strike: float, inst_type: str) -> float | None:
    grp = opt_idx.get((strike, inst_type))
    if grp is None or grp.empty:
        return None
    pos = grp["date"].searchsorted(ts)
    if pos >= len(grp):
        pos = len(grp) - 1
    row = grp.iloc[pos]
    if abs((row["date"] - ts).total_seconds()) > 600:
        return None
    return float(row["close"])

# ─────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────
def run_backtest(fut_raw: pd.DataFrame,
                 opt_idx: dict,
                 tf_rule: str,
                 rsi_period: int,
                 long_rsi_thresh: float,
                 short_rsi_thresh: float,
                 use_oi_filter: bool,
                 sl_pct: float,
                 tgt_pct: float,
                 txn_cost: float,
                 lot_size: int,
                 strategy_name: str):

    fut = resample(fut_raw, tf_rule)
    close = fut["close"]
    oi    = fut["oi"]

    fut["rsi"]     = rsi_wilder(close, rsi_period)
    fut["oi_rsi"]  = oi_weighted_rsi(close, oi, rsi_period)
    k, d           = stoch_rsi(close, rsi_p=rsi_period)
    fut["stoch_k"] = k
    fut["stoch_d"] = d

    trades   = []
    position = None

    for i in range(3, len(fut)):
        row  = fut.iloc[i]
        prev = fut.iloc[i - 1]

        ts       = row["date"]
        cur_rsi  = row["rsi"]
        prev_rsi = prev["rsi"]
        cur_k    = row["stoch_k"]
        prev_k   = prev["stoch_k"]
        cur_d    = row["stoch_d"]
        prev_d   = prev["stoch_d"]
        cur_oi_rsi = row["oi_rsi"]

        if pd.isna(cur_rsi) or pd.isna(cur_k):
            continue

        # ── EOD force-close ──
        if ts >= EOD_TIME:
            if position:
                ep = opt_price_at(opt_idx, ts, position["strike"], position["inst_type"])
                ep = ep or position["entry_price"]
                pnl_pts = ep - position["entry_price"]
                trades.append({**position,
                                "exit_time": ts,
                                "exit_price": round(ep, 2),
                                "exit_reason": "EOD",
                                "pnl_pts": round(pnl_pts, 2),
                                "pnl_rs":  round(pnl_pts * lot_size - txn_cost, 2),
                                "pnl_pct": round(pnl_pts / position["entry_price"] * 100, 2)})
            break

        # ── ENTRY ──
        if position is None:
            # Long: RSI cross above threshold OR StochRSI K cross D from oversold
            long_rsi_cross  = cur_rsi >  long_rsi_thresh and prev_rsi <= long_rsi_thresh
            long_stoch_xo   = (cur_rsi > 50 and
                                cur_k > cur_d and prev_k <= prev_d and cur_k < 40)
            long_oi_ok      = (not use_oi_filter) or (cur_oi_rsi < 50)

            # Short: RSI cross below threshold OR StochRSI K cross D from overbought
            short_rsi_cross = cur_rsi <  short_rsi_thresh and prev_rsi >= short_rsi_thresh
            short_stoch_xo  = (cur_rsi < 50 and
                                cur_k < cur_d and prev_k >= prev_d and cur_k > 60)
            short_oi_ok     = (not use_oi_filter) or (cur_oi_rsi > 50)

            long_signal  = (long_rsi_cross  or long_stoch_xo)  and long_oi_ok
            short_signal = (short_rsi_cross or short_stoch_xo) and short_oi_ok

            if long_signal or short_signal:
                direction = "LONG" if long_signal else "SHORT"
                inst_type = "CE"   if long_signal else "PE"
                atm       = round(row["close"] / 50) * 50

                ep = opt_price_at(opt_idx, ts, atm, inst_type)
                if ep and ep > 5:
                    position = {
                        "strategy":    strategy_name,
                        "direction":   direction,
                        "entry_time":  ts,
                        "entry_fut":   round(row["close"], 1),
                        "entry_rsi":   round(cur_rsi, 1),
                        "strike":      atm,
                        "inst_type":   inst_type,
                        "entry_price": round(ep, 2),
                        "sl_price":    round(ep * (1 - sl_pct / 100), 2),
                        "tgt_price":   round(ep * (1 + tgt_pct / 100), 2),
                    }

        # ── EXIT ──
        else:
            ep = opt_price_at(opt_idx, ts, position["strike"], position["inst_type"])
            if ep is None:
                continue

            exit_reason = None
            exit_price  = ep

            if ep <= position["sl_price"]:
                exit_reason = "Stop Loss"
                exit_price  = position["sl_price"]
            elif ep >= position["tgt_price"]:
                exit_reason = "Target Hit"
                exit_price  = position["tgt_price"]
            elif position["direction"] == "LONG" and cur_rsi < 50 and prev_rsi >= 50:
                exit_reason = "RSI Cross 50 ↓"
            elif position["direction"] == "SHORT" and cur_rsi > 50 and prev_rsi <= 50:
                exit_reason = "RSI Cross 50 ↑"

            if exit_reason:
                pnl_pts = exit_price - position["entry_price"]
                trades.append({**position,
                                "exit_time":  ts,
                                "exit_price": round(exit_price, 2),
                                "exit_reason": exit_reason,
                                "pnl_pts":    round(pnl_pts, 2),
                                "pnl_rs":     round(pnl_pts * lot_size - txn_cost, 2),
                                "pnl_pct":    round(pnl_pts / position["entry_price"] * 100, 2)})
                position = None

    return pd.DataFrame(trades), fut

# ─────────────────────────────────────────────────────
# SIDEBAR PARAMETERS
# ─────────────────────────────────────────────────────
st.sidebar.header("Parameters")
lot_size = st.sidebar.number_input("Lot Size (NIFTY)", value=75, min_value=1)
txn_cost = st.sidebar.number_input("Transaction Cost ₹/trade", value=30)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy A — Intraday (3-min)")
st.sidebar.markdown("RSI(9) · Near-expiry (24-Mar) · ATM CE/PE")
a_sl  = st.sidebar.slider("Stop Loss %", 10, 50, 25, key="a_sl")
a_tgt = st.sidebar.slider("Target %",    20, 100, 50, key="a_tgt")
a_oi  = st.sidebar.checkbox("OI-RSI Filter", value=True, key="a_oi")

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy B — Positional (15-min)")
st.sidebar.markdown("RSI(14) · Weekly expiry (30-Mar) · ATM CE/PE")
b_sl  = st.sidebar.slider("Stop Loss %", 15, 60, 35, key="b_sl")
b_tgt = st.sidebar.slider("Target %",    30, 150, 80, key="b_tgt")
b_oi  = st.sidebar.checkbox("OI-RSI Filter", value=True, key="b_oi")

# ─────────────────────────────────────────────────────
# PREPARE DATA
# ─────────────────────────────────────────────────────
fut_raw = (df_all[(df_all["name"] == "NIFTY") &
                  (df_all["instrument_type"] == "FUT") &
                  (df_all["expiry"] == FUT_EXPIRY)]
           .sort_values("date").reset_index(drop=True))

opts_near   = df_all[(df_all["name"] == "NIFTY") &
                     (df_all["instrument_type"].isin(["CE", "PE"])) &
                     (df_all["expiry"] == NEAR_EXP)].copy()

opts_weekly = df_all[(df_all["name"] == "NIFTY") &
                     (df_all["instrument_type"].isin(["CE", "PE"])) &
                     (df_all["expiry"] == WEEKLY_EXP)].copy()

# ─────────────────────────────────────────────────────
# RUN BACKTESTS
# ─────────────────────────────────────────────────────
with st.spinner("Building option price index…"):
    oi_near   = build_opt_index(opts_near,   "3min")
    oi_weekly = build_opt_index(opts_weekly, "15min")

with st.spinner("Running Strategy A (Intraday 3-min)…"):
    trades_a, fut_3m = run_backtest(
        fut_raw, oi_near, "3min",
        rsi_period=9,
        long_rsi_thresh=50, short_rsi_thresh=50,
        use_oi_filter=a_oi,
        sl_pct=a_sl, tgt_pct=a_tgt,
        txn_cost=txn_cost, lot_size=lot_size,
        strategy_name="Strategy A · Intraday 3-min"
    )

with st.spinner("Running Strategy B (Positional 15-min)…"):
    trades_b, fut_15m = run_backtest(
        fut_raw, oi_weekly, "15min",
        rsi_period=14,
        long_rsi_thresh=32, short_rsi_thresh=68,
        use_oi_filter=b_oi,
        sl_pct=b_sl, tgt_pct=b_tgt,
        txn_cost=txn_cost, lot_size=lot_size,
        strategy_name="Strategy B · Positional 15-min"
    )

# ─────────────────────────────────────────────────────
# SUMMARY HELPER
# ─────────────────────────────────────────────────────
def summarise(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    wins      = trades[trades["pnl_rs"] > 0]
    losses    = trades[trades["pnl_rs"] <= 0]
    win_rate  = len(wins) / len(trades) * 100
    avg_win   = wins["pnl_rs"].mean()   if not wins.empty   else 0
    avg_loss  = losses["pnl_rs"].mean() if not losses.empty else 0
    pf        = (wins["pnl_rs"].sum() / abs(losses["pnl_rs"].sum())
                 if not losses.empty and losses["pnl_rs"].sum() != 0 else np.inf)
    max_dd    = (trades["pnl_rs"].cumsum() - trades["pnl_rs"].cumsum().cummax()).min()
    return {
        "trades":   len(trades),
        "wins":     len(wins),
        "losses":   len(losses),
        "win_rate": round(win_rate, 1),
        "net_pnl":  round(trades["pnl_rs"].sum(), 2),
        "avg_win":  round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "pf":       round(pf, 2) if pf != np.inf else "∞",
        "max_dd":   round(max_dd, 2),
    }

sa = summarise(trades_a)
sb = summarise(trades_b)

# ─────────────────────────────────────────────────────
# DAY HEADER METRICS
# ─────────────────────────────────────────────────────
day_chg = fut_raw.iloc[-1]["close"] - fut_raw.iloc[0]["close"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Date",         "23 Mar 2026 (Mon)")
c2.metric("FUT Open",     f"₹{fut_raw.iloc[0]['close']:,.1f}")
c3.metric("FUT Close",    f"₹{fut_raw.iloc[-1]['close']:,.1f}")
c4.metric("Day Change",   f"{day_chg:+.1f} pts",
          delta=f"{day_chg/fut_raw.iloc[0]['close']*100:+.2f}%",
          delta_color="normal")
c5.metric("Day Bias",     "📉 Bearish" if day_chg < 0 else "📈 Bullish")

st.markdown("---")

# ─────────────────────────────────────────────────────
# SUMMARY SCORECARD
# ─────────────────────────────────────────────────────
st.subheader("Strategy Scorecard")
s1, s2 = st.columns(2)

def scorecard(col, name, s, color):
    with col:
        pnl_color = "green" if s.get("net_pnl", 0) >= 0 else "red"
        col.markdown(f"""
<div style="background:{color};padding:18px;border-radius:12px;color:white">
  <h4 style="margin:0 0 12px 0">{name}</h4>
  <table style="width:100%;font-size:14px">
    <tr><td>Total Trades</td><td align="right"><b>{s.get('trades',0)}</b></td></tr>
    <tr><td>Wins / Losses</td><td align="right"><b>{s.get('wins',0)} / {s.get('losses',0)}</b></td></tr>
    <tr><td>Win Rate</td><td align="right"><b>{s.get('win_rate','N/A')}%</b></td></tr>
    <tr><td>Net P&L (₹)</td><td align="right"><b>₹{s.get('net_pnl','N/A')}</b></td></tr>
    <tr><td>Avg Win</td><td align="right"><b>₹{s.get('avg_win','N/A')}</b></td></tr>
    <tr><td>Avg Loss</td><td align="right"><b>₹{s.get('avg_loss','N/A')}</b></td></tr>
    <tr><td>Profit Factor</td><td align="right"><b>{s.get('pf','N/A')}</b></td></tr>
    <tr><td>Max Drawdown</td><td align="right"><b>₹{s.get('max_dd','N/A')}</b></td></tr>
  </table>
</div>""", unsafe_allow_html=True)

scorecard(s1, "Strategy A · Intraday 3-min (RSI-9)", sa, "#1565C0")
scorecard(s2, "Strategy B · Positional 15-min (RSI-14)", sb, "#2E7D32")

st.markdown("---")

# ─────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────
def build_chart(fut_df: pd.DataFrame, trades: pd.DataFrame,
                rsi_col: str, tf_label: str, strategy_label: str):

    if trades.empty:
        st.warning(f"No trades generated for {strategy_label}.")
        return

    # Equity curve
    trades["cum_pnl"] = trades["pnl_rs"].cumsum()

    rows = 4
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.20, 0.22, 0.20],
        subplot_titles=[
            f"NIFTY FUT · {tf_label} — Entry/Exit Signals",
            f"RSI({rsi_col})",
            "Option P&L per Trade (₹)",
            "Cumulative Equity Curve (₹)"
        ],
        vertical_spacing=0.06
    )

    # ── Price candlestick ──
    fig.add_trace(go.Candlestick(
        x=fut_df["date"], open=fut_df["open"], high=fut_df["high"],
        low=fut_df["low"],  close=fut_df["close"], name="NIFTY FUT",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=True
    ), row=1, col=1)

    # ── Entry / Exit markers on price ──
    for _, tr in trades.iterrows():
        marker_color  = "#00E676" if tr["direction"] == "LONG" else "#FF1744"
        exit_color    = "#76FF03" if tr["pnl_rs"] > 0 else "#FF6D00"
        entry_symbol  = "triangle-up"   if tr["direction"] == "LONG" else "triangle-down"

        fig.add_trace(go.Scatter(
            x=[tr["entry_time"]], y=[tr["entry_fut"]],
            mode="markers+text",
            marker=dict(symbol=entry_symbol, size=14, color=marker_color,
                        line=dict(color="white", width=1)),
            text=[f"BUY {tr['inst_type']}<br>{int(tr['strike'])}"],
            textposition="top center" if tr["direction"]=="LONG" else "bottom center",
            textfont=dict(size=9, color=marker_color),
            name=f"{tr['direction']} {tr['inst_type']} entry", showlegend=False
        ), row=1, col=1)

        # exit line
        fig.add_shape(type="line",
            x0=tr["entry_time"], x1=tr["exit_time"],
            y0=tr["entry_fut"],  y1=tr["entry_fut"],
            line=dict(color=marker_color, dash="dot", width=1),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[tr["exit_time"]], y=[tr["entry_fut"]],
            mode="markers+text",
            marker=dict(symbol="x", size=12, color=exit_color,
                        line=dict(color="white", width=1)),
            text=[tr["exit_reason"]],
            textposition="top right",
            textfont=dict(size=8, color=exit_color),
            name="exit", showlegend=False
        ), row=1, col=1)

    # ── RSI ──
    fig.add_trace(go.Scatter(
        x=fut_df["date"], y=fut_df["rsi"],
        name=f"RSI({rsi_col})", line=dict(color="#2196F3", width=1.8)
    ), row=2, col=1)

    for level, color, lbl in [(70, "rgba(239,83,80,0.7)", "70"),
                               (50, "rgba(150,150,150,0.5)", "50"),
                               (30, "rgba(38,166,154,0.7)", "30")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      annotation_text=lbl, annotation_font_size=9, row=2, col=1)

    # mark entry RSI dots
    if not trades.empty:
        for _, tr in trades.iterrows():
            rsi_row = fut_df[fut_df["date"] <= tr["entry_time"]]
            if not rsi_row.empty:
                rsi_val = rsi_row.iloc[-1]["rsi"]
                fig.add_trace(go.Scatter(
                    x=[tr["entry_time"]], y=[rsi_val],
                    mode="markers",
                    marker=dict(size=10, color="#00E676" if tr["direction"]=="LONG" else "#FF1744",
                                symbol="circle"),
                    showlegend=False
                ), row=2, col=1)

    # ── Per-trade P&L bars ──
    bar_colors = ["#26a69a" if p > 0 else "#ef5350" for p in trades["pnl_rs"]]
    fig.add_trace(go.Bar(
        x=trades["exit_time"],
        y=trades["pnl_rs"],
        name="Trade P&L (₹)",
        marker_color=bar_colors,
        text=[f"₹{p:+,.0f}" for p in trades["pnl_rs"]],
        textposition="outside",
        textfont=dict(size=10)
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="white", line_dash="dot", row=3, col=1)

    # ── Equity curve ──
    cum_color = "#26a69a" if trades["cum_pnl"].iloc[-1] >= 0 else "#ef5350"
    fig.add_trace(go.Scatter(
        x=trades["exit_time"], y=trades["cum_pnl"],
        name="Cumulative P&L", fill="tozeroy",
        line=dict(color=cum_color, width=2),
        fillcolor=f"rgba(38,166,154,0.15)" if cum_color == "#26a69a" else "rgba(239,83,80,0.15)"
    ), row=4, col=1)
    fig.add_hline(y=0, line_color="white", line_dash="dot", row=4, col=1)

    fig.update_layout(
        height=820,
        template="plotly_dark",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=60, b=40),
        title=dict(text=f"{strategy_label}", font=dict(size=14))
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI",   row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="₹",     row=3, col=1)
    fig.update_yaxes(title_text="₹",     row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────
# STRATEGY A SECTION
# ─────────────────────────────────────────────────────
st.subheader("Strategy A — Intraday Scalping (3-min · RSI-9 · Near-Expiry Options)")
st.markdown("""
**Logic:**
- Signal on NIFTY FUT 3-min · Trade ATM CE/PE on **24-Mar expiry** (near-weekly — high gamma)
- **Long (Buy CE):** RSI(9) crosses above 50 **OR** StochRSI %K crosses %D from oversold (<40) while RSI > 50
- **Short (Buy PE):** RSI(9) crosses below 50 **OR** StochRSI %K crosses %D from overbought (>60) while RSI < 50
- **OI-RSI Filter:** OI-weighted RSI must agree with direction
- **Exit:** RSI(9) crosses 50 in opposite direction · OR Stop Loss · OR Target · OR 15:15 EOD
""")

build_chart(fut_3m, trades_a, "9", "3-min",
            f"Strategy A · Intraday 3-min | SL {a_sl}% · TGT {a_tgt}% · Near-expiry (24-Mar)")

# Trade log A
if not trades_a.empty:
    with st.expander("Strategy A — Trade Log"):
        disp_a = trades_a[["entry_time","exit_time","direction","inst_type","strike",
                             "entry_price","exit_price","exit_reason","pnl_pts","pnl_rs","pnl_pct"]].copy()
        disp_a.columns = ["Entry Time","Exit Time","Direction","Type","Strike",
                           "Entry ₹","Exit ₹","Exit Reason","P&L Pts","P&L ₹","P&L %"]
        def color_pnl(val):
            return "color: #26a69a; font-weight:bold" if val > 0 else "color: #ef5350; font-weight:bold"
        st.dataframe(disp_a.style.applymap(color_pnl, subset=["P&L ₹"]),
                     use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────
# STRATEGY B SECTION
# ─────────────────────────────────────────────────────
st.subheader("Strategy B — Positional / Swing (15-min · RSI-14 · Weekly-Expiry Options)")
st.markdown("""
**Logic:**
- Signal on NIFTY FUT 15-min · Trade ATM CE/PE on **30-Mar expiry** (weekly — more time value)
- **Long (Buy CE):** RSI(14) crosses above 32 (oversold bounce) with OI-RSI agreement
- **Short (Buy PE):** RSI(14) crosses below 68 (overbought drop) with OI-RSI agreement
- **Exit:** RSI(14) crosses 50 · OR Stop Loss · OR Target · OR 15:15 EOD
- *Note: True positional requires multi-day data — this simulates the entry/exit discipline on 15-min.*
""")

build_chart(fut_15m, trades_b, "14", "15-min",
            f"Strategy B · Positional 15-min | SL {b_sl}% · TGT {b_tgt}% · Weekly expiry (30-Mar)")

# Trade log B
if not trades_b.empty:
    with st.expander("Strategy B — Trade Log"):
        disp_b = trades_b[["entry_time","exit_time","direction","inst_type","strike",
                             "entry_price","exit_price","exit_reason","pnl_pts","pnl_rs","pnl_pct"]].copy()
        disp_b.columns = ["Entry Time","Exit Time","Direction","Type","Strike",
                           "Entry ₹","Exit ₹","Exit Reason","P&L ₹ pts","P&L ₹","P&L %"]
        def color_pnl(val):
            return "color: #26a69a; font-weight:bold" if val > 0 else "color: #ef5350; font-weight:bold"
        st.dataframe(disp_b.style.applymap(color_pnl, subset=["P&L ₹"]),
                     use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────
# COMBINED COMPARISON
# ─────────────────────────────────────────────────────
st.subheader("Combined Strategy Comparison")

all_trades = pd.concat([trades_a, trades_b], ignore_index=True)
if not all_trades.empty:
    all_trades = all_trades.sort_values("entry_time").reset_index(drop=True)
    all_trades["cum_pnl_a"] = trades_a["pnl_rs"].cumsum() if not trades_a.empty else pd.Series(dtype=float)
    all_trades["cum_pnl_b"] = trades_b["pnl_rs"].cumsum() if not trades_b.empty else pd.Series(dtype=float)

    fig_cmp = go.Figure()
    if not trades_a.empty:
        trades_a["cum_pnl"] = trades_a["pnl_rs"].cumsum()
        fig_cmp.add_trace(go.Scatter(
            x=trades_a["exit_time"], y=trades_a["cum_pnl"],
            name="Strategy A · Intraday",
            line=dict(color="#2196F3", width=2),
            mode="lines+markers", marker=dict(size=7)
        ))
    if not trades_b.empty:
        trades_b["cum_pnl"] = trades_b["pnl_rs"].cumsum()
        fig_cmp.add_trace(go.Scatter(
            x=trades_b["exit_time"], y=trades_b["cum_pnl"],
            name="Strategy B · Positional",
            line=dict(color="#4CAF50", width=2),
            mode="lines+markers", marker=dict(size=7)
        ))
    fig_cmp.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_cmp.update_layout(
        template="plotly_dark", height=320,
        title="Cumulative P&L Comparison (₹)",
        yaxis_title="₹", xaxis_title="Exit Time",
        legend=dict(orientation="h"),
        margin=dict(l=50, r=30, t=50, b=40)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ─────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────
st.markdown("---")
st.info("""
**Backtest Notes:**
- Based on **1 day of data** — results are indicative only. Statistical significance requires 30+ days / 100+ trades.
- Uses **real option prices** from the feather file (no synthetic pricing).
- Transaction cost: ₹{} per round trip (brokerage + STT approximation).
- No slippage modelled — actual fills may differ especially on near-expiry options.
- For true **positional** backtesting, upload multiple days of feather data.
- NIFTY lot size assumed: **{} shares**.
""".format(int(txn_cost), int(lot_size)))

st.caption("Backtest · 23-Mar-2026 · RSI Research | Branch: claude/rsi-indicator-research-JcNUM")

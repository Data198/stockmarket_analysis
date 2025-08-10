# 7🔴_OI_Phase_Live_NoRepaint.py
# Non‑repainting, bar‑close engine for SB/LB/SC/LU + WA inventories + WA-only signals
# Modes:
#   • Backtest (file upload) → sequential pass, no repaint
#   • Live Mode (simulated poll loop) → append new bars, fire signals at bar close
#
# Optional: log each fired signal to PostgreSQL.

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OI Live (No Repaint)", layout="wide")
st.title("🔴 OI Phase Engine — Live, No‑Repaint (Bar‑Close Signals)")

# ──────────────────────────────────────────────────────────────────────────────
# DB (optional)
# ──────────────────────────────────────────────────────────────────────────────
def get_engine():
    try:
        from sqlalchemy import create_engine
        u = st.secrets["postgres"]["user"]
        p = st.secrets["postgres"]["password"]
        h = st.secrets["postgres"]["host"]
        pr = st.secrets["postgres"]["port"]
        d = st.secrets["postgres"]["database"]
        return create_engine(f"postgresql+psycopg2://{u}:{p}@{h}:{pr}/{d}")
    except Exception:
        return None

def ensure_signal_table(engine):
    from sqlalchemy import text
    ddl = """
    CREATE TABLE IF NOT EXISTS oi_signals (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        session_date DATE NOT NULL,
        ts TIMESTAMP NOT NULL,
        side TEXT NOT NULL,              -- 'LONG' or 'SHORT'
        entry NUMERIC(18,6) NOT NULL,
        stop  NUMERIC(18,6) NOT NULL,
        target NUMERIC(18,6) NOT NULL,
        class  TEXT,                     -- SB/LB/SC/LU at signal bar
        wa_long_avg NUMERIC(18,6),
        wa_short_avg NUMERIC(18,6),
        long_pressure NUMERIC(18,6),
        short_pressure NUMERIC(18,6),
        params JSONB DEFAULT '{}'::jsonb -- thresholds snapshot
    );
    """
    with engine.begin() as con:
        con.execute(text(ddl))

def log_signal(engine, row, meta):
    if engine is None: return
    from sqlalchemy import text
    q = text("""
        INSERT INTO oi_signals
        (symbol, session_date, ts, side, entry, stop, target, class, wa_long_avg, wa_short_avg, long_pressure, short_pressure, params)
        VALUES (:symbol, :session_date, :ts, :side, :entry, :stop, :target, :class, :wa_long_avg, :wa_short_avg, :long_pressure, :short_pressure, :params::jsonb)
    """)
    with engine.begin() as con:
        con.execute(q, {
            "symbol": meta.get("symbol","NIFTY?"),
            "session_date": meta.get("session_date", datetime.today().date()),
            "ts": row["timestamp"],
            "side": row["type"],  # LONG ▲ / SHORT ▼
            "entry": row["entry"],
            "stop": row["stop"],
            "target": row["target"],
            "class": row.get("class"),
            "wa_long_avg": row.get("wa_long_avg"),
            "wa_short_avg": row.get("wa_short_avg"),
            "long_pressure": row.get("long_pressure"),
            "short_pressure": row.get("short_pressure"),
            "params": meta.get("params_json","{}"),
        })

# ──────────────────────────────────────────────────────────────────────────────
# Parsing & helpers (no repaint: we’ll advance STATE sequentially)
# ──────────────────────────────────────────────────────────────────────────────
def parse_time_series(df: pd.DataFrame, date_hint: str | None = None) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    ts_col = next((cols[k] for k in cols if k in ["time","timestamp","date","datetime"] or "time" in k), None)
    price_col = next((cols[k] for k in cols if k in ["close","price","ltp","last price","last traded price"] or "close" in k or "price" in k), None)
    oi_col = next((cols[k] for k in cols if k in ["oi","open interest","open_interest","total oi"] or ("open" in k and "interest" in k)), None)
    if ts_col is None or price_col is None or oi_col is None:
        raise ValueError(f"Could not detect Time/Price/OI columns. Columns: {list(df.columns)}")

    df = df.rename(columns={ts_col:"timestamp", price_col:"close", oi_col:"oi"}).copy()

    def to_ts(x):
        s = str(x)
        try:
            return pd.to_datetime(s)
        except Exception:
            pass
        for fmt in ("%H:%M","%H:%M:%S"):
            try:
                t = pd.to_datetime(s, format=fmt).time()
                base = pd.to_datetime(date_hint).date() if date_hint else datetime.today().date()
                return pd.Timestamp.combine(base, t)
            except Exception:
                continue
        return pd.NaT

    df["timestamp"] = df["timestamp"].apply(to_ts)
    df = df.dropna(subset=["timestamp"])
    for c in ["close","oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close","oi"]).sort_values("timestamp").reset_index(drop=True)
    return df

def classify_from_changes(price_change, oi_change):
    # SB/LB/SC/LU based only on current bar deltas
    if price_change is None or oi_change is None: return None
    if price_change == 0 or oi_change == 0: return None
    if price_change > 0 and oi_change > 0: return "LB"
    if price_change < 0 and oi_change > 0: return "SB"
    if price_change > 0 and oi_change < 0: return "SC"
    if price_change < 0 and oi_change < 0: return "LU"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Sequential engine (NO REPAINT)
# ──────────────────────────────────────────────────────────────────────────────
class EngineState:
    def __init__(self, price_thr_rupees, price_thr_pct, oi_thr_frac, oi_thr_abs, confirm_bars,
                 min_long_pressure, min_short_pressure, require_wa_cross,
                 min_gap_minutes, max_per_side, rr, buffer):
        # thresholds
        self.price_thr_rupees = price_thr_rupees
        self.price_thr_pct    = price_thr_pct
        self.oi_thr_frac      = oi_thr_frac
        self.oi_thr_abs       = oi_thr_abs
        self.confirm_bars     = confirm_bars

        # signal filters (WA-only)
        self.min_long_pressure  = min_long_pressure
        self.min_short_pressure = min_short_pressure
        self.require_wa_cross   = require_wa_cross
        self.min_gap            = timedelta(minutes=min_gap_minutes)
        self.max_per_side       = max_per_side
        self.rr                 = rr
        self.buffer             = buffer

        # running state
        self.prev_close = None
        self.prev_oi    = None
        self.prev_ts    = None

        self.prev_class = None         # last confirmed class
        self.flip_buffer = []          # collect consecutive signals to confirm flip

        # WA inventories
        self.long_qty = 0.0
        self.long_avg = np.nan
        self.short_qty = 0.0
        self.short_avg = np.nan

        # Derived series to display (append-only)
        self.rows = []
        self.phase_id = 0
        self.last_signal_time_long = None
        self.last_signal_time_short = None
        self.signals = []  # list of dicts (signal tape)
        self.count_long = 0
        self.count_short = 0

    def _meets_significance(self, close, oi):
        # dynamic thresholds based on prev values
        if self.prev_close is None or self.prev_oi is None:
            return False
        pc = close - self.prev_close
        oc = oi - self.prev_oi
        dyn_price_thr = max(self.price_thr_rupees, self.price_thr_pct * self.prev_close)
        dyn_oi_thr = max(self.oi_thr_abs, self.oi_thr_frac * abs(self.prev_oi))
        return (abs(pc) >= dyn_price_thr) and (abs(oc) >= dyn_oi_thr)

    def _confirm_class(self, raw_class):
        # returns confirmed class (no repaint: only present & past info)
        if raw_class is None:
            return self.prev_class
        if self.prev_class is None:
            self.prev_class = raw_class
            self.flip_buffer.clear()
            return self.prev_class

        if raw_class == self.prev_class:
            self.flip_buffer.clear()
            return self.prev_class

        # potential flip → need N consecutive confirmations
        self.flip_buffer.append(raw_class)
        if len(self.flip_buffer) >= self.confirm_bars and all(x == raw_class for x in self.flip_buffer):
            self.prev_class = raw_class
            self.flip_buffer.clear()
        return self.prev_class

    def _update_wa(self, seg, oi_change, price):
        # WA-long
        if seg == "LB" and oi_change > 0:
            if self.long_qty <= 0:
                self.long_qty = oi_change
                self.long_avg = price
            else:
                self.long_avg = (self.long_avg*self.long_qty + price*oi_change) / (self.long_qty + oi_change)
                self.long_qty += oi_change
        elif seg == "LU" and oi_change < 0:
            out = -oi_change
            self.long_qty = max(0.0, self.long_qty - out)
            if self.long_qty == 0: self.long_avg = np.nan

        # WA-short
        if seg == "SB" and oi_change > 0:
            if self.short_qty <= 0:
                self.short_qty = oi_change
                self.short_avg = price
            else:
                self.short_avg = (self.short_avg*self.short_qty + price*oi_change) / (self.short_qty + oi_change)
                self.short_qty += oi_change
        elif seg == "SC" and oi_change < 0:
            out = -oi_change
            self.short_qty = max(0.0, self.short_qty - out)
            if self.short_qty == 0: self.short_avg = np.nan

    def _try_signal(self, ts, close, seg, long_pressure, short_pressure):
        # WA-only logic; spacing; cap max signals per side
        out = None
        # SHORT ▼ when longs hurting + bearish phase + price below WA long avg
        if (long_pressure is not None and long_pressure < self.min_long_pressure
            and seg in ("SB","LU") and not np.isnan(self.long_avg) and close < self.long_avg
            and self.count_short < self.max_per_side):

            # optional: require WA-cross on this bar
            wa_cross_ok = True
            if self.require_wa_cross and self.prev_close is not None and not np.isnan(self.long_avg):
                prev_above = (self.prev_close >= self.long_avg)
                now_below  = (close < self.long_avg)
                wa_cross_ok = (prev_above and now_below)

            # spacing
            spacing_ok = (self.last_signal_time_short is None) or (ts - self.last_signal_time_short >= self.min_gap)

            if wa_cross_ok and spacing_ok:
                # stop above swing & WA long avg
                stop = max(close, self.long_avg) + self.buffer
                # using last few bars for high would be “live-ish”; but to avoid repaint, use only current info:
                # we’ll still put SL above WA long avg for safety
                risk = stop - close
                target = close - self.rr * risk
                out = {
                    "timestamp": ts, "type": "SHORT ▼", "entry": round(close,2), "stop": round(stop,2), "target": round(target,2),
                    "class": seg, "wa_long_avg": float(self.long_avg) if not np.isnan(self.long_avg) else None,
                    "wa_short_avg": float(self.short_avg) if not np.isnan(self.short_avg) else None,
                    "long_pressure": float(long_pressure) if long_pressure is not None else None,
                    "short_pressure": float(short_pressure) if short_pressure is not None else None,
                }
                self.last_signal_time_short = ts
                self.count_short += 1

        # LONG ▲ when shorts hurting + bullish phase + price above WA short avg
        if (short_pressure is not None and short_pressure < self.min_short_pressure
            and seg in ("LB","SC") and not np.isnan(self.short_avg) and close > self.short_avg
            and self.count_long < self.max_per_side):

            wa_cross_ok = True
            if self.require_wa_cross and self.prev_close is not None and not np.isnan(self.short_avg):
                prev_below = (self.prev_close <= self.short_avg)
                now_above  = (close > self.short_avg)
                wa_cross_ok = (prev_below and now_above)

            spacing_ok = (self.last_signal_time_long is None) or (ts - self.last_signal_time_long >= self.min_gap)

            if wa_cross_ok and spacing_ok:
                stop = min(close, self.short_avg) - self.buffer
                risk = close - stop
                target = close + self.rr * risk
                out = {
                    "timestamp": ts, "type": "LONG ▲", "entry": round(close,2), "stop": round(stop,2), "target": round(target,2),
                    "class": seg, "wa_long_avg": float(self.long_avg) if not np.isnan(self.long_avg) else None,
                    "wa_short_avg": float(self.short_avg) if not np.isnan(self.short_avg) else None,
                    "long_pressure": float(long_pressure) if long_pressure is not None else None,
                    "short_pressure": float(short_pressure) if short_pressure is not None else None,
                }
                self.last_signal_time_long = ts
                self.count_long += 1

        if out:
            self.signals.append(out)
        return out

    def step_bar(self, row):
        # compute deltas against previous
        ts, close, oi = row["timestamp"], float(row["close"]), float(row["oi"])
        price_change = None if self.prev_close is None else close - self.prev_close
        oi_change    = None if self.prev_oi    is None else oi - self.prev_oi

        # significance filter first (to avoid noise flip)
        seg = None
        if self._meets_significance(close, oi):
            seg = classify_from_changes(price_change, oi_change)

        # confirm class (2‑bar rule) using only past/present info
        confirmed = self._confirm_class(seg)

        # new phase id?
        if not self.rows:
            self.phase_id = 1
        else:
            if confirmed != self.rows[-1]["class"]:
                self.phase_id += 1

        # update WA inventories using confirmed seg and current oi_change
        if price_change is not None and oi_change is not None and confirmed is not None:
            self._update_wa(confirmed, oi_change, close)

        # pressures (based on current WA lines)
        long_pressure  = None if np.isnan(self.long_avg)  else (close - self.long_avg)
        short_pressure = None if np.isnan(self.short_avg) else (self.short_avg - close)

        # record row (append‑only; no rewrite)
        rec = {
            "timestamp": ts, "close": close, "oi": oi,
            "price_change": price_change, "oi_change": oi_change,
            "class": confirmed, "phase_id": self.phase_id,
            "wa_long_avg": None if np.isnan(self.long_avg) else float(self.long_avg),
            "wa_short_avg": None if np.isnan(self.short_avg) else float(self.short_avg),
            "long_pressure": long_pressure, "short_pressure": short_pressure
        }
        self.rows.append(rec)

        # try to fire signal (bar close only — i.e., right here)
        fired = self._try_signal(ts, close, confirmed, long_pressure, short_pressure)

        # update prev refs
        self.prev_close, self.prev_oi, self.prev_ts = close, oi, ts

        return rec, fired

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Core")
symbol = st.sidebar.text_input("Symbol label", value="NIFTY 25xxx CE/PE")
session_date = st.sidebar.date_input("Session Date (for HH:MM files)", value=pd.to_datetime("today").date())

price_thr_rupees = st.sidebar.number_input("Price threshold (₹)", min_value=0.0, value=1.0, step=0.1)
price_thr_pct    = st.sidebar.number_input("Price threshold (% of price)", min_value=0.0, value=0.3, step=0.05) / 100.0
oi_thr_frac      = st.sidebar.number_input("OI threshold (% of OI)", min_value=0.0, value=0.5, step=0.1) / 100.0
oi_thr_abs       = st.sidebar.number_input("OI threshold (absolute)", min_value=0.0, value=500.0, step=100.0)
confirm_bars     = st.sidebar.slider("Confirm flip after N bars", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 WA‑Only Signal Filters")
min_long_pressure = st.sidebar.number_input("Shorts: Long pressure below (₹)", value=-0.8, step=0.1)
min_short_pressure = st.sidebar.number_input("Longs: Short pressure below (₹)", value=-0.8, step=0.1)
require_wa_cross = st.sidebar.checkbox("Require WA‑line cross this bar", value=True)
min_gap_minutes = st.sidebar.slider("Min spacing between signals (min)", 6, 60, 18, step=3)
max_per_side = st.sidebar.slider("Max signals per side", 1, 3, 2)
rr = st.sidebar.number_input("Target Risk:Reward", value=1.5, step=0.1)
buffer = st.sidebar.number_input("SL buffer (₹)", value=0.3, step=0.1)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Backtest (file)", "Live (simulate)"])
log_to_db = st.sidebar.checkbox("Log signals to PostgreSQL", value=False)
engine = get_engine() if log_to_db else None
if engine is None and log_to_db:
    st.sidebar.warning("DB engine unavailable from st.secrets; logging disabled.")
    log_to_db = False
if log_to_db and engine is not None:
    ensure_signal_table(engine)

# Keep thresholds snapshot for logging
params_json = {
    "price_thr_rupees": price_thr_rupees, "price_thr_pct": price_thr_pct,
    "oi_thr_frac": oi_thr_frac, "oi_thr_abs": oi_thr_abs, "confirm_bars": confirm_bars,
    "min_long_pressure": min_long_pressure, "min_short_pressure": min_short_pressure,
    "require_wa_cross": require_wa_cross, "min_gap_minutes": min_gap_minutes,
    "max_per_side": max_per_side, "rr": rr, "buffer": buffer
}
import json
params_json_str = json.dumps(params_json)

# ──────────────────────────────────────────────────────────────────────────────
# Data input
# ──────────────────────────────────────────────────────────────────────────────
if mode == "Backtest (file)":
    uploaded = st.file_uploader("Upload Excel/CSV with Time, Price, OI", type=["xlsx","xls","csv"])
    if uploaded is None:
        st.info("Upload a file to backtest sequentially.")
        st.stop()
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        df0 = pd.read_excel(uploaded, header=None)
        header_row = None
        for i in range(min(50, len(df0))):
            if df0.iloc[i].astype(str).str.contains(r"\bTime\b", case=False, regex=True).any():
                header_row = i; break
        raw = pd.read_excel(uploaded, header=header_row) if header_row is not None else pd.read_excel(uploaded)
        if "Time" in raw.columns and raw.shape[1] > 15:
            raw = raw.iloc[:, :15]

    data = parse_time_series(raw, date_hint=str(session_date))

else:
    st.info("Live simulation expects a CSV that grows over time (or we will stream it row by row).")
    uploaded = st.file_uploader("Upload CSV (Time, Price, OI)", type=["csv"])
    if uploaded is None:
        st.stop()
    data = parse_time_series(pd.read_csv(uploaded), date_hint=str(session_date))

# ──────────────────────────────────────────────────────────────────────────────
# Run sequential engine (no repaint)
# ──────────────────────────────────────────────────────────────────────────────
state = EngineState(price_thr_rupees, price_thr_pct, oi_thr_frac, oi_thr_abs, confirm_bars,
                    min_long_pressure, min_short_pressure, require_wa_cross,
                    min_gap_minutes, max_per_side, rr, buffer)

placeholder_chart, placeholder_signals, placeholder_table = st.empty(), st.empty(), st.empty()

def render(state_df, signals):
    # Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(state_df["timestamp"], state_df["close"], label="Close", linewidth=1.2)

    # phases
    colors = {"LB":"green","SB":"red","SC":"orange","LU":"blue"}
    for _, g in state_df.groupby("phase_id"):
        ax.axvspan(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], color=colors.get(g["class"].iloc[0],"gray"), alpha=0.1)

    ax.plot(state_df["timestamp"], state_df["wa_long_avg"], label="WA Long Avg", linestyle="--", linewidth=1.2)
    ax.plot(state_df["timestamp"], state_df["wa_short_avg"], label="WA Short Avg", linestyle="--", linewidth=1.2)

    # signals
    if len(signals):
        s_longs = [s for s in signals if s["type"].startswith("LONG")]
        s_shorts = [s for s in signals if s["type"].startswith("SHORT")]
        if s_longs:
            ax.scatter([s["timestamp"] for s in s_longs], [s["entry"] for s in s_longs],
                       marker="^", s=90, label="LONG ▲")
        if s_shorts:
            ax.scatter([s["timestamp"] for s in s_shorts], [s["entry"] for s in s_shorts],
                       marker="v", s=90, label="SHORT ▼")

    ax.set_title("Live, No‑Repaint — Bar‑Close Phases & WA Lines", fontsize=14)
    ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
    placeholder_chart.pyplot(fig)

    # Signal tape
    if len(signals):
        tape_df = pd.DataFrame(signals)[["timestamp","type","entry","stop","target","class","wa_long_avg","wa_short_avg","long_pressure","short_pressure"]]
    else:
        tape_df = pd.DataFrame(columns=["timestamp","type","entry","stop","target","class","wa_long_avg","wa_short_avg","long_pressure","short_pressure"])
    placeholder_signals.markdown("### 🧾 Signal Tape (bar‑close)")
    placeholder_signals.dataframe(tape_df, use_container_width=True, hide_index=True)

    # Table (preview)
    cols = ["timestamp","close","oi","class","phase_id","wa_long_avg","wa_short_avg","long_pressure","short_pressure"]
    placeholder_table.markdown("### 🧩 State (preview)")
    placeholder_table.dataframe(state_df[cols], use_container_width=True)

# Backtest sequential pass
if mode == "Backtest (file)":
    rows = []
    for i, r in data.iterrows():
        rec, fired = state.step_bar(r)
        rows.append(rec)
        # (no repaint) we could render progressively; but for speed, render once at the end
    state_df = pd.DataFrame(rows)
    render(state_df, state.signals)

    # Optional DB logging
    if log_to_db and engine is not None and len(state.signals):
        meta = {"symbol": symbol, "session_date": pd.to_datetime(session_date).date(), "params_json": params_json_str}
        for s in state.signals:
            log_signal(engine, s, meta)

# Live simulation (iterate with small delay)
else:
    rows = []
    # Limit loop for safety in Streamlit
    for i, r in data.iterrows():
        rec, fired = state.step_bar(r)
        rows.append(rec)
        state_df = pd.DataFrame(rows)
        render(state_df, state.signals)

        # Log immediately when a signal fires
        if fired and log_to_db and engine is not None:
            meta = {"symbol": symbol, "session_date": pd.to_datetime(session_date).date(), "params_json": params_json_str}
            log_signal(engine, fired, meta)

        # simulate new bar arrival (tune down if you want it faster)
        time.sleep(0.2)

    st.success("Live simulation finished. Engine did not repaint; signals fired at bar close only.")

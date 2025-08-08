# 6📈_OI_Phase_Chart_Filtered.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="OI Phase Chart (Filtered Signals)", layout="wide")
st.title("📈 3‑min Option OI Phases — High‑Probability Signals")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def parse_time_series(df: pd.DataFrame, date_hint: str | None = None) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    ts_col = next((cols[k] for k in cols if k in ["time","timestamp","date","datetime"] or "time" in k), None)
    price_col = next((cols[k] for k in cols if k in ["close","price","ltp","last price","last traded price"] or "close" in k or "price" in k), None)
    oi_col = next((cols[k] for k in cols if k in ["oi","open interest","open_interest","total oi"] or ("open" in k and "interest" in k)), None)
    vwap_col = next((cols[k] for k in cols if "vwap" in k), None)

    if ts_col is None or price_col is None or oi_col is None:
        raise ValueError(f"Could not detect Time/Price/OI columns. Columns found: {list(df.columns)}")

    df = df.rename(columns={ts_col:"timestamp", price_col:"close", oi_col:"oi"}).copy()
    if vwap_col: df = df.rename(columns={vwap_col:"vwap"})

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
    if "vwap" in df.columns:
        df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
    df = df.dropna(subset=["close","oi"]).sort_values("timestamp").reset_index(drop=True)

    # fallback VWAP = simple rolling mean if not available
    if "vwap" not in df.columns or df["vwap"].isna().all():
        df["vwap"] = df["close"].rolling(10, min_periods=1).mean()
    return df

def classify(pc, oic):
    if pd.isna(pc) or pd.isna(oic) or pc == 0 or oic == 0: return None
    if pc > 0 and oic > 0: return "LB"
    if pc < 0 and oic > 0: return "SB"
    if pc > 0 and oic < 0: return "SC"
    if pc < 0 and oic < 0: return "LU"
    return None

def stabilize_classes(df, price_thr_rupees=1.0, price_thr_pct=0.003, oi_thr_frac=0.005, oi_thr_abs=500, confirm_bars=2):
    df["price_change"] = df["close"].diff()
    df["oi_change"] = df["oi"].diff()
    df["dyn_price_thr"] = np.maximum(price_thr_rupees, price_thr_pct * df["close"].shift(1).fillna(df["close"]))
    df["dyn_oi_thr"] = np.maximum(oi_thr_abs, oi_thr_frac * df["oi"].shift(1).abs().fillna(df["oi"].abs()))
    df["raw_class"] = [classify(pc, oic) for pc, oic in zip(df["price_change"], df["oi_change"])]
    sig = (df["price_change"].abs() >= df["dyn_price_thr"]) & (df["oi_change"].abs() >= df["dyn_oi_thr"])
    df["class_sig"] = np.where(sig, df["raw_class"], None)

    out, prev, buf = [], None, []
    for val in df["class_sig"]:
        if val is None:
            out.append(prev); continue
        if prev is None:
            prev = val; out.append(prev); buf.clear(); continue
        if val == prev:
            buf.clear(); out.append(prev); continue
        buf.append(val)
        if len(buf) >= confirm_bars and all(b == val for b in buf):
            prev = val; buf.clear()
        out.append(prev)
    df["class"] = pd.Series(out).ffill()
    df["phase_id"] = (df["class"] != df["class"].shift(1)).cumsum()
    return df

def compute_wa_inventories(df):
    LQ=LA=SQ=SA=[]
    long_qty=0.0; long_avg=np.nan
    short_qty=0.0; short_avg=np.nan
    LQ, LA, SQ, SA = [], [], [], []
    for _, r in df.iterrows():
        seg = r["class"]; d_oi = 0.0 if pd.isna(r["oi_change"]) else float(r["oi_change"]); px = r["close"]
        if seg=="LB" and d_oi>0:
            if long_qty<=0: long_qty=d_oi; long_avg=px
            else: long_avg=(long_avg*long_qty + px*d_oi)/(long_qty+d_oi); long_qty+=d_oi
        elif seg=="LU" and d_oi<0:
            out=-d_oi; long_qty=max(0.0, long_qty-out);  long_avg=np.nan if long_qty==0 else long_avg
        elif seg=="SB" and d_oi>0:
            if short_qty<=0: short_qty=d_oi; short_avg=px
            else: short_avg=(short_avg*short_qty + px*d_oi)/(short_qty+d_oi); short_qty+=d_oi
        elif seg=="SC" and d_oi<0:
            out=-d_oi; short_qty=max(0.0, short_qty-out); short_avg=np.nan if short_qty==0 else short_avg
        LQ.append(long_qty); LA.append(long_avg); SQ.append(short_qty); SA.append(short_avg)
    df["wa_long_qty"]=LQ; df["wa_long_avg"]=LA; df["wa_short_qty"]=SQ; df["wa_short_avg"]=SA
    df["long_pressure"]  = df["close"] - df["wa_long_avg"]
    df["short_pressure"] = df["wa_short_avg"] - df["close"]
    return df

def summarize_phases(df):
    rows=[]
    for pid, g in df.groupby("phase_id"):
        rows.append({
            "phase_id": int(pid),
            "segment": g["class"].iloc[0],
            "start": g["timestamp"].iloc[0],
            "end": g["timestamp"].iloc[-1],
            "bars": len(g),
            "duration_min": len(g)*3,
            "oi_change": float(g["oi"].iloc[-1]-g["oi"].iloc[0]),
            "price_change": float(g["close"].iloc[-1]-g["close"].iloc[0]),
            "start_price": float(g["close"].iloc[0]),
            "end_price": float(g["close"].iloc[-1]),
        })
    return pd.DataFrame(rows)

# ── NEW: filter to 1–2 high-probability signals per side ─────────────────────
def pick_high_prob_signals(df,
                           min_long_pressure=-0.8,    # longs must be < -0.8 to short
                           min_short_pressure=-0.8,   # shorts must be < -0.8 to long
                           require_trend=True,
                           min_gap_minutes=18,
                           max_per_side=2):
    """Return two DataFrames: filtered_shorts (▼) and filtered_longs (▲)."""
    # Base candidates
    shorts = df[
        (df["long_pressure"] < min_long_pressure) &
        (df["class"].isin(["SB","LU"]))
    ].copy()
    longs = df[
        (df["short_pressure"] < min_short_pressure) &
        (df["class"].isin(["LB","SC"]))
    ].copy()

    # Trend filter: below/above WA & VWAP
    if require_trend:
        shorts = shorts[(shorts["close"] < shorts["wa_long_avg"]) & (shorts["close"] < shorts["vwap"])]
        longs  = longs[(longs["close"] > longs["wa_short_avg"]) & (longs["close"] > longs["vwap"])]

    # Spacing rule
    def enforce_spacing(cands):
        keep=[]; last=None; gap=pd.Timedelta(minutes=min_gap_minutes)
        for _, r in cands.iterrows():
            if last is None or (r["timestamp"] - last) >= gap:
                keep.append(r)
                last = r["timestamp"]
            if len(keep) >= max_per_side:
                break
        return pd.DataFrame(keep)

    shorts = enforce_spacing(shorts)
    longs  = enforce_spacing(longs)
    return shorts, longs

# Basic stop/target helper
def make_trade_cards(signals: pd.DataFrame, side: str, df: pd.DataFrame, rr=1.5, buffer=0.3):
    """Create a small decision table for signals."""
    rows=[]
    for _, r in signals.iterrows():
        ts  = r["timestamp"]; px = r["close"]
        if side=="short":
            # SL: highest high of last 3 bars vs WA long avg (whichever is higher), + buffer
            i = df.index[df["timestamp"]==ts][0]
            h3 = df.iloc[max(0, i-3):i+1]["close"].max()
            sl = max(h3, r["wa_long_avg"]) + buffer
            risk = sl - px
            tgt = px - rr*risk
            rows.append({"time":ts, "type":"SHORT ▼", "entry":round(px,2), "stop":round(sl,2), "target":round(tgt,2), "RR":round((px-tgt)/risk if risk>0 else np.nan,2)})
        else:
            # LONG: SL: lowest low of last 3 bars vs WA short avg (whichever is lower), - buffer
            i = df.index[df["timestamp"]==ts][0]
            l3 = df.iloc[max(0, i-3):i+1]["close"].min()
            sl = min(l3, r["wa_short_avg"]) - buffer
            risk = px - sl
            tgt = px + rr*risk
            rows.append({"time":ts, "type":"LONG ▲", "entry":round(px,2), "stop":round(sl,2), "target":round(tgt,2), "RR":round((tgt-px)/risk if risk>0 else np.nan,2)})
    return pd.DataFrame(rows)

def draw_chart(df, shorts_f, longs_f) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["timestamp"], df["close"], label="Close", linewidth=1.2)

    colors = {"LB":"green","SB":"red","SC":"orange","LU":"blue"}
    for _, g in df.groupby("phase_id"):
        ax.axvspan(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], color=colors.get(g["class"].iloc[0],"gray"), alpha=0.1)

    ax.plot(df["timestamp"], df["wa_long_avg"], label="WA Long Avg", linestyle="--", linewidth=1.2)
    ax.plot(df["timestamp"], df["wa_short_avg"], label="WA Short Avg", linestyle="--", linewidth=1.2)
    ax.plot(df["timestamp"], df["vwap"], label="VWAP", linestyle=":", linewidth=1.0)

    if not longs_f.empty:
        ax.scatter(longs_f["timestamp"], longs_f["close"], marker="^", s=90, label="Long Entry (filtered)")
    if not shorts_f.empty:
        ax.scatter(shorts_f["timestamp"], shorts_f["close"], marker="v", s=90, label="Short Entry (filtered)")

    ax.set_title("High‑Probability Signals with SB/LB/SC/LU Phases", fontsize=14)
    ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(); plt.xticks(rotation=45); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150); buf.seek(0)
    return buf

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Filters")
session_date = st.sidebar.date_input("Session Date (for HH:MM files)", value=pd.to_datetime("today").date())
price_thr_rupees = st.sidebar.number_input("Price threshold (₹)", min_value=0.0, value=1.0, step=0.1)
price_thr_pct    = st.sidebar.number_input("Price threshold (% of price)", min_value=0.0, value=0.3, step=0.05) / 100.0
oi_thr_frac      = st.sidebar.number_input("OI threshold (% of OI)", min_value=0.0, value=0.5, step=0.1) / 100.0
oi_thr_abs       = st.sidebar.number_input("OI threshold (absolute)", min_value=0.0, value=500.0, step=100.0)
confirm_bars     = st.sidebar.slider("Confirm flip after N bars", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 High‑Probability Signal Filters")
min_long_pressure = st.sidebar.number_input("Shorts: Long pressure below (₹)", value=-0.8, step=0.1)
min_short_pressure = st.sidebar.number_input("Longs: Short pressure below (₹)", value=-0.8, step=0.1)
require_trend = st.sidebar.checkbox("Require trend (below/above WA & VWAP)", value=True)
min_gap_minutes = st.sidebar.slider("Min spacing between signals (minutes)", 6, 60, 18, step=3)
max_per_side = st.sidebar.slider("Max signals per side", 1, 3, 2)
rr = st.sidebar.number_input("Target Risk:Reward", value=1.5, step=0.1)
buffer = st.sidebar.number_input("SL buffer (₹)", value=0.3, step=0.1)

# ──────────────────────────────────────────────────────────────────────────────
# Upload input
# ──────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Excel/CSV with Time, Price, OI (VWAP optional)", type=["xlsx","xls","csv"])
if uploaded is None:
    st.info("Upload your 3‑min options OI file to continue.")
    st.stop()

try:
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
            raw = raw.iloc[:, :15]  # first panel only

    df = parse_time_series(raw, date_hint=str(session_date))
    df = stabilize_classes(df, price_thr_rupees, price_thr_pct, oi_thr_frac, oi_thr_abs, confirm_bars)
    df = compute_wa_inventories(df)

    shorts_f, longs_f = pick_high_prob_signals(
        df,
        min_long_pressure=min_long_pressure,
        min_short_pressure=min_short_pressure,
        require_trend=require_trend,
        min_gap_minutes=min_gap_minutes,
        max_per_side=max_per_side
    )

except Exception as e:
    st.error(f"Failed to process file: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])
with left:
    img_buf = draw_chart(df, shorts_f, longs_f)
    st.image(img_buf, caption="Filtered entries only (blue ▼ shorts, magenta ▲ longs)")

with right:
    st.markdown("### 📋 Decision Box — Shorts")
    short_cards = make_trade_cards(shorts_f, "short", df, rr=rr, buffer=buffer)
    st.dataframe(short_cards, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Decision Box — Longs")
    long_cards = make_trade_cards(longs_f, "long", df, rr=rr, buffer=buffer)
    st.dataframe(long_cards, use_container_width=True, hide_index=True)

st.markdown("### 🧩 Classified 3‑min (preview)")
show_cols = ["timestamp","close","oi","class","vwap","wa_long_avg","wa_short_avg","long_pressure","short_pressure"]
st.dataframe(df[show_cols], use_container_width=True)

st.download_button("⬇️ Download shorts signals (CSV)", short_cards.to_csv(index=False).encode(), "short_signals.csv", "text/csv")
st.download_button("⬇️ Download longs signals (CSV)",  long_cards.to_csv(index=False).encode(),  "long_signals.csv",  "text/csv")

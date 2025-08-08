import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

st.set_page_config(page_title="OI Phase Chart", layout="wide")
st.title("📈 3‑min Option OI Phases — SB / LB / SC / LU")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def parse_time_series(df: pd.DataFrame, date_hint: str | None = None) -> pd.DataFrame:
    """
    Flexible parser: tries to find timestamp, price, and oi columns even if headers differ.
    Accepts TradingView/Exported tables where 'Time' might be hh:mm and needs a date attached.
    """
    cols = {str(c).strip().lower(): c for c in df.columns}
    # Guess columns
    ts_col = next((cols[k] for k in cols if k in ["time","timestamp","date","datetime"] or "time" in k), None)
    price_col = next((cols[k] for k in cols if k in ["close","price","ltp","last price","last traded price"] or "close" in k or "price" in k), None)
    oi_col = next((cols[k] for k in cols if k in ["oi","open interest","open_interest","total oi"] or "open" in k and "interest" in k), None)

    if ts_col is None or price_col is None or oi_col is None:
        raise ValueError(f"Could not detect Time/Price/OI columns. Columns found: {list(df.columns)}")

    df = df.rename(columns={ts_col: "timestamp", price_col: "close", oi_col: "oi"}).copy()

    # If timestamp is just time-of-day, attach a date (use sidebar date or today as hint)
    def to_ts(x):
        s = str(x)
        try:
            # Full timestamp already
            return pd.to_datetime(s)
        except Exception:
            pass
        # hh:mm / hh:mm:ss case
        try:
            t = pd.to_datetime(s, format="%H:%M").time()
        except Exception:
            try:
                t = pd.to_datetime(s, format="%H:%M:%S").time()
            except Exception:
                return pd.NaT
        base = pd.to_datetime(date_hint).date() if date_hint else datetime.today().date()
        return pd.Timestamp.combine(base, t)

    df["timestamp"] = df["timestamp"].apply(to_ts)
    df = df.dropna(subset=["timestamp"])
    # Ensure numeric
    for c in ["close","oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close","oi"]).sort_values("timestamp").reset_index(drop=True)
    return df

def classify(pc, oic):
    if pd.isna(pc) or pd.isna(oic) or pc == 0 or oic == 0:
        return None
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

    # confirmation: flip only after N (confirm_bars) consecutive signals
    out = []
    prev = None
    buf = []
    for val in df["class_sig"]:
        if val is None:
            out.append(prev)
            continue
        if prev is None:
            prev = val
            out.append(prev)
            buf.clear()
            continue
        if val == prev:
            buf.clear()
            out.append(prev)
            continue
        # potential flip
        buf.append(val)
        if len(buf) >= confirm_bars and all(b == val for b in buf):
            prev = val
            buf.clear()
        out.append(prev)
    df["class"] = pd.Series(out).ffill()
    df["phase_id"] = (df["class"] != df["class"].shift(1)).cumsum()
    return df

def compute_wa_inventories(df):
    long_qty = 0.0; long_avg = np.nan
    short_qty = 0.0; short_avg = np.nan
    LQ, LA, SQ, SA = [], [], [], []

    for _, r in df.iterrows():
        seg = r["class"]; d_oi = 0.0 if pd.isna(r["oi_change"]) else float(r["oi_change"])
        px = r["close"]

        if seg == "LB" and d_oi > 0:
            if long_qty <= 0:
                long_qty = d_oi; long_avg = px
            else:
                long_avg = (long_avg*long_qty + px*d_oi) / (long_qty + d_oi)
                long_qty += d_oi

        elif seg == "LU" and d_oi < 0:
            out = -d_oi; long_qty = max(0.0, long_qty - out)
            if long_qty == 0: long_avg = np.nan

        elif seg == "SB" and d_oi > 0:
            if short_qty <= 0:
                short_qty = d_oi; short_avg = px
            else:
                short_avg = (short_avg*short_qty + px*d_oi) / (short_qty + d_oi)
                short_qty += d_oi

        elif seg == "SC" and d_oi < 0:
            out = -d_oi; short_qty = max(0.0, short_qty - out)
            if short_qty == 0: short_avg = np.nan

        LQ.append(long_qty); LA.append(long_avg)
        SQ.append(short_qty); SA.append(short_avg)

    df["wa_long_qty"] = LQ
    df["wa_long_avg"] = LA
    df["wa_short_qty"] = SQ
    df["wa_short_avg"] = SA
    df["long_pressure"]  = df["close"] - df["wa_long_avg"]     # <0 → longs hurting
    df["short_pressure"] = df["wa_short_avg"] - df["close"]    # <0 → shorts hurting
    return df

def summarize_phases(df):
    rows = []
    for pid, g in df.groupby("phase_id"):
        rows.append({
            "phase_id": int(pid),
            "segment": g["class"].iloc[0],
            "start": g["timestamp"].iloc[0],
            "end": g["timestamp"].iloc[-1],
            "bars": len(g),
            "duration_min": len(g) * 3,
            "oi_change": float(g["oi"].iloc[-1] - g["oi"].iloc[0]),
            "price_change": float(g["close"].iloc[-1] - g["close"].iloc[0]),
            "start_price": float(g["close"].iloc[0]),
            "end_price": float(g["close"].iloc[-1]),
            "start_oi": float(g["oi"].iloc[0]),
            "end_oi": float(g["oi"].iloc[-1]),
        })
    return pd.DataFrame(rows)

def draw_chart(df) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["timestamp"], df["close"], label="Close Price", linewidth=1.2)

    phase_colors = {"LB":"green","SB":"red","SC":"orange","LU":"blue"}
    for _, g in df.groupby("phase_id"):
        seg = g["class"].iloc[0]
        ax.axvspan(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], color=phase_colors.get(seg,"gray"), alpha=0.1)

    ax.plot(df["timestamp"], df["wa_long_avg"], label="WA Long Avg", linestyle="--", linewidth=1.2)
    ax.plot(df["timestamp"], df["wa_short_avg"], label="WA Short Avg", linestyle="--", linewidth=1.2)

    # Trade suggestions
    entry_longs = df[df["short_pressure"] < 0]  # shorts under pressure
    entry_shorts = df[df["long_pressure"]  < 0] # longs under water
    ax.scatter(entry_longs["timestamp"], entry_longs["close"], marker="^", s=70, label="Long Entry (Shorts Squeezed)")
    ax.scatter(entry_shorts["timestamp"], entry_shorts["close"], marker="v", s=70, label="Short Entry (Longs Trapped)")

    ax.set_title("3‑min OI Phases with WA Cost Levels & Trade Suggestions", fontsize=14)
    ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(); plt.xticks(rotation=45); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Classification Settings")
session_date = st.sidebar.date_input("Session Date (for HH:MM data)", value=pd.to_datetime("today").date())

price_thr_rupees = st.sidebar.number_input("Price threshold (₹)", min_value=0.0, value=1.0, step=0.1)
price_thr_pct    = st.sidebar.number_input("Price threshold (% of price)", min_value=0.0, value=0.3, step=0.05) / 100.0
oi_thr_frac      = st.sidebar.number_input("OI threshold (% of OI)", min_value=0.0, value=0.5, step=0.1) / 100.0
oi_thr_abs       = st.sidebar.number_input("OI threshold (absolute)", min_value=0.0, value=500.0, step=100.0)
confirm_bars     = st.sidebar.slider("Confirm flip after N bars", min_value=1, max_value=3, value=2)

st.sidebar.caption("Tip: start with ₹1 + 0.3% price, 0.5% OI, 2‑bar confirmation.")

# ──────────────────────────────────────────────────────────────────────────────
# Input: Upload file
# ──────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Excel/CSV with Time, Price, OI columns", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Upload a file to continue. (Your exported OI breakup Excel/CSV works.)")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        # try to locate header row containing 'Time'
        df0 = pd.read_excel(uploaded, header=None)
        header_row = None
        for i in range(min(50, len(df0))):
            if df0.iloc[i].astype(str).str.contains(r"\bTime\b", case=False, regex=True).any():
                header_row = i
                break
        if header_row is not None:
            raw = pd.read_excel(uploaded, header=header_row)
            # if the export has duplicated columns panel, keep the first 15 cols
            if raw.shape[1] > 15 and "Time" in raw.columns:
                raw = raw.iloc[:, :15]
        else:
            raw = pd.read_excel(uploaded)

    df = parse_time_series(raw, date_hint=str(session_date))
    df = stabilize_classes(
        df,
        price_thr_rupees=price_thr_rupees,
        price_thr_pct=price_thr_pct,
        oi_thr_frac=oi_thr_frac,
        oi_thr_abs=oi_thr_abs,
        confirm_bars=confirm_bars
    )
    df = compute_wa_inventories(df)
    phases = summarize_phases(df)

except Exception as e:
    st.error(f"Failed to process file: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    img_buf = draw_chart(df)
    st.image(img_buf, caption="OI Phases + WA Cost + Trade Suggestions", use_column_width=True)

with right:
    st.markdown("### Phase Counts")
    pc = phases["segment"].value_counts().rename_axis("segment").reset_index(name="count")
    st.dataframe(pc, hide_index=True, use_container_width=True)

    st.markdown("### Largest Builds")
    lb_max = phases.loc[phases["segment"]=="LB","oi_change"].max() if (phases["segment"]=="LB").any() else 0
    sb_max = phases.loc[phases["segment"]=="SB","oi_change"].max() if (phases["segment"]=="SB").any() else 0
    st.metric("Max Long Build OI Add", f"{lb_max:,.0f}")
    st.metric("Max Short Build OI Add", f"{sb_max:,.0f}")

st.markdown("### ⏱️ Phases (runs)")
st.dataframe(phases, use_container_width=True)

st.markdown("### 🧩 Classified 3‑min candles")
show_cols = ["timestamp","close","oi","price_change","oi_change","class",
             "wa_long_qty","wa_long_avg","wa_short_qty","wa_short_avg",
             "long_pressure","short_pressure"]
st.dataframe(df[show_cols], use_container_width=True)

# Downloads
csv1 = df[show_cols].to_csv(index=False).encode()
csv2 = phases.to_csv(index=False).encode()
st.download_button("⬇️ Download Classified Candles CSV", csv1, "classified_candles.csv", "text/csv")
st.download_button("⬇️ Download Phase Summary CSV", csv2, "phase_summary.csv", "text/csv")

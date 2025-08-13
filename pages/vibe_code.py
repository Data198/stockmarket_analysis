(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF'
diff --git a/streamlit_app.py b/streamlit_app.py
--- a/streamlit_app.py
+++ b/streamlit_app.py
@@ -0,0 +1,240 @@
+# -*- coding: utf-8 -*-
+from typing import Dict, List, Tuple
+
+import matplotlib.pyplot as plt
+import numpy as np
+import pandas as pd
+import streamlit as st
+
+
+SIGNAL_ORDER: List[str] = ["LB", "SB", "LU", "SC"]
+COLOR_MAP: Dict[str, str] = {
+    "LB": "#2ca02c",  # green
+    "SB": "#d62728",  # red
+    "LU": "#ff7f0e",  # orange
+    "SC": "#1f77b4",  # blue
+}
+
+
+def get_required_columns() -> List[str]:
+    return ["timestamp", "open", "high", "low", "close", "volume", "oi"]
+
+
+def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
+    normalized_columns = [c.strip().lower() for c in df.columns]
+    missing = [c for c in get_required_columns() if c not in normalized_columns]
+    return (len(missing) == 0, missing)
+
+
+def load_csv_data(uploaded_file) -> pd.DataFrame:
+    try:
+        raw_df = pd.read_csv(uploaded_file)
+    except Exception as exc:
+        raise ValueError(f"Failed to read CSV: {exc}") from exc
+
+    raw_df.columns = [c.strip().lower() for c in raw_df.columns]
+
+    ok, missing = validate_required_columns(raw_df)
+    if not ok:
+        raise ValueError(
+            "Missing required columns: " + ", ".join(missing) +
+            " (expected: timestamp, open, high, low, close, volume, oi)"
+        )
+
+    # Coerce types and clean
+    df = raw_df.copy()
+    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
+    for col in ["open", "high", "low", "close", "volume", "oi"]:
+        df[col] = pd.to_numeric(df[col], errors="coerce")
+
+    # Drop invalid rows and sort
+    df = df.dropna(subset=["timestamp", "close", "oi"]).sort_values("timestamp").reset_index(drop=True)
+    return df
+
+
+def build_sample_data(rows: int = 120) -> pd.DataFrame:
+    rng = np.random.default_rng(42)
+
+    start = pd.Timestamp.today().normalize() + pd.Timedelta(hours=9, minutes=15)
+    timestamps = pd.date_range(start, periods=rows, freq="3min")
+
+    # Synthetic close as a gentle random walk
+    close_changes = rng.normal(loc=0.0, scale=0.6, size=rows)
+    close = 100 + close_changes.cumsum()
+
+    # Open is prior close, typical for intraday bars
+    open_prices = np.concatenate([[close[0]], close[:-1]])
+
+    # High/Low around open/close with small ranges
+    ranges = rng.uniform(0.05, 0.35, size=rows)
+    high = np.maximum(open_prices, close) + ranges
+    low = np.minimum(open_prices, close) - ranges
+
+    # Volume and OI series
+    volume = rng.integers(200, 3000, size=rows)
+
+    # OI random walk with both increases and decreases
+    oi_changes = rng.integers(-300, 400, size=rows)
+    oi = 100_000 + np.cumsum(oi_changes)
+
+    df = pd.DataFrame({
+        "timestamp": timestamps,
+        "open": open_prices,
+        "high": high,
+        "low": low,
+        "close": close,
+        "volume": volume,
+        "oi": oi.astype(int),
+    })
+    return df
+
+
+def compute_deltas_and_signals(df: pd.DataFrame) -> pd.DataFrame:
+    out = df.copy()
+
+    out["delta_price"] = out["close"].diff()
+    out["delta_oi"] = out["oi"].diff()
+
+    conditions = [
+        (out["delta_price"] > 0) & (out["delta_oi"] > 0),  # LB
+        (out["delta_price"] < 0) & (out["delta_oi"] > 0),  # SB
+        (out["delta_price"] < 0) & (out["delta_oi"] < 0),  # LU
+        (out["delta_price"] > 0) & (out["delta_oi"] < 0),  # SC
+    ]
+    out["signal"] = np.select(conditions, SIGNAL_ORDER, default=np.nan)
+    out["signal"] = pd.Categorical(out["signal"], categories=SIGNAL_ORDER, ordered=False)
+
+    return out
+
+
+def filter_valid_signals(df: pd.DataFrame, min_abs_delta_price: float, min_abs_delta_oi: int) -> pd.DataFrame:
+    out = df.copy()
+    out["is_valid"] = (
+        out["signal"].notna()
+        & (out["delta_price"].abs() >= float(min_abs_delta_price))
+        & (out["delta_oi"].abs() >= int(min_abs_delta_oi))
+    )
+    return out
+
+
+def summarize_signals(df: pd.DataFrame) -> Dict[str, int]:
+    counts = {sig: int(df.loc[df["signal"] == sig].shape[0]) for sig in SIGNAL_ORDER}
+    return counts
+
+
+def plot_close_with_signals(df: pd.DataFrame, min_abs_delta_price: float, min_abs_delta_oi: int):
+    fig, ax = plt.subplots(figsize=(11, 4))
+    ax.plot(df["timestamp"], df["close"], color="#444", linewidth=1.5, label="Close")
+
+    valid_mask = (
+        df["signal"].notna()
+        & (df["delta_price"].abs() >= float(min_abs_delta_price))
+        & (df["delta_oi"].abs() >= int(min_abs_delta_oi))
+    )
+
+    for sig in SIGNAL_ORDER:
+        mask = (df["signal"] == sig) & valid_mask
+        ax.scatter(
+            df.loc[mask, "timestamp"],
+            df.loc[mask, "close"],
+            s=40,
+            color=COLOR_MAP[sig],
+            label=sig,
+            zorder=3,
+        )
+
+    ax.set_title("Close with OI Signals (LB/SB/LU/SC)")
+    ax.set_xlabel("Time")
+    ax.set_ylabel("Price")
+    ax.grid(True, alpha=0.25)
+    ax.legend(ncol=4, frameon=False, loc="upper left")
+    fig.autofmt_xdate()
+
+    return fig
+
+
+# --------------- Streamlit App ---------------
+st.set_page_config(page_title="OI Signal Dashboard", layout="wide")
+st.title("Intraday OI Signal Dashboard - LB / SB / LU / SC")
+st.caption(
+    "Upload a CSV with columns: timestamp, open, high, low, close, volume, oi. "
+    "Use the sliders to filter signals by minimum |Delta Price| and |Delta OI|."
+)
+
+uploaded = st.file_uploader("Upload CSV file", type=["csv"])  # single page
+
+if uploaded is None:
+    st.info("No file uploaded. Using a small synthetic 3-min sample dataset so you can explore the app.")
+    data = build_sample_data(rows=120)
+else:
+    try:
+        data = load_csv_data(uploaded)
+    except ValueError as e:
+        st.error(str(e))
+        st.stop()
+
+# Compute signals
+with_signals = compute_deltas_and_signals(data)
+all_signals = with_signals.loc[with_signals["signal"].notna()].copy()
+
+# Slider ranges derived from data
+max_abs_dp = float(np.nanmax(np.abs(with_signals["delta_price"])) if with_signals["delta_price"].notna().any() else 1.0)
+max_abs_doi = int(np.nanmax(np.abs(with_signals["delta_oi"])) if with_signals["delta_oi"].notna().any() else 1)
+
+# Reasonable slider defaults
+price_slider_max = round(max(0.01, max_abs_dp), 2)
+price_slider_step = 0.01
+
+oi_slider_max = max(1, int(max_abs_doi))
+oi_slider_step = 1
+
+col_th1, col_th2 = st.columns(2)
+with col_th1:
+    min_abs_delta_price = st.slider(
+        "Minimum |Delta Price|",
+        min_value=0.0,
+        max_value=float(price_slider_max),
+        value=0.0,
+        step=float(price_slider_step),
+        help="Bars must have absolute price change at least this amount to be considered a valid signal.",
+    )
+with col_th2:
+    min_abs_delta_oi = st.slider(
+        "Minimum |Delta OI|",
+        min_value=0,
+        max_value=int(oi_slider_max),
+        value=0,
+        step=int(oi_slider_step),
+        help="Bars must have absolute OI change at least this amount to be considered a valid signal.",
+    )
+
+scored = filter_valid_signals(with_signals, min_abs_delta_price, min_abs_delta_oi)
+valid_signals = scored.loc[scored["is_valid"]].copy()
+
+# Summary counts
+total_counts = summarize_signals(all_signals)
+filtered_counts = summarize_signals(valid_signals)
+
+st.subheader("Summary counts")
+summary_df = pd.DataFrame({
+    "signal": SIGNAL_ORDER,
+    "total": [total_counts[s] for s in SIGNAL_ORDER],
+    "filtered": [filtered_counts[s] for s in SIGNAL_ORDER],
+}).set_index("signal")
+st.dataframe(summary_df, use_container_width=True)
+
+# Latest 20 filtered rows
+st.subheader("Latest 20 valid signals")
+if valid_signals.empty:
+    st.info("No rows pass the current thresholds.")
+else:
+    display_cols = [
+        "timestamp", "open", "high", "low", "close", "volume", "oi", "delta_price", "delta_oi", "signal",
+    ]
+    latest = valid_signals.sort_values("timestamp", ascending=False)[display_cols].head(20)
+    st.dataframe(latest, use_container_width=True)
+
+# Chart
+st.subheader("Chart")
+fig = plot_close_with_signals(with_signals, min_abs_delta_price, min_abs_delta_oi)
+st.pyplot(fig, clear_figure=True)
EOF
)

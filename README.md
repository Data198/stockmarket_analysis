# OI Signal Dashboard (Streamlit)

A tiny single-page Streamlit dashboard that ingests a CSV of 3-minute OHLC + OI and surfaces intraday signals LB/SB/LU/SC.

## Requirements
- Python 3.9+
- Dependencies: streamlit, pandas, matplotlib

Install:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

Upload a CSV with columns:
- timestamp
- open
- high
- low
- close
- volume
- oi

If no CSV is uploaded, a small synthetic dataset is generated so you can try the app.

import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from urllib.parse import quote
import streamlit as st
from login_page import login

# --- DB connection ---
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- Enforce login ---
if not login(engine):
    st.stop()

# --- Page UI ---
st.title("📤 Insert Bhavcopies to Database")
st.markdown("This tool loads a selected bhavcopy `.csv` file based on date from your local folder into the PostgreSQL database.")
FOLDER_PATH = "F:/My Drive/Personal Info/Stock Market/New OI Analysis"
st.write(f"📁 Folder path: `{FOLDER_PATH}`")

# --- Helper Functions ---
def already_loaded(trade_date):
    query = text("SELECT 1 FROM fact_oi_bhavcopy WHERE trade_date = :dt LIMIT 1")
    with engine.connect() as conn:
        return conn.execute(query, {"dt": trade_date}).fetchone() is not None

def log_insert(filename):
    query = text("INSERT INTO bhavcopy_load_log (filename, loaded_on) VALUES (:file, now()) ON CONFLICT DO NOTHING")
    with engine.begin() as conn:
        conn.execute(query, {"file": filename})

def log_skip(filepath, reason):
    query = text("INSERT INTO bhavcopy_load_log (filename, loaded_on, skip_reason) VALUES (:file, now(), :reason) ON CONFLICT DO NOTHING")
    with engine.begin() as conn:
        conn.execute(query, {"file": os.path.basename(filepath), "reason": reason})

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.warning(f"❌ Could not read `{os.path.basename(filepath)}`: {e}")
        log_skip(filepath, f"Read error: {e}")
        return None

    df = df.rename(columns={
        'TradDt': 'trade_date',
        'TckrSymb': 'symbol',
        'XpryDt': 'expiry_date',
        'OptnTp': 'option_type',
        'StrkPric': 'strike_price',
        'ClsPric': 'premium',
        'ChngInOpnIntrst': 'change_in_oi',
        'OpnIntrst': 'open_interest'
    })

    df = df[['trade_date', 'symbol', 'expiry_date', 'option_type',
             'strike_price', 'premium', 'change_in_oi', 'open_interest']]
    df = df.dropna(subset=['strike_price', 'premium', 'change_in_oi', 'open_interest'])

    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed', dayfirst=True).dt.date
    df['expiry_date'] = pd.to_datetime(df['expiry_date'], format='mixed', dayfirst=True).dt.date

    df['ce_oi_change'] = df.apply(lambda row: row['change_in_oi'] * row['premium'] if row['option_type'] == 'CE' else None, axis=1)
    df['pe_oi_change'] = df.apply(lambda row: row['change_in_oi'] * row['premium'] if row['option_type'] == 'PE' else None, axis=1)
    df['ce_oi_eod'] = df.apply(lambda row: row['open_interest'] * row['premium'] if row['option_type'] == 'CE' else None, axis=1)
    df['pe_oi_eod'] = df.apply(lambda row: row['open_interest'] * row['premium'] if row['option_type'] == 'PE' else None, axis=1)
    df['resistance'] = df.apply(lambda row: row['strike_price'] + row['premium'] if row['option_type'] == 'CE' else None, axis=1)
    df['support'] = df.apply(lambda row: row['strike_price'] - row['premium'] if row['option_type'] == 'PE' else None, axis=1)

    return df

# --- Main Loader Based on Date Selection ---
def load_bhavcopy_by_date(selected_date):
    inserted = []
    skipped = []

    date_str = selected_date.strftime("%Y%m%d")
    matching_files = [
        f for f in os.listdir(FOLDER_PATH)
        if f.endswith(".csv") and date_str in f
    ]

    if not matching_files:
        st.warning(f"⚠️ No bhavcopy found for {selected_date.strftime('%d-%b-%Y')}")
        return inserted, skipped

    for file in matching_files:
        full_path = os.path.join(FOLDER_PATH, file)
        df = process_file(full_path)

        if df is None or df.empty:
            skipped.append(file)
            log_skip(full_path, "Empty or invalid data")
            continue

        trade_date = df['trade_date'].iloc[0]

        if already_loaded(trade_date):
            skipped.append(file)
            log_skip(full_path, "Already loaded")
            continue

        try:
            df.to_sql("fact_oi_bhavcopy", con=engine, if_exists='append', index=False)
            log_insert(file)
            inserted.append(file)
        except Exception as e:
            st.error(f"❌ Failed to insert `{file}`: {e}")
            skipped.append(file)
            log_skip(full_path, f"Insert error: {e}")

    return inserted, skipped

# --- UI Section ---
selected_date = st.date_input("📅 Select Bhavcopy Date to Insert", datetime.today().date())

if st.button("📤 Insert Bhavcopy for Selected Date"):
    with st.spinner("🔄 Processing and uploading..."):
        inserted, skipped = load_bhavcopy_by_date(selected_date)

    st.success(f"✅ {len(inserted)} file(s) inserted successfully!")
    if skipped:
        st.warning(f"⚠️ {len(skipped)} file(s) skipped.")
        st.code("\n".join(skipped), language="text")

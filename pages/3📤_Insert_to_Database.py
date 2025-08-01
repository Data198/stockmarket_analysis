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

FOLDER_PATH = r"F:\My Drive\Personal Info\Stock Market\New OI Analysis"
st.write(f"📁 Folder path: `{FOLDER_PATH}`")

# --- Check if folder exists ---
if not os.path.exists(FOLDER_PATH):
    st.error(f"❌ Folder path does not exist: `{FOLDER_PATH}`. Please update the path to a valid directory.")
    st.stop()

# --- Helper Functions ---
def already_loaded(trade_date):
    query = text("SELECT 1 FROM fact_oi_bhavcopy WHERE trade_date = :dt LIMIT 1")
    with engine.connect() as conn:
        return conn.execute(query, {"dt": trade_date}).fetchone() is not None

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.warning(f"❌ Could not read `{os.path.basename(filepath)}`: {e}")
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
            continue

        trade_date = df['trade_date'].iloc[0]

        if already_loaded(trade_date):
            skipped.append(file)
            continue

        try:
            df.to_sql("fact_oi_bhavcopy", con=engine, if_exists='append', index=False)
            inserted.append(file)
        except Exception as e:
            st.error(f"❌ Failed to insert `{file}`: {e}")
            skipped.append(file)

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

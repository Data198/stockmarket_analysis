import streamlit as st
import requests
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import io

# === Target directory for saving only the extracted CSV ===
TARGET_FOLDER = r"F:\My Drive\Personal Info\Stock Market\New OI Analysis"

def get_latest_available_date():
    now = datetime.now()
    if now.weekday() >= 5:
        return get_previous_weekday()
    elif now.hour >= 18:
        return now
    else:
        return get_previous_weekday()

def get_previous_weekday():
    today = datetime.now()
    offset = 1
    while True:
        prev_day = today - timedelta(days=offset)
        if prev_day.weekday() < 5:
            return prev_day
        offset += 1

def download_and_extract_csv_only(date_obj, save_dir=TARGET_FOLDER):
    date_str = date_obj.strftime("%Y%m%d")
    url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }

    try:
        session = requests.Session()
        # First visit NSE homepage to get cookies
        home_response = session.get("https://www.nseindia.com", headers=headers)
        if home_response.status_code != 200:
            return f"❌ Failed to access NSE homepage | Status: {home_response.status_code}"

        # Now download the Bhavcopy zip with same session and headers
        response = session.get(url, headers=headers)
        if response.status_code == 200:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file in z.namelist():
                    if file.endswith(".csv"):
                        z.extract(file, path=save_path)
                        return f"✅ CSV extracted: {save_path / file}"
            return "❌ No CSV file found inside the zip."
        elif response.status_code == 404:
            return f"❌ File not found for {date_str} (404)"
        else:
            return f"❌ Download failed for {date_str} | Status: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def download_date_range(start_date, end_date, save_dir=TARGET_FOLDER):
    results = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            result = download_and_extract_csv_only(current_date, save_dir)
            results.append(result)
        current_date += timedelta(days=1)
    return results

# Streamlit UI
st.title("NSE FnO Bhavcopy Downloader")

# Single date download
st.header("Download for a Single Date")
single_date = st.date_input(
    "Select Date",
    value=datetime.now().date() - timedelta(days=1),
    max_value=datetime.now().date(),
    key="single_date"
)
if st.button("Download Single Date"):
    result = download_and_extract_csv_only(datetime.combine(single_date, datetime.min.time()))
    st.write(result)

# Date range download
st.header("Download for a Date Range")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now().date() - timedelta(days=7),
        max_value=datetime.now().date(),
        key="start_date"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date(),
        key="end_date"
    )
if st.button("Download Date Range"):
    if start_date <= end_date:
        with st.spinner("Downloading..."):
            results = download_date_range(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time())
            )
        for res in results:
            st.write(res)
    else:
        st.error("Start date must be before or equal to end date.")

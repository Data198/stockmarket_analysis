import streamlit as st
from login_page import login
from sqlalchemy import create_engine
from urllib.parse import quote

# Modern, active UI Styling
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #fff !important;
            color: #111 !important;
        }
        .stSidebar {
            background: #fff !important;
        }
        .stSidebar .stMarkdown {
            color: #111 !important;
        }
        .stTitle {
            color: #111 !important;
        }
        .stMarkdown {
            color: #111 !important;
        }
        .icon-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #f1f5f9;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            padding: 1.2em 0.5em 0.8em 0.5em;
            margin-bottom: 1em;
            transition: box-shadow 0.2s, border 0.2s;
            cursor: pointer;
            min-height: 110px;
        }
        .icon-btn:hover {
            box-shadow: 0 4px 24px rgba(37,99,235,0.10);
            border: 1.5px solid #2563eb;
        }
        .icon-emoji {
            font-size: 2.2rem;
            margin-bottom: 0.3em;
        }
        .icon-label {
            font-size: 1.05rem;
            color: #111;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# DB connection
user = st.secrets["postgres"]["user"]
password = quote(st.secrets["postgres"]["password"])
host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
db = st.secrets["postgres"]["database"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# Auth
if not login(engine):
    st.stop()

st.title("Welcome to Stock Market Analytics")
st.markdown("**Navigate using the icons below:**")

# Define your pages/features and icons
pages = [
    {"name": "Download Bhavcopy", "icon": "📥"},
    {"name": "Intraday Analysis", "icon": "📊"},
    {"name": "OI MultiDay Trends", "icon": "📈"},
    {"name": "Live OHLC", "icon": "⏱️"},
    {"name": "Options 1min OHLC", "icon": "🕒"},
    {"name": "Smart Signal Zone", "icon": "🚦"},
    {"name": "Sentiment Tag", "icon": "🧠"},
    {"name": "Upload Option OHLC", "icon": "📅"},
    {"name": "Upload Nifty 3-Min OHLC", "icon": "📥"},
    {"name": "OI Interpretation", "icon": "🔎"},
]

# --- PAGE FUNCTION PLACEHOLDERS ---
def download_bhavcopy_app():
    st.subheader("📥 Download Bhavcopy")
    st.write("This is the Download Bhavcopy page. (Insert your logic here)")

def intraday_analysis_app():
    st.subheader("📊 Intraday Analysis")
    st.write("This is the Intraday Analysis page. (Insert your logic here)")

def oi_multiday_trends_app():
    st.subheader("📈 OI MultiDay Trends")
    st.write("This is the OI MultiDay Trends page. (Insert your logic here)")

def live_ohlc_app():
    st.subheader("⏱️ Live OHLC")
    st.write("This is the Live OHLC page. (Insert your logic here)")

def options_1min_ohlc_app():
    st.subheader("🕒 Options 1min OHLC")
    st.write("This is the Options 1min OHLC page. (Insert your logic here)")

def smart_signal_zone_app():
    st.subheader("🚦 Smart Signal Zone")
    st.write("This is the Smart Signal Zone page. (Insert your logic here)")

def sentiment_tag_app():
    st.subheader("🧠 Sentiment Tag")
    st.write("This is the Sentiment Tag page. (Insert your logic here)")

def upload_option_ohlc_app():
    st.subheader("📅 Upload Option OHLC")
    st.write("This is the Upload Option OHLC page. (Insert your logic here)")

def upload_nifty_3min_ohlc_app():
    st.subheader("📥 Upload Nifty 3-Min OHLC")
    st.write("This is the Upload Nifty 3-Min OHLC page. (Insert your logic here)")

def oi_interpretation_app():
    st.subheader("🔎 OI Interpretation")
    st.write("This is the OI Interpretation page. (Insert your logic here)")

# Map page names to functions
page_functions = {
    "Download Bhavcopy": download_bhavcopy_app,
    "Intraday Analysis": intraday_analysis_app,
    "OI MultiDay Trends": oi_multiday_trends_app,
    "Live OHLC": live_ohlc_app,
    "Options 1min OHLC": options_1min_ohlc_app,
    "Smart Signal Zone": smart_signal_zone_app,
    "Sentiment Tag": sentiment_tag_app,
    "Upload Option OHLC": upload_option_ohlc_app,
    "Upload Nifty 3-Min OHLC": upload_nifty_3min_ohlc_app,
    "OI Interpretation": oi_interpretation_app,
}

# --- ICON GRID NAVIGATION ---
cols = st.columns(3)
if "page" not in st.session_state:
    st.session_state["page"] = None

for idx, page in enumerate(pages):
    with cols[idx % 3]:
        if st.button(f"{page['icon']} {page['name']}", key=page['name'], use_container_width=True):
            st.session_state["page"] = page["name"]

# --- PAGE ROUTING ---
selected_page = st.session_state.get('page', None)
if selected_page:
    # Call the corresponding function
    page_functions[selected_page]()
else:
    st.markdown(
        "<div style='margin-top:2em; color:#888;'>Select a feature above to get started.</div>",
        unsafe_allow_html=True
    )
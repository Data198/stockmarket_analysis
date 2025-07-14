# Home.py
import streamlit as st
from login_page import login
from sqlalchemy import create_engine
from urllib.parse import quote

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

# Welcome
st.title("📈 F&O Bhavcopy Analytics")
st.markdown("Navigate using the sidebar to download bhavcopies or analyze Open Interest trends.")

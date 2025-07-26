# login_page.py
import streamlit as st
from sqlalchemy import text
import bcrypt

# Modern, active UI Styling
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #fff !important;
            color: #111 !important;
        }
        .stTextInput > div > div > input {
            background-color: #fff !important;
            color: #111 !important;
            border: 1px solid #ddd !important;
            border-radius: 5px;
            transition: border 0.2s, box-shadow 0.2s;
        }
        .stTextInput > div > div > input:focus {
            border: 1.5px solid #2563eb !important;
            box-shadow: 0 0 0 2px #2563eb22;
        }
        .stButton > button {
            background-color: #2563eb !important;
            color: #fff !important;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background-color: #1e40af !important;
        }
        .stForm {
            background: #fff !important;
            border-radius: 10px;
            box-shadow: 0 4px 24px rgba(37,99,235,0.08);
            padding: 2em 1.5em;
            border: 1px solid #e5e7eb;
        }
        .stSidebar {
            background: #fff !important;
        }
        .stSidebar .stMarkdown {
            color: #111 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def login(engine):
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    def logout():
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    def login_form():
        st.title("🔐 Login to OI Analytics")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT password_hash FROM users WHERE username = :u"), {"u": username}).fetchone()
                    if result and bcrypt.checkpw(password.encode(), result[0].encode()):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success(f"Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

    if not st.session_state.logged_in:
        login_form()
        return False
    else:
        formatted_name = st.session_state.username.replace('_', ' ').title()
        st.sidebar.markdown(f"**Welcome, {formatted_name} 👋**")
        if st.sidebar.button("Logout"):
            logout()
        return True
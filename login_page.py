# login_page.py
import streamlit as st
from sqlalchemy import text
import bcrypt

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

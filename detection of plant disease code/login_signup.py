import streamlit as st
import os
import json

# Directory and file for storing user credentials
USER_DATA_DIR = 'users'
USER_DATA_FILE = os.path.join(USER_DATA_DIR, 'users.json')

# Check if the user data directory exists, otherwise create it
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# Check if the user data file exists, otherwise create an empty JSON file
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump({}, file)  # Initialize with an empty dictionary

def load_users():
    """Load user data from the JSON file, ensuring valid JSON is loaded."""
    try:
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        with open(USER_DATA_FILE, 'w') as file:
            json.dump({}, file)
        return {}

def save_user(username, password):
    """Function to save new user to the JSON file"""
    users = load_users()
    users[username] = password
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file, indent=4)

def validate_user(username, password):
    """Function to validate user credentials from the JSON file"""
    users = load_users()
    return username in users and users[username] == password

def login_signup():
    """Handle login/signup interface"""
    if "action" not in st.session_state:
        st.session_state["action"] = "Login"

    action = st.session_state["action"]

    st.markdown("""
        <div style="max-width: 400px; margin: auto;">
    """, unsafe_allow_html=True)

    if action == "Login":
        st.subheader("Login")
        username = st.text_input("Username", key="login_username", help="Enter your username")
        password = st.text_input("Password", key="login_password", type="password", help="Enter your password")
        if st.button("Login", key="login_btn"):
            if validate_user(username, password):
                st.success(f"Welcome back, {username}!")
                return True
            else:
                st.error("Invalid credentials. Please try again.")
        if st.button("Switch to Sign Up", key="switch_signup_btn"):
            st.session_state["action"] = "Signup"

    elif action == "Signup":
        st.subheader("Sign Up")
        new_username = st.text_input("New Username", key="signup_username", help="Choose a new username")
        new_password = st.text_input("New Password", key="signup_password", type="password", help="Choose a new password")
        confirm_password = st.text_input("Confirm Password", key="signup_confirm_password", type="password", help="Confirm your password")
        if st.button("Sign Up", key="signup_btn"):
            users = load_users()
            if new_username in users:
                st.error("Username already exists. Please choose a different one.")
            elif new_password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                save_user(new_username, new_password)
                st.success(f"Account created for {new_username}! Please log in.")
                st.session_state["action"] = "Login"
        if st.button("Switch to Login", key="switch_login_btn"):
            st.session_state["action"] = "Login"

    st.markdown("</div>", unsafe_allow_html=True)
    return False

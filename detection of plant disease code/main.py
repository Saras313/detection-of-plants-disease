import streamlit as st
from login_signup import login_signup
from model_prediction import upload_and_predict

def load_css(file_path):
    """Load a CSS file and return its contents."""
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"

def main():
    """Main function to run the Streamlit app"""
    # Load and apply global styles
    css = load_css("styles.css")  # Ensure styles.css is in the same directory as main.py
    st.markdown(css, unsafe_allow_html=True)

    # Inject a custom navbar
    st.markdown(
        """
        <div class="navbar">
            <a class="navbar-brand">Plant Disease Detection System</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state for login status
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Login/Signup process
    if not st.session_state["logged_in"]:
        if login_signup():  # Successful login
            st.session_state["logged_in"] = True

    # Show the homepage after login
    if st.session_state["logged_in"]:
        st.write("# Welcome to the Plant Disease Detection System")
        upload_and_predict()

if __name__ == '__main__':
    main()

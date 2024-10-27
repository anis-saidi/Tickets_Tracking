import streamlit as st



# Login user by checking credentials
def login_user(username, password):
    from db_functions import get_user, check_password,register_user
    user = get_user(username)
    if user:
        stored_password = user[1]
        if check_password(password, stored_password):
            return user
    return None

# Function to manage the login session
def manage_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        st.write(f"Hello, {st.session_state['username']}! You are logged in as {st.session_state['role']}.")
        
        # Logout button
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.experimental_rerun()
    else:
        st.write("Please log in.")

import streamlit as st
from streamlit_option_menu import option_menu
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from zammad_py import ZammadAPI
from db_functions import connect_to_db, register_user
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from auth import login_user
import psycopg2
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Ollama model
model = OllamaFunctions(model="llama3.1", temperature=0, format="json")

# Define the prompt for ticket creation and general response
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. If the input is related to an issue, use the create-ticket function to generate a ticket. If the input is not related to a ticket, respond with a friendly message and provide general help.",
        ),
        ("human", "{input}"),
    ]
)

# Define the function for creating tickets
func = [
    {
        "name": "create_ticket",
        "description": "Create a ticket for a user in Zammad and generate a response.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the issue"},
                "description": {"type": "string", "description": "Detailed description of the issue"},
                "phone": {"type": "string", "description": "Phone number of the customer"},
                "group_id": {"type": "integer", "description": "ID of the group to assign the ticket to"},
                "user_message": {"type": "string", "description": "Message to send back to the user"}
            },
            "required": ["title", "description", "group_id", "user_message"]
        }
    }
]

# Bind the function to the model
model = model.bind(functions=func)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Connect to the database
def connect_to_db():
    conn = psycopg2.connect(
        host="localhost",
        database="poc_order",
        user="postgres",
        password="hpcai123"
    )
    return conn

# Function to handle user input and create a ticket if necessary
def create_ticket_from_input(user_input):
    logging.info(f"Received user input: {user_input}")
   
    # Check for greetings and general responses
    if user_input.lower() in ["hello", "hi", "hey"]:
        response_message = "Hello! How can I assist you today?"
        logging.info(response_message)
        return {}, response_message

    try:
        # Prepare the input for the model
        input_data = {"input": user_input}

        # Invoke the model
        run = prompt | model
        logging.info(f"Invoking model with input: {input_data}")

        # The input must be passed as a dictionary containing a string for 'input'
        response = run.invoke(input_data)

        # If the response has no tool calls, return a general response
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            logging.info("No tool calls returned, responding with a general message.")
            return {}, "Sorry, I'm here to help you solve issues and create tickets. How can I assist you today?"

        # Extract tool calls for creating a ticket
        tool_calls = response.tool_calls
        ticket_data = {}
        user_message = ""

        for tool_call in tool_calls:
            if tool_call['name'] == 'create_ticket':
                ticket_data = tool_call['args']
                user_message = ticket_data.pop('user_message', '')

        # Ensure ticket data is not empty
        if not ticket_data:
            logging.error("No valid ticket data returned.")
            return {}, "I'm here to help you solve issues and create tickets. How can I assist you today?"

        logging.info(f"Extracted ticket data: {ticket_data}")
        return ticket_data, user_message

    except KeyError as e:
        logging.error(f"KeyError: {e} - Please check your prompt formatting.")
        return {}, "There was an error processing your request."
    except Exception as e:
        logging.error(f"Unexpected error invoking model: {e}")
        return {}, "There was an issue processing your request."

# Function to create a ticket in Zammad
def create_ticket_in_zammad(ticket_data):
    if 'title' not in ticket_data or 'description' not in ticket_data:
        raise ValueError("Missing required fields: 'title' and 'description'")

    client = ZammadAPI(url='https://anissaidi13.zammad.com/api/v1/', username='moansa55@gmail.com', password='Internship123')

    title = ticket_data['title']
    description = ticket_data['description']
    group_id = ticket_data['group_id']

    # Get the logged-in user's information from the session
    username = st.session_state.get('username')
    role = st.session_state.get('role')

    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, username, phone FROM users WHERE username = %s AND role = 'driver'", (username,))
    user_info = cursor.fetchone()

    if not user_info:
        raise ValueError("Logged-in driver's information could not be found in the database.")

    customer_id = user_info[0]
    username = user_info[1]
    phone = user_info[2]

    # First, check if the customer already exists in Zammad using their phone number
    zammad_user_id = None
    page = 1  # Start with page 1 for paginated results
    try:
        # Debug: Check the phone number being used for comparison
        normalized_phone = phone.strip().replace(" ", "")
        logging.info(f"Normalized phone from database: {normalized_phone}")

        while True:
            users = client.user.all(page=page)  # Fetch users page by page
            if not users:
                break  # If no users are returned, we've fetched all pages

            for user in users:
                zammad_phone = user.get('phone', '').strip().replace(" ", "")
                logging.info(f"Checking Zammad user ID {user['id']} with phone: {zammad_phone}")

                if zammad_phone == normalized_phone:
                    zammad_user_id = user['id']
                    logging.info(f"Matching Zammad user found: {zammad_user_id}")
                    break

            if zammad_user_id:
                break  # Exit the loop if we've found the user
            page += 1  # Move to the next page if the user was not found on the current page

        # If the user does not exist in Zammad, create a new user
        if not zammad_user_id:
            logging.info(f"No matching Zammad user found for phone: {phone}. Creating a new user.")
            new_user = client.user.create({
                "firstname": username,
                "lastname": "Driver",
                "phone": phone,
                "role_ids": [3],  # Assuming role_id 3 is for drivers
            })
            zammad_user_id = new_user['id']
            logging.info(f"New Zammad user created with ID: {zammad_user_id}")
    except Exception as e:
        logging.error(f"Failed to fetch or create the user in Zammad: {e}")
        raise ValueError(f"Zammad API error while handling user: {e}")

    if not zammad_user_id:
        raise ValueError("Failed to retrieve or create Zammad user.")

    # Now that we have the Zammad user ID, create the ticket
    params = {
        "title": title,
        "group_id": group_id,
        "customer_id": zammad_user_id,
        "article": {
            "subject": title,
            "body": description,
            "type": "note",
            "internal": False
        }
    }

    try:
        new_ticket = client.ticket.create(params=params)
        logging.info(f"Ticket created in Zammad with ID: {new_ticket['id']}")
    except Exception as e:
        logging.error(f"Failed to create the ticket in Zammad: {e}")
        raise ValueError(f"Zammad API error while creating ticket: {e}")

    try:
        cursor.execute(
            "INSERT INTO incidents (user_id, description, external_id) VALUES (%s, %s, %s)",
            (customer_id, description, new_ticket['id'])
        )
        conn.commit()
        logging.info(f"Ticket stored in database with external ID: {new_ticket['id']}")
    except Exception as e:
        logging.error(f"Failed to store the ticket in the database: {e}")
        conn.rollback()
        raise ValueError(f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()

    return new_ticket


# Add background image and logo
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://www.example.com/path-to-background.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: transparent;
}
button {
    background-color: green; /* Default button color */
    color: white; /* Text color */
    border: none; /* Remove border */
    border-radius: 5px; /* Rounded corners */
    padding: 10px 20px; /* Button padding */
    font-size: 16px; /* Font size */
    cursor: pointer; /* Change cursor to pointer */
}
button:hover {
    background-color: lightgreen; /* Button color on hover */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
# Add HPE Logo
logo_path = "venv/images/hpe_logo.png"  # Ensure this path is correct
st.sidebar.image(logo_path, use_column_width=True)  # Display the logo
 
# Add Truck Image
truck_image_path = "venv/images/truck_image.png"  # Ensure this path is correct
st.sidebar.image(truck_image_path, use_column_width=True)  # Display the truck image
 
# App Title
st.title("Truck Driver Smart Assistant")

# Function to fetch the Zammad customer_id using the phone from the database
def get_zammad_customer_id():
    client = ZammadAPI(url='https://anissaidi13.zammad.com/api/v1/', username='moansa55@gmail.com', password='Internship123')

    # Get the logged-in user's phone from the database
    username = st.session_state.get('username')
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("SELECT phone FROM users WHERE username = %s", (username,))
    user_info = cursor.fetchone()

    if not user_info:
        raise ValueError("Logged-in user's information could not be found in the database.")

    phone = user_info[0]  # Assuming phone is unique

    # Use paginated search to find the Zammad user based on the phone number
    zammad_user_id = None
    page = 1  # Start with page 1
    try:
        normalized_phone = phone.strip().replace(" ", "")
        logging.info(f"Normalized phone from database: {normalized_phone}")

        while True:
            users = client.user.all(page=page)  # Fetch users page by page
            if not users:
                break  # If no users are returned, all pages have been fetched

            for user in users:
                zammad_phone = user.get('phone', '').strip().replace(" ", "")
                logging.info(f"Checking Zammad user ID {user['id']} with phone: {zammad_phone}")

                if zammad_phone == normalized_phone:
                    zammad_user_id = user['id']
                    logging.info(f"Matching Zammad user found with ID: {zammad_user_id}")
                    break

            if zammad_user_id:
                break  # Exit the loop if a matching user is found
            page += 1  # Move to the next page if the user is not found on the current page
    except Exception as e:
        logging.error(f"Failed to fetch Zammad user by phone number: {e}")
        st.error(f"Failed to retrieve Zammad user: {e}")
        return None

    if zammad_user_id:
        return zammad_user_id
    else:
        logging.error("No matching Zammad user found for the phone number.")
        return None

# Function to fetch all tickets created by the logged-in user from Zammad
def get_user_tickets_from_zammad():
    zammad_api_url = 'https://anissaidi13.zammad.com/api/v1/tickets'
    zammad_api_token = 'xy2Y88Daf6hHf5iHeRQa6Oc6wneDsg-OTgyq26iYl-9rD4vRJUr27siE_juxET6l'

    headers = {
        'Authorization': f'Token token={zammad_api_token}',
        'Content-Type': 'application/json'
    }

    # Get the Zammad customer ID based on the logged-in user's phone
    zammad_customer_id = get_zammad_customer_id()

    if not zammad_customer_id:
        st.error("Unable to retrieve Zammad customer ID.")
        return []

    try:
        # Fetch all tickets from Zammad using the API
        response = requests.get(zammad_api_url, headers=headers)

        # Check if the API call was successful
        if response.status_code != 200:
            logging.error(f"Failed to fetch tickets from Zammad. Status Code: {response.status_code}")
            st.error(f"Failed to retrieve tickets: {response.text}")
            return []

        # Convert the response to JSON
        tickets = response.json()

        # Ensure that the response is a list
        if not isinstance(tickets, list):
            logging.error(f"Unexpected response format: {tickets}")
            st.error(f"Failed to retrieve tickets: unexpected response format.")
            return []

        logging.info(f"Fetched {len(tickets)} tickets from Zammad.")

        # Filter tickets by `customer_id`
        user_tickets = [ticket for ticket in tickets if ticket['customer_id'] == zammad_customer_id]

        logging.info(f"Found {len(user_tickets)} tickets for user with Zammad customer ID: {zammad_customer_id}")

        return user_tickets

    except Exception as e:
        logging.error(f"Failed to fetch the user's tickets from Zammad: {e}")
        st.error(f"Failed to retrieve tickets: {e}")
        return []


# Mapping of state_id to status names for categorized view
state_mapping = {
    1: "ongoing",  # Ongoing
    2: "escalated",  # Escalated
   # 3: "pending reminder",
    4: "completed",  # Completed
    #6: "pending close",
}

# Define colors for each status
status_colors = {
    "ongoing": "#FFDD00",  # Ongoing - Yellow
    "escalated": "#FF0000",  # Escalated - Red
    "completed": "#00E5EE",  # Completed - Cyan
}

# Function to display tickets in a categorized format
def display_tickets_ui():
    st.title("Ticket Status")
    tickets = get_user_tickets_from_zammad()

    # Handle cases where state_id is not in state_mapping
    ongoing_tickets = [ticket for ticket in tickets if state_mapping.get(ticket.get('state_id')) == "ongoing"]
    escalated_tickets = [ticket for ticket in tickets if state_mapping.get(ticket.get('state_id')) == "escalated"]
    completed_tickets = [ticket for ticket in tickets if state_mapping.get(ticket.get('state_id')) == "completed"]

    # Layout: Create three columns for Ongoing, Escalated, and Completed
    col1, col2, col3 = st.columns(3)

    # Ongoing Tickets
    with col1:
        st.markdown("<h2 style='text-align: center; color: #FFDD00;'>Ongoing</h2>", unsafe_allow_html=True)
        for ticket in ongoing_tickets:
            display_ticket(ticket)

    # Escalated Tickets
    with col2:
        st.markdown("<h2 style='text-align: center; color: #FF0000;'>Escalated</h2>", unsafe_allow_html=True)
        for ticket in escalated_tickets:
            display_ticket(ticket)

    # Completed Tickets
    with col3:
        st.markdown("<h2 style='text-align: center; color: #00E5EE;'>Completed</h2>", unsafe_allow_html=True)
        for ticket in completed_tickets:
            display_ticket(ticket)

# Function to display a ticket card
def display_ticket(ticket):
    state_name = state_mapping.get(ticket['state_id'], "Unknown Status")
    color = status_colors.get(state_name, "#FFFFFF")  # Default to white if status not found

    st.markdown(f"""
    <div style="background-color:{color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <strong>Ticket #{ticket['id']}:</strong> {ticket['title']}<br>
        <strong>Status:</strong> {state_name}<br>
        <strong>Created At:</strong> {ticket['created_at']}
    </div>
    """, unsafe_allow_html=True)

# Function to submit an issue
def submit_issue():
    st.title("Submit an Issue")

    user_input = st.text_input("What is the issue?", key="issue_input")

    if st.button("Submit Issue", key="submit_issue_button"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Process the input to create a ticket
            ticket_data, user_message = create_ticket_from_input(user_input)

            if ticket_data:
                new_ticket = create_ticket_in_zammad(ticket_data)
                st.success(f"Ticket created! Ticket ID: {new_ticket['id']}")
           # else:
            #    st.error("Unable to process the ticket data. Please try again.")

            st.session_state.messages.append({"role": "assistant", "content": user_message})

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**👨‍💼 User:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**👩‍💻 Assistant:** {message['content']}")

# Main function to display options after login
def logged_in_interface():
    selected_tab = option_menu(
        menu_title=None,
        options=["Submit Issue", "View Tickets"],
        icons=["plus", "clipboard"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        key="logged_in_menu",  # Unique key for the logged-in menu
        styles={
            "container": {"padding": "5!important", "background-color": "#f0f2f5"},
            "icon": {"font-size": "20px"},
            "nav-link": {"font-size": "18px", "color": "black"},
            "nav-link-selected": {"background-color": "green", "color": "white"},
            "nav-link-hover": {"background-color": "lightgreen", "color": "black"},
        }
    )

    if selected_tab == "Submit Issue":
        submit_issue()
    elif selected_tab == "View Tickets":
        display_tickets_ui()

# Conditional rendering based on whether the user is logged in
if not st.session_state['logged_in']:
    selected_tab = option_menu(
        menu_title=None,
        options=["Login", "Sign Up"],
        icons=["box-arrow-in-right", "person-plus"],
        menu_icon="cast",
        default_index=0,
        key="login_menu",  # Unique key for the login menu
        orientation="horizontal",
        styles={
            "container": {"padding": "5!important", "background-color": "#f0f2f5"},
            "icon": {"font-size": "20px"},
            "nav-link": {"font-size": "18px", "color": "black"},
            "nav-link-selected": {"background-color": "green", "color": "white"},
            "nav-link-hover": {"background-color": "lightgreen", "color": "black"},
        }
    )

    if selected_tab == "Login":
        st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
        username = st.text_input("User ID / Email Address*", placeholder="your.email@company.com OR user123", key="login_username")
        password = st.text_input("Password*", type="password", placeholder="Enter your password", key="login_password")

        if st.button("Login", key="login_button"):
            user = login_user(username, password)
            if user:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid username or password.")

    elif selected_tab == "Sign Up":
        st.markdown("<h2 style='text-align: center;'>Register New Account</h2>", unsafe_allow_html=True)
        new_username = st.text_input("User ID*", placeholder="user123", key="register_username")
        new_password = st.text_input("Password*", type="password", placeholder="Enter your password", key="register_password")
        full_name = st.text_input("Full Name*", placeholder="Enter your full name", key="register_fullname")
        phone_number = st.text_input("Phone Number*", placeholder="Enter your phone number")  # Added phone number field
        role = st.selectbox("Role", ["Driver", "Admin", "Support"], key="register_role")

        if st.button("Sign Up", key="signup_button"):
            if register_user(new_username, new_password, full_name,phone_number, role):
                st.success("Registration successful!")
            else:
                st.error("Registration failed! Username might already exist.")
else:
    logged_in_interface()

# Sidebar for logout and user information
def sidebar():
    with st.sidebar:
        st.header("Navigation")
        if st.session_state.get('logged_in'):
            st.write(f"Logged in as {st.session_state['username']}")
            if st.button("Logout", key="logout_button"):
                st.session_state['logged_in'] = False
                st.session_state['messages'] = []
                st.success("Logged out successfully!")
        else:
            st.write("Please login to use the system")

# Main function to run the app
def main():
    sidebar()
    #if st.session_state['logged_in']:
     #   logged_in_interface()

if __name__ == "__main__":
    main()

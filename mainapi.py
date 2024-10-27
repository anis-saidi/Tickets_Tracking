from fastapi import FastAPI, Depends, HTTPException, status,Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from zammad_py import ZammadAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from db_functions import register_user,get_user,connect_to_db
from datetime import datetime, timedelta
import requests
import logging
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
import httpx

# --- JWT Configuration ---
SECRET_KEY = "eW3Hg8N4qB3JkdF8Tk1fZc12wK6kP1Rt"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- PostgreSQL Database Configuration ---
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:09846690@localhost/poc_order"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- JWT Auth ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- FastAPI Application ---
app = FastAPI()



# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    #hashed_password = Column(String)  # This is the hashed password field
    full_name = Column(String)
    phone = Column(String)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Pydantic Schemas ---
class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str
    phone: str
    role: str

from pydantic import BaseModel

class TicketCreate(BaseModel):
    title: str
    description: str
    group_id: int = None  # Optional
    customer_id: int      # Required

# Define the model to represent user input and response
class UserInput(BaseModel):
    user_message: str
# Define response model for the created ticket
class TicketResponse(BaseModel):
    ticket_id: int
    message: str

# --- Utility Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.password):  # Change to user.password
        return False
    return user


def get_current_user(db=Depends(SessionLocal), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = get_user_by_username(db, username)
        if user is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return user

# --- API Endpoints ---

# Register User
@app.post("/register", status_code=201)
def register_user(user: UserCreate, db: SessionLocal = Depends(get_db)):
    print("Received user data:", user)
    # Check for existing user
    db_user = get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        password=get_password_hash(user.password),
        full_name=user.full_name,
        phone=user.phone,
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

# Login User
@app.post("/login")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    return {"access_token": access_token, "token_type": "bearer"}

# Connect to the database
def connect_to_db():
    conn = psycopg2.connect(
        host="localhost",
        database="poc_order",
        user="postgres",
        password="hpcai123"
    )
    return conn

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

# Function to handle user input and create a ticket if necessary
# Function to handle user input and create a ticket if necessary
async def create_ticket_from_input(user_input, customer_id):
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
        
        # Log the full response for debugging purposes
        logging.info(f"Model response: {response}")
        
        # Check for tool calls, which indicate recognized issues
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            logging.info("No tool calls returned, responding with a general message.")
            return {}, "Sorry, I'm here to help you solve issues and create tickets. How can I assist you today?"
        
        # Proceed to extract tool calls for ticket creation
        tool_calls = response.tool_calls
        ticket_data = {}
        user_message = ""

        for tool_call in tool_calls:
            if tool_call['name'] == 'create_ticket':
                logging.info("Model recognized an issue, extracting ticket data.")
                ticket_data = tool_call['args']
                user_message = ticket_data.pop('user_message', '')

        # Ensure ticket data has the required fields (title and description)
        if 'title' not in ticket_data or 'description' not in ticket_data:
            logging.error("Ticket data is missing 'title' or 'description'.")
            return {}, "I couldn't create a ticket because the necessary details were not provided. Please describe the issue more clearly."

        # Log the ticket data before proceeding
        logging.info(f"Ticket data extracted from model: {ticket_data}")

        # Add customer_id to ticket data
        ticket_data['customer_id'] = customer_id

        # Create ticket in Zammad
        new_ticket = create_ticket_in_zammad(ticket_data)

        return new_ticket, user_message

    except KeyError as e:
        logging.error(f"KeyError: {e} - Please check your prompt formatting.")
        return {}, "There was an error processing your request."
    except Exception as e:
        logging.error(f"Unexpected error invoking model: {e}")
        return {}, "There was an issue processing your request."

# Function to create a ticket in Zammad
def create_ticket_in_zammad(ticket_data):
    client = ZammadAPI(url='https://anissaidi13.zammad.com/api/v1/', username='moansa55@gmail.com', password='Internship123')
 
    # Log the ticket data before sending
    logging.info(f"Ticket data to send to Zammad: {ticket_data}")
 
    # Ensure 'title' and 'description' are present
    if 'title' not in ticket_data or 'description' not in ticket_data:
        logging.error("Missing required fields: 'title' and 'description'")
        raise ValueError("Missing required fields: 'title' and 'description'")
 
    try:
        # Create ticket with required fields
        new_ticket = client.ticket.create({
            "title": ticket_data['title'],
            "group_id": ticket_data.get('group_id', 1),   # Default to group_id 1 if not provided
            "customer_id": ticket_data['customer_id'],  # Customer ID linked to the requester
            "article": {
                "subject": ticket_data['title'],
                "body": ticket_data['description'],
                "type": "note",  # You can change this to email or other type
                "internal": False  # This is a public ticket, change to True if it's an internal note
            },
            "priority_id": 2  # Default to normal priority (adjust as needed)
        })
       
        logging.info(f"Ticket created in Zammad with ID: {new_ticket['id']}")
        logging.info(f"Full Zammad response: {new_ticket}")  # Log the full response for debugging
        return new_ticket
 
    except Exception as e:
        logging.error(f"Failed to create ticket in Zammad: {e}")
        raise ValueError(f"Zammad API error while creating ticket: {e}")
    
@app.post("/submit-issue/{customer_id}")
async def submit_issue(customer_id: int, request: Request):
    try:
        # Parse the incoming JSON request body
        body = await request.json()
        user_input = body.get('issue')  # Expecting an 'issue' field in the JSON
 
        if not user_input:
            raise HTTPException(status_code=400, detail="Missing required field: issue")
 
        # Process the user input and create the ticket
        ticket_data, user_message = await create_ticket_from_input(user_input, customer_id)

        # Check if ticket_data is empty and respond accordingly
        if not ticket_data:
            return {"message": "Sorry, I'm here to help you solve issues and create tickets. How can I assist you today?"}
        
        return {"message": "Ticket created successfully!", "ticket_data": ticket_data, "user_message": user_message}
 
    except Exception as e:
        logging.error(f"Error in submit_issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Zammad API configuration
ZAMMAD_URL = 'https://anissaidi13.zammad.com/api/v1/tickets'  # Endpoint to get tickets
API_TOKEN = 'xy2Y88Daf6hHf5iHeRQa6Oc6wneDsg-OTgyq26iYl-9rD4vRJUr27siE_juxET6l'  # Use your actual Bearer token

class Ticket(BaseModel):
    id: int
    title: str
    description: Optional[str] = None  # Make description optional
    status: Optional[str] = None        # Make status optional
    customer_id: int
    created_at: str
    updated_at: str

class TicketResponse(BaseModel):
    tickets: List[Ticket]

@app.get("/api/tickets/{customer_id}", response_model=TicketResponse)
async def get_tickets(customer_id: int):
    headers = {
        'Authorization': f'Token token={API_TOKEN}',  # Use Bearer token
        'Content-Type': 'application/json'
    }
    
    # Fetch tickets from Zammad
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ZAMMAD_URL}?customer_id={customer_id}", headers=headers)  # Use GET with query parameters
        
    if response.status_code != 200:
        logging.error(f"HTTP error occurred: {response.text}")  # Log error
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve tickets from Zammad")

    tickets_data = response.json()

    # Filter tickets by customer_id if needed
    customer_tickets = [
        Ticket(**ticket) for ticket in tickets_data if ticket['customer_id'] == customer_id
    ]  # Adjust based on actual Zammad response structure

    if not customer_tickets:
        raise HTTPException(status_code=404, detail="No tickets found for this customer ID")

    return {"tickets": customer_tickets}



# Get All Tickets (Admin) 
@app.get("/tickets/all")
def get_all_tickets(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    zammad_url = "https://anissaidi13.zammad.com/api/v1/tickets"
    headers = {
        "Authorization": "xy2Y88Daf6hHf5iHeRQa6Oc6wneDsg-OTgyq26iYl-9rD4vRJUr27siE_juxET6l",
        "Content-Type": "application/json"
    }
    response = requests.get(zammad_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch tickets")
 
# Update Ticket Status
@app.put("/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: int, status: str, current_user: User = Depends(get_current_user)):
    zammad_url = f"https://anissaidi13.zammad.com/api/v1/tickets/{ticket_id}"
    headers = {
        "Authorization": "vogycCLFaFlBgoMYz8X1sotEJvVgePxXHd2ZJVwS5HaCxjabsIligx3SJz4590or",
        "Content-Type": "application/json"
    }
    ticket_update = {
        "state": status  # e.g., "closed", "open", etc.
    }
    response = requests.put(zammad_url, json=ticket_update, headers=headers)
    if response.status_code == 200:
        return {"message": "Ticket updated successfully"}
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to update ticket")
 
# Delete Ticket
@app.delete("/tickets/{ticket_id}")
def delete_ticket(ticket_id: int, current_user: User = Depends(get_current_user)):
    zammad_url = f"https://anissaidi.zammad.com/api/v1/tickets/{ticket_id}"
    headers = {
        "Authorization": "vogycCLFaFlBgoMYz8X1sotEJvVgePxXHd2ZJVwS5HaCxjabsIligx3SJz4590or",
        "Content-Type": "application/json"
    }
    response = requests.delete(zammad_url, headers=headers)
    if response.status_code == 204:
        return {"message": "Ticket deleted successfully"}
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to delete ticket")
 
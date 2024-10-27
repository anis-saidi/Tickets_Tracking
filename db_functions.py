import psycopg2
from psycopg2 import sql
import bcrypt
from zammad_py import ZammadAPI

def connect_to_db():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
    dbname="poc_order",
    user="postgres",
    password="hpcai123",
    host="localhost",
    port="5432"
    )
   
   
    return conn

# Function to retrieve user by username
def get_user(username):
    conn = connect_to_db()
    cursor = conn.cursor()
    query = sql.SQL("SELECT username, password, role FROM users WHERE username = %s")
    cursor.execute(query, (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to register a new user
def register_user(username, password, full_name, phone, role='driver'):
    conn = connect_to_db()
    cursor = conn.cursor()
    role = role.lower()

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        cursor.execute(
            "INSERT INTO users (username, password, full_name, phone, role) VALUES (%s, %s, %s, %s, %s)",
            (username, hashed_password, full_name, phone, role)
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Function to check password
def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,  -- Store hashed passwords
    full_name VARCHAR(100) NOT NULL,
    phone VARCHAR(15) UNIQUE NOT NULL,  -- Include phone number
    role VARCHAR(20) NOT NULL CHECK (role IN ('driver', 'admin', 'support')),  -- User roles
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE incidents (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    description TEXT NOT NULL,
    external_id VARCHAR(100),
    platform VARCHAR(50),
    status VARCHAR(50) DEFAULT 'submitted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE  -- Reference to user_id
);

-- Insert an initial user into the users table
INSERT INTO users (username, password, full_name, role, phone)
VALUES ('admin_user', 'hashed_password_here', 'Admin Name', 'admin', '+21612345678');

-- Create Tickets Table
CREATE TABLE tickets (
    ticket_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    customer_id INT NOT NULL,
    group_id INT NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (customer_id) REFERENCES users(user_id) ON DELETE CASCADE
);
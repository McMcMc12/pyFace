import sqlite3

# Create or open a SQLite database file
conn = sqlite3.connect('users.db')

# Create a cursor object using the cursor() method
c = conn.cursor()

# Create table with the necessary schema
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, encoding BLOB)''')

# Commit the changes
conn.commit()

# Close the connection
conn.close()

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    try:
        # Connect using environment variables
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host="localhost", # Change to "db" if running INSIDE Docker
            port=os.getenv('DB_PORT')
        )
        cur = conn.cursor()
        
        # Price History Table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10),
                timestamp TIMESTAMP,
                close_price NUMERIC
            )
        ''')
        
        # Sentiment & Signal Logs Table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                asset VARCHAR(10),
                sentiment_label VARCHAR(20),
                sentiment_score NUMERIC,
                reasoning TEXT
            )
        ''')
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ SUCCESS: PostgreSQL tables initialized.")
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")

if __name__ == "__main__":
    init_db()
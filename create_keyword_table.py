#!/usr/bin/env python3
"""Create keyword_inverted_index table."""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Use connection from environment
conn_str = os.getenv('DATABASE_URL')

try:
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    
    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS keyword_inverted_index (
            id SERIAL PRIMARY KEY,
            term VARCHAR(255) NOT NULL,
            chunk_id VARCHAR(255) NOT NULL,
            score FLOAT NOT NULL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create index
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_keyword_inverted_index_term 
        ON keyword_inverted_index(term)
    """)
    
    conn.commit()
    print("✅ Created keyword_inverted_index table and index")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")

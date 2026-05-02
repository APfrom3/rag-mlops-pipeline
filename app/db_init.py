import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def initialize_db():
    conn = get_connection()
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create table to store document chunks and their embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            filing_id TEXT,
            company TEXT,
            chunk_text TEXT,
            embedding vector(1024),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Create an index for fast similarity search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS embedding_idx
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    initialize_db()
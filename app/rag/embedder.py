import os
import json
import tiktoken
from anthropic import Anthropic
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from app.db_init import get_connection

load_dotenv()

client = Anthropic()

# Tiktoken is OpenAI's tokenizer — we use it here just for counting
# tokens accurately since most embedding models share similar tokenization
tokenizer = tiktoken.get_encoding("cl100k_base")

# How many tokens per chunk and how many tokens overlap between chunks
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def chunk_text(text: str, ticker: str, company: str, year: int, quarter: int) -> list[dict]:
    """
    Splits a transcript into overlapping chunks of roughly CHUNK_SIZE tokens.

    Why chunk at all? Embedding models have token limits and work better on
    focused passages than entire documents. Smaller chunks also mean more
    precise retrieval — we return exactly the passage that answers the question
    rather than the entire transcript.

    Why overlap? If a key sentence falls at the boundary between two chunks,
    overlap ensures it appears fully in at least one of them. Without overlap
    you'd silently lose context at every boundary.

    Alternative: Semantic chunking splits on meaning rather than token count.
    More accurate but slower and more complex to implement. Token-based
    chunking is the industry standard starting point.
    """
    words = text.split()
    chunks = []
    current_words = []
    current_tokens = 0
    chunk_index = 0

    for word in words:
        word_tokens = count_tokens(word)

        if current_tokens + word_tokens > CHUNK_SIZE and current_words:
            chunk_text = " ".join(current_words)
            chunks.append({
                "ticker": ticker,
                "company": company,
                "year": year,
                "quarter": quarter,
                "chunk_index": chunk_index,
                "text": chunk_text
            })
            chunk_index += 1

            # Keep the last CHUNK_OVERLAP tokens worth of words for the next chunk
            overlap_words = current_words[-CHUNK_OVERLAP:]
            current_words = overlap_words + [word]
            current_tokens = count_tokens(" ".join(current_words))
        else:
            current_words.append(word)
            current_tokens += word_tokens

    # Don't forget the last chunk
    if current_words:
        chunks.append({
            "ticker": ticker,
            "company": company,
            "year": year,
            "quarter": quarter,
            "chunk_index": chunk_index,
            "text": " ".join(current_words)
        })

    return chunks


def embed_text(text: str) -> list[float]:
    """
    Converts a text string into a vector embedding using Voyage's
    finance-tuned model via the voyageai client.

    Why voyage-finance-2 specifically? It's domain-tuned on financial
    text so it understands terminology like 'operating margin', 'guidance',
    'YoY growth' better than a general purpose embedding model would.
    """
    import voyageai
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = vo.embed([text], model="voyage-finance-2")
    return result.embeddings[0]


def embed_and_store(transcript_dir: str = "data/transcripts"):
    """
    Main pipeline: loads transcripts, chunks them, embeds each chunk,
    and stores everything in pgvector.
    """
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()

    for filename in os.listdir(transcript_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(transcript_dir, filename)
        with open(filepath) as f:
            data = json.load(f)

        ticker = data["ticker"]
        print(f"Processing {ticker}...")

        chunks = chunk_text(
            text=data["text"],
            ticker=ticker,
            company=data.get("company", ticker),
            year=data["year"],
            quarter=data["quarter"]
        )

        print(f"  {len(chunks)} chunks created")

        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk["text"])

            cur.execute("""
                INSERT INTO document_chunks
                    (filing_id, company, chunk_text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                f"{ticker}_Q{chunk['quarter']}_{chunk['year']}",
                chunk["company"],
                chunk["text"],
                embedding,
                json.dumps({
                    "ticker": ticker,
                    "year": chunk["year"],
                    "quarter": chunk["quarter"],
                    "chunk_index": chunk["chunk_index"]
                })
            ))

            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(chunks)} chunks")

        conn.commit()
        print(f"  Done with {ticker}")

    cur.close()
    conn.close()
    print("All transcripts embedded and stored.")


if __name__ == "__main__":
    embed_and_store()
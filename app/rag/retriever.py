import os
import json
import voyageai
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from app.db_init import get_connection

load_dotenv()

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


def embed_query(query: str) -> list[float]:
    """
    Embeds the user's question using the same model we used to embed
    the chunks. This is critical — query and chunks must be embedded
    with the same model so their vectors live in the same space and
    cosine similarity is meaningful.

    Alternative: Some systems use different models for queries vs documents
    (asymmetric embedding). Voyage supports this with input_type parameter.
    For our use case symmetric embedding is fine.
    """
    result = vo.embed(
        [query],
        model="voyage-finance-2",
        input_type="query"
    )
    return result.embeddings[0]


def retrieve_chunks(query: str, top_k: int = 5, company: str = None) -> list[dict]:
    """
    Takes a natural language query, embeds it, and finds the most
    semantically similar chunks in pgvector using cosine similarity.

    Why cosine similarity over euclidean distance? Cosine similarity
    measures the angle between vectors rather than their magnitude.
    For text embeddings this is more meaningful — two passages can
    be about the same topic regardless of their length, and cosine
    similarity captures that while euclidean distance would penalize
    length differences.

    Why top_k=5? Gives Claude enough context to synthesize a good
    answer without blowing up the context window. In production this
    would be tunable based on query complexity and latency requirements.

    The optional company filter lets users scope queries to a specific
    ticker — important when someone asks about a specific company rather
    than across all transcripts.
    """
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()

    query_embedding = embed_query(query)

    if company:
        cur.execute("""
            SELECT
                chunk_text,
                company,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE LOWER(company) = LOWER(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, company, query_embedding, top_k))
    else:
        cur.execute("""
            SELECT
                chunk_text,
                company,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        chunk_text, company_name, metadata, similarity = row
        results.append({
            "text": chunk_text,
            "company": company_name,
            "metadata": metadata,
            "similarity": round(float(similarity), 4)
        })

    return results


if __name__ == "__main__":
    # Quick test to verify retrieval is working
    query = "What did management say about revenue growth?"
    print(f"Query: {query}\n")

    results = retrieve_chunks(query, top_k=3)
    for i, r in enumerate(results):
        print(f"Result {i+1} | {r['company']} | similarity: {r['similarity']}")
        print(f"{r['text'][:300]}...")
        print()
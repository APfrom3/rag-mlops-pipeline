import json
from datetime import datetime
from app.db_init import get_connection


def create_eval_table():
    """
    Creates the eval_logs table if it doesn't exist.
    We keep this separate from document_chunks since it serves
    a completely different purpose — operational monitoring vs
    knowledge storage.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_logs (
            id SERIAL PRIMARY KEY,
            query TEXT,
            answer TEXT,
            relevance_score FLOAT,
            faithfulness_score FLOAT,
            answer_quality_score FLOAT,
            composite_score FLOAT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            company_filter TEXT,
            top_k INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


def log_eval(
    query: str,
    answer: str,
    scores: dict,
    usage: dict,
    company_filter: str = None,
    top_k: int = 5
):
    """
    Persists evaluation scores and token usage for every query to
    the database.

    Why log every query? This builds a dataset over time that lets
    you spot degradation — if faithfulness scores start dropping
    after you change your prompt or chunking strategy, you'll see
    it immediately. This is the core of LLMOps monitoring.

    In production you'd build a dashboard on top of this table
    (Grafana, Metabase, or a custom React dashboard) so the team
    can monitor quality in real time.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO eval_logs (
            query,
            answer,
            relevance_score,
            faithfulness_score,
            answer_quality_score,
            composite_score,
            input_tokens,
            output_tokens,
            company_filter,
            top_k
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        query,
        answer,
        scores["relevance"],
        scores["faithfulness"],
        scores["answer_quality"],
        scores["composite_score"],
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        company_filter,
        top_k
    ))

    conn.commit()
    cur.close()
    conn.close()


def get_eval_summary():
    """
    Returns aggregate metrics across all logged queries.
    Useful for understanding overall pipeline health.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) as total_queries,
            ROUND(AVG(relevance_score)::numeric, 4) as avg_relevance,
            ROUND(AVG(faithfulness_score)::numeric, 4) as avg_faithfulness,
            ROUND(AVG(answer_quality_score)::numeric, 4) as avg_quality,
            ROUND(AVG(composite_score)::numeric, 4) as avg_composite,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens
        FROM eval_logs;
    """)

    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or row[0] == 0:
        return {"message": "No eval logs yet"}

    return {
        "total_queries": row[0],
        "avg_relevance": float(row[1]),
        "avg_faithfulness": float(row[2]),
        "avg_quality": float(row[3]),
        "avg_composite": float(row[4]),
        "total_input_tokens": row[5],
        "total_output_tokens": row[6]
    }
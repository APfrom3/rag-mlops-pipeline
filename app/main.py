from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from app.claude_client import answer_question
from app.eval.logger import create_eval_table, get_eval_summary
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Earnings Call RAG API",
    description="Query earnings call transcripts using natural language powered by Claude and pgvector",
    version="1.0.0"
)

# Create eval table on startup if it doesn't exist
@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts. We use it to ensure the eval
    table exists before any requests come in.

    Why here instead of in db_init.py? Startup events are the FastAPI
    pattern for initialization logic — it keeps the API self-contained
    and ensures the table exists regardless of whether db_init.py was
    run manually beforehand. Important for the Docker/Cloud Run deployment
    where we can't assume anything was run manually.
    """
    create_eval_table()


# --- Request and Response Models ---
# Pydantic models define the shape of our API's inputs and outputs.
# FastAPI uses these to automatically validate requests, return clear
# error messages for bad input, and generate API documentation.
# Alternative: you could use plain dicts but you'd lose validation,
# docs generation, and type safety.

class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    top_k: Optional[int] = 5

class SourceModel(BaseModel):
    text: str
    company: str
    similarity: float
    metadata: dict

class EvalScores(BaseModel):
    relevance: float
    faithfulness: float
    answer_quality: float
    composite_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceModel]
    eval_scores: EvalScores
    usage: dict
    model: str


# --- Endpoints ---

@app.get("/")
def root():
    """Health check endpoint. Standard practice for any deployed service
    so load balancers and monitoring tools can verify the service is up."""
    return {"status": "ok", "service": "Earnings Call RAG API"}


@app.post("/query", response_model=QueryResponse)
async def query_transcripts(request: QueryRequest):
    """
    Main endpoint. Takes a natural language question and optional company
    filter, runs the full RAG pipeline, and returns an answer with sources
    and eval scores.

    Why POST instead of GET? We're sending a request body with potentially
    long query text. GET requests put data in the URL which has length limits
    and gets logged in server access logs — bad for potentially sensitive
    financial queries. POST keeps the payload in the request body.

    Alternative: GraphQL would give clients more flexibility to request
    exactly the fields they need. REST is simpler and more universally
    understood, which is right for a first version.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    try:
        result = answer_question(
            query=request.query,
            company=request.company,
            top_k=request.top_k
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies")
def list_companies():
    """
    Returns the list of companies available in the database.
    Useful for frontend dropdowns or client discovery.
    """
    from app.db_init import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT company FROM document_chunks ORDER BY company;")
    companies = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return {"companies": companies}


@app.get("/eval/summary")
def eval_summary():
    """
    Returns aggregate eval metrics across all logged queries.
    This is your MLOps monitoring endpoint — in production you'd
    hook this up to a dashboard like Grafana or Metabase to track
    pipeline health over time.
    """
    return get_eval_summary()
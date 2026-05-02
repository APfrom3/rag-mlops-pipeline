import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

def score_relevance(query: str, chunks: list[dict]) -> float:
    """
    Measures whether the retrieved chunks are actually relevant to the query.
    This is called 'retrieval precision' in MLOps literature.

    Why use Claude to score this? Rule-based metrics like keyword matching
    miss semantic relevance. Claude can judge whether a passage genuinely
    addresses a question the same way a human evaluator would.
    """
    context = "\n\n".join([
        f"[Chunk {i+1}]: {c['text'][:500]}"
        for i, c in enumerate(chunks)
    ])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate how relevant these retrieved passages are to the query on a scale of 0.0 to 1.0.

Query: {query}

Retrieved passages:
{context}

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
        }]
    )

    try:
        score = float(response.content[0].text.strip())
        return round(min(max(score, 0.0), 1.0), 4)
    except ValueError:
        return 0.0


def score_faithfulness(query: str, answer: str, chunks: list[dict]) -> float:
    """
    Measures whether Claude's answer is grounded in the retrieved context
    or whether it hallucinated information not present in the chunks.
    This is the most critical metric for financial RAG systems — an answer
    that sounds confident but isn't in the source material is dangerous.

    Why is this especially important for finance? If an analyst makes a
    decision based on a hallucinated revenue figure, that's a real business
    risk. Faithfulness scoring lets you catch and flag these cases.

    Alternative: Some teams use natural language inference (NLI) models
    to check entailment between answer and context. Claude-as-judge is
    simpler to implement and often more accurate for complex reasoning.
    """
    context = "\n\n".join([
        f"[Source {i+1}]: {c['text'][:500]}"
        for i, c in enumerate(chunks)
    ])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate how faithfully this answer is grounded in the provided source passages on a scale of 0.0 to 1.0.
1.0 means every claim in the answer is directly supported by the sources.
0.0 means the answer contains significant information not found in the sources.

Query: {query}
Answer: {answer}

Source passages:
{context}

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
        }]
    )

    try:
        score = float(response.content[0].text.strip())
        return round(min(max(score, 0.0), 1.0), 4)
    except ValueError:
        return 0.0


def score_answer_quality(query: str, answer: str) -> float:
    """
    Measures the overall quality of the answer independent of the sources —
    is it well structured, complete, and actually addresses the question?

    Why three separate metrics instead of one overall score? Each metric
    diagnoses a different failure mode:
    - Low relevance = retrieval problem (wrong chunks being fetched)
    - Low faithfulness = hallucination problem (Claude going off-script)
    - Low answer quality = generation problem (Claude not synthesizing well)

    Separating them tells you exactly where to improve the pipeline.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate the quality of this answer to the given query on a scale of 0.0 to 1.0.
Consider: completeness, clarity, and whether it directly addresses the question.

Query: {query}
Answer: {answer}

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
        }]
    )

    try:
        score = float(response.content[0].text.strip())
        return round(min(max(score, 0.0), 1.0), 4)
    except ValueError:
        return 0.0


def evaluate(query: str, answer: str, chunks: list[dict]) -> dict:
    """
    Runs all three metrics and returns a combined evaluation report.
    This is what gets logged to the database after every query.
    """
    print("  Scoring relevance...")
    relevance = score_relevance(query, chunks)

    print("  Scoring faithfulness...")
    faithfulness = score_faithfulness(query, answer, chunks)

    print("  Scoring answer quality...")
    quality = score_answer_quality(query, answer)

    # Composite score weighted toward faithfulness since hallucination
    # is the most dangerous failure mode in a financial context
    composite = round(
        (relevance * 0.3) + (faithfulness * 0.5) + (quality * 0.2),
        4
    )

    return {
        "relevance": relevance,
        "faithfulness": faithfulness,
        "answer_quality": quality,
        "composite_score": composite
    }
import os
from app.eval.metrics import evaluate
from app.eval.logger import log_eval, create_eval_table
from anthropic import Anthropic
from dotenv import load_dotenv
from app.rag.retriever import retrieve_chunks

load_dotenv()

client = Anthropic()

SYSTEM_PROMPT = """You are a financial analyst assistant with access to earnings call transcripts 
from major technology companies. Your job is to answer questions about company performance, 
strategy, and outlook based strictly on the provided transcript excerpts.

Guidelines:
- Only use information from the provided context
- Always cite which company and quarter the information comes from
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and precise — you are speaking to sophisticated financial analysts
- Never speculate beyond what management explicitly stated"""


def build_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a readable context block for Claude.

    Why format it this way? Claude performs better when context is clearly
    structured with explicit source attribution. It makes it easier for the
    model to cite sources accurately and reduces hallucination risk since
    each passage is clearly labeled with where it came from.
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        metadata = chunk["metadata"]
        header = f"[Source {i+1}: {chunk['company']} Q{metadata['quarter']} {metadata['year']} | similarity: {chunk['similarity']}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def answer_question(
    query: str,
    company: str = None,
    top_k: int = 5,
    conversation_history: list = None
) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate an answer
    using Claude with those chunks as context.

    Why include conversation_history? This enables multi-turn conversations
    where the user can ask follow-up questions. Claude can refer back to
    previous context without us re-explaining everything each turn.

    Why return a dict instead of just the text? We want to expose the
    retrieved sources alongside the answer so the frontend can show
    citations. This is critical for financial use cases where analysts
    need to verify the source of every claim.

    Alternative: Some RAG systems re-rank chunks after retrieval using
    a cross-encoder model before sending to the LLM. This improves
    quality but adds latency. For my use case single-stage retrieval
    is sufficient.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(query, top_k=top_k, company=company)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in the transcripts.",
            "sources": [],
            "query": query
        }

    # Step 2: Build context from chunks
    context = build_context(chunks)

    # Step 3: Build messages array
    # We include conversation history for multi-turn support
    messages = conversation_history or []

    messages.append({
        "role": "user",
        "content": f"""Based on the following earnings call transcript excerpts, please answer this question:

Question: {query}

Context:
{context}"""
    })

    # Step 4: Call Claude
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    answer = response.content[0].text

    # Step 5: Append Claude's response to history for multi-turn support
    messages.append({
        "role": "assistant",
        "content": answer
    })

    # Step 6: Evaluate and log
    print("Evaluating response...")
    scores = evaluate(query, answer, chunks)
    log_eval(
        query=query,
        answer=answer,
        scores=scores,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        },
        company_filter=company,
        top_k=top_k
    )

    return {
        "answer": answer,
        "sources": chunks,
        "query": query,
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        },
        "eval_scores": scores
    }


if __name__ == "__main__":
    # Test the full pipeline end to end
    if __name__ == "__main__":
        create_eval_table()
    print("Testing RAG pipeline...\n")

    result = answer_question(
        "What did management say about AI investments and their expected returns?",
        top_k=5
    )

    print(f"Question: {result['query']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources used:")
    for s in result['sources']:
        print(f"  - {s['company']} (similarity: {s['similarity']})")
    print(f"\nTokens used: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")
    print(f"\nEval Scores:")
    print(f"  Relevance:       {result['eval_scores']['relevance']}")
    print(f"  Faithfulness:    {result['eval_scores']['faithfulness']}")
    print(f"  Answer Quality:  {result['eval_scores']['answer_quality']}")
    print(f"  Composite:       {result['eval_scores']['composite_score']}")
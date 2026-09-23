from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from src.config.config import config

# ---------------------------------------------------------------------------
# Module-level cache so the collection is opened only once per process
# ---------------------------------------------------------------------------
_collection = None


def get_collection():
    """Return the ChromaDB collection, opening it lazily and caching it."""
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=config["chroma_db_path"])
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=config["embedding_model_name"]
    )
    _collection = client.get_or_create_collection(
        name=config["rag_collection_name"],
        embedding_function=embed_fn,
    )
    return _collection


def retrieve(question: str, n_results: int = 3) -> list[dict]:
    """
    Query the vector store for the top-n most relevant drug documents.

    Returns:
        list of dicts with keys: document, metadata, distance
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count() or 1),
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"document": doc, "metadata": meta, "distance": dist})
    return hits


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Format retrieved docs into a RAG prompt."""
    context_blocks = []
    for i, hit in enumerate(retrieved, 1):
        drug_name = hit["metadata"].get("drug_name", "Unknown")
        context_blocks.append(f"[{i}] {drug_name}\n{hit['document']}")

    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a clinical drug-information assistant.\n"
        "Answer the question based on the clinical dataset context provided below.\n\n"
        "Guidelines:\n"
        "1. Summarize and explain the relevant clinical information from the context that answers the user's question (e.g. uses/indications, side effects, substitutes, therapeutic or chemical class, habit-forming status).\n"
        "2. If the context covers some parts of the question (for example, the side effects) but not others (for example, prevention or unrelated advice), answer the covered parts and state that the rest of the information is not in the dataset.\n"
        "3. If the context contains no relevant information to answer the question, respond with: \"I don't have that information in my dataset.\"\n"
        "4. Do not invent or hallucinate clinical information not supported by the context.\n\n"
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        "=== QUESTION ===\n"
        f"{question}"
    )
    return prompt


def rag_answer(question: str, n_results: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → LLM answer.

    Returns:
        {
            "answer": str,
            "sources": list[str],   # drug names from retrieved docs
        }
    """
    retrieved = retrieve(question, n_results=n_results)
    sources = [h["metadata"].get("drug_name", "Unknown") for h in retrieved]

    api_key = config.get("openai_api_key") or ""
    if not api_key.strip():
        # Retrieval-only mode — no LLM key configured
        formatted = "\n\n".join(
            f"**{h['metadata'].get('drug_name', 'Drug')}**\n{h['document']}"
            for h in retrieved
        )
        answer = (
            "⚠️ *Retrieval-only mode (no OpenAI key configured)*\n\n"
            "Here are the most relevant entries from the dataset:\n\n"
            f"{formatted}"
        )
        return {"answer": answer, "sources": sources}

    # Full LLM generation
    prompt = build_prompt(question, retrieved)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "sources": sources}

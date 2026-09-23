import os
import math
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config.config import config


def _resolve_csv_path(csv_path: str | None) -> str:
    """Resolve the CSV path, preferring sampled, falling back to cleaned."""
    if csv_path and os.path.exists(csv_path):
        return csv_path

    candidates = [
        "src/drugs_sampled.csv",
        "src/drugs_cleaned.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find 'src/drugs_sampled.csv' or 'src/drugs_cleaned.csv'. "
        "Run 'python -m src.main' first to generate the cleaned dataset, "
        "then 'python -m src.utils.sample_dataset' to produce the sampled file."
    )


def build_index(csv_path: str | None = None, force: bool = False) -> None:
    """
    Build (or skip if already built) a ChromaDB vector index from rag_doc column.

    Args:
        csv_path: Override the CSV source. Defaults to drugs_sampled.csv → drugs_cleaned.csv.
        force:    If True, drop and re-index even if the collection already has documents.
    """
    resolved_path = _resolve_csv_path(csv_path)
    print(f"[RAG] Loading dataset from '{resolved_path}'...")
    df = pd.read_csv(resolved_path, low_memory=False)

    # Drop rows where rag_doc is null — nothing to embed without it
    df = df.dropna(subset=["rag_doc"])
    # Use positional index as Chroma ID to guarantee uniqueness even when
    # the source CSV's 'id' column contains corrupted/duplicated values.
    df = df.reset_index(drop=True)
    print(f"[RAG] {len(df):,} documents available for indexing.")

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=config["chroma_db_path"])
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=config["embedding_model_name"]
    )

    if force:
        # Delete and recreate collection
        try:
            client.delete_collection(name=config["rag_collection_name"])
            print("[RAG] Existing collection deleted (force=True).")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config["rag_collection_name"],
        embedding_function=embed_fn,
    )

    if not force and collection.count() > 0:
        print(
            f"[RAG] Collection '{config['rag_collection_name']}' already has "
            f"{collection.count():,} documents. Skipping re-indexing. "
            "Pass force=True to rebuild."
        )
        return

    # Build metadata — Chroma requires str/int/float/bool values (no NaN)
    meta_cols = ["drug_name", "therapeutic_class", "habit_forming"]
    for col in meta_cols:
        if col not in df.columns:
            df[col] = "NA"
    df[meta_cols] = df[meta_cols].fillna("NA").astype(str)

    metadatas = df[meta_cols].to_dict(orient="records")
    documents = df["rag_doc"].tolist()
    ids = [str(i) for i in df.index]  # positional — always unique

    # Batch into chunks of 2000 (ChromaDB max batch size)
    batch_size = 2000
    total_batches = math.ceil(len(documents) / batch_size)
    print(f"[RAG] Indexing {len(documents):,} documents in {total_batches} batch(es)...")

    for i in range(total_batches):
        start = i * batch_size
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"[RAG]   Batch {i + 1}/{total_batches} done ({min(end, len(documents)):,} docs indexed).")

    print(f"[RAG] Index built successfully. Total: {collection.count():,} documents.")


if __name__ == "__main__":
    build_index()

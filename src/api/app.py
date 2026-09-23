import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build RAG index on startup so first query is never cold."""
    try:
        from src.rag.build_index import build_index
        build_index()
    except Exception as e:
        print(f"[RAG] Startup index build failed (non-fatal): {e}")
    yield


app = FastAPI(
    title="Drug Classification API",
    description="FastAPI service serving analytical charts and chatbot interactions for drug data",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset cache
_DATA_CACHE: Dict[str, Any] = {}


def get_data_path() -> str:
    """Finds the most processed CSV available."""
    candidates = [
        "src/drugs_cleaned.csv",
        "src/drugs_with_rag.csv",
        "src/drugs.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "src/drugs.csv"


def load_dataset() -> pd.DataFrame:
    """Loads and caches dataset in memory."""
    if "df" in _DATA_CACHE:
        return _DATA_CACHE["df"]

    path = get_data_path()
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found at '{path}'. Please run data generation first."
        )

    print(f"[API] Loading dataset from '{path}'...")
    df = pd.read_csv(path, low_memory=False)
    _DATA_CACHE["df"] = df
    return df


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    sources: List[str] = []


@app.get("/")
def root():
    return {"message": "Drug Classification API is online and healthy."}


@app.get("/api/stats")
def get_stats():
    """Returns general metrics and record counts."""
    df = load_dataset()
    total_records = len(df)
    categories = int(df["therapeutic_class"].nunique()) if "therapeutic_class" in df.columns else 0

    return {
        "total_records": total_records,
        "therapeutic_classes_count": categories,
        "columns": list(df.columns)
    }


@app.get("/api/charts/therapeutic-class")
def get_therapeutic_class_distribution(top_n: int = 15):
    """Returns Top N therapeutic classes with their counts."""
    df = load_dataset()
    col = "therapeutic_class" if "therapeutic_class" in df.columns else "Therapeutic Class"
    
    if col not in df.columns:
        return {"classes": [], "counts": []}

    counts = df[col].dropna().value_counts().head(top_n)
    return {
        "classes": counts.index.tolist(),
        "counts": counts.values.tolist()
    }


@app.get("/api/charts/habit-forming")
def get_habit_forming_split():
    """Returns Habit-forming split (Yes vs No counts)."""
    df = load_dataset()
    col = "habit_forming" if "habit_forming" in df.columns else "Habit Forming"

    if col not in df.columns:
        return {"labels": [], "counts": []}

    counts = df[col].dropna().astype(str).str.capitalize().value_counts()
    return {
        "labels": counts.index.tolist(),
        "counts": counts.values.tolist()
    }


@app.get("/api/charts/missing-data")
def get_missing_data_percentages():
    """Returns the percentage of null/empty values per column."""
    df = load_dataset()
    # Calculate null or empty string percentage
    missing_pct = (df.isnull() | (df == "") | (df == "NA")).mean() * 100
    sorted_missing = missing_pct.sort_values(ascending=False)

    return {
        "columns": sorted_missing.index.tolist(),
        "percentages": [round(float(v), 2) for v in sorted_missing.values]
    }


@app.get("/api/charts/side-effects-distribution")
def get_side_effects_distribution():
    """Returns statistics and distribution histogram bins of side effect counts per drug."""
    df = load_dataset()
    col = "side_effects" if "side_effects" in df.columns else "sideeffects"

    if col not in df.columns:
        return {"counts": [], "stats": {}}

    # Count items separated by comma
    counts = df[col].dropna().apply(lambda x: len([s.strip() for s in str(x).split(",") if s.strip()]))

    stats = {
        "mean": round(float(counts.mean()), 1) if not counts.empty else 0,
        "median": float(counts.median()) if not counts.empty else 0,
        "max": int(counts.max()) if not counts.empty else 0,
        "min": int(counts.min()) if not counts.empty else 0
    }

    # Binned distribution (e.g. 1-5, 6-10, 11-15, etc.)
    bins = [0, 2, 5, 10, 15, 20, 30, 50]
    labels = ["1-2", "3-5", "6-10", "11-15", "16-20", "21-30", "31+"]
    binned = pd.cut(counts, bins=bins, labels=labels).value_counts().reindex(labels).fillna(0)

    return {
        "binned_labels": binned.index.tolist(),
        "binned_counts": binned.values.tolist(),
        "stats": stats
    }


@app.get("/api/charts/top-side-effects")
def get_top_side_effects(top_n: int = 20):
    """Returns Top 20 most frequent side effects across the dataset."""
    df = load_dataset()
    col = "side_effects" if "side_effects" in df.columns else "sideeffects"

    if col not in df.columns:
        return {"side_effects": [], "counts": []}

    # Split by comma, explode and count frequencies
    top_effects = (
        df[col]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )
    # Remove empty or NA entries
    top_effects = top_effects[~top_effects.str.lower().isin(["", "na", "n/a", "none"])]
    counts = top_effects.value_counts().head(top_n)

    return {
        "side_effects": counts.index.tolist(),
        "counts": counts.values.tolist()
    }


@app.get("/api/charts/uses-text")
def get_uses_text():
    """Returns sampled concatenated text of drug uses for generating a word cloud."""
    df = load_dataset()
    col = "uses" if "uses" in df.columns else "use0"

    if col not in df.columns:
        return {"text": ""}

    # Sample to keep payload fast and lightweight
    sample_text = " ".join(df[col].dropna().sample(min(len(df), 5000), random_state=42).astype(str))
    return {"text": sample_text}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Chatbot endpoint powered by ChromaDB RAG + OpenAI GPT-4o-mini.
    Falls back to a friendly placeholder if RAG is unavailable.
    """
    msg = payload.message.strip()
    timestamp = datetime.now().strftime("%I:%M %p")

    try:
        from src.rag.query import rag_answer
        result = rag_answer(msg)
        return ChatResponse(
            reply=result["answer"],
            timestamp=timestamp,
            sources=result.get("sources", []),
        )
    except Exception as e:
        print(f"[RAG] Query failed, using fallback: {e}")
        # Graceful fallback — never 500 during a live demo
        lower_msg = msg.lower()
        if "side effect" in lower_msg:
            reply = "Common side effects in the dataset include Nausea, Vomiting, and Diarrhea. (RAG unavailable — check index build.)"
        elif "habit" in lower_msg:
            reply = "Our dataset flags over 6,000 habit-forming medicines. (RAG unavailable — check index build.)"
        else:
            reply = f'Your question "{msg}" was received. RAG pipeline is warming up — try again in a moment.'
        return ChatResponse(reply=reply, timestamp=timestamp, sources=[])

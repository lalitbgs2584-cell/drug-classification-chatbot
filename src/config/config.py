import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "spark.master": os.getenv("SPARK_MASTER", "local[*]"),
    "raw_data_path": os.getenv("RAW_DATA_PATH"),
    "processed_data_path": os.getenv("PROCESSED_DATA_PATH"),
    "model_path": os.getenv("MODEL_PATH"),
    "target_column": os.getenv("TARGET_COLUMN"),
    "feature_columns": os.getenv("FEATURE_COLUMNS"),
    "test_size": os.getenv("TEST_SIZE", "0.2"),
    "random_state": os.getenv("RANDOM_STATE", "42"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "aws_region": os.getenv("AWS_REGION"),
    "aws_bucket_name": os.getenv("AWS_BUCKET_NAME"),
    "chroma_db_path": os.getenv("CHROMA_DB_PATH", "src/chroma_db"),
    "embedding_model_name": os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
    "rag_collection_name": os.getenv("RAG_COLLECTION_NAME", "medicines"),
}
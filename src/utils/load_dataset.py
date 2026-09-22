import os
from src.config.config import config
from src.s3.generate_client import generate_s3_client


def download_file(local_path="src/drugs.csv", force=False):
    if os.path.exists(local_path) and not force:
        print(f"'{local_path}' already exists locally. Skipping S3 download.")
        return

    print(f"Downloading dataset from S3 to '{local_path}'...")
    s3 = generate_s3_client()
    s3.download_file(
        config["aws_bucket_name"],
        "medicine_dataset.csv",
        local_path
    )
    print("Downloaded!")
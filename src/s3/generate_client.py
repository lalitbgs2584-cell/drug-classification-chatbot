import boto3
from src.config.config import config


def generate_s3_client():
    s3 = boto3.client("s3",
                aws_access_key_id=config["aws_access_key_id"],
                aws_secret_access_key=config["aws_secret_access_key"],
                region_name=config["aws_region"])
    return s3


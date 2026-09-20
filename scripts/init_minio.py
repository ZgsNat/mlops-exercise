"""
Initialize the MinIO bucket for MLflow artifacts.
"""
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

BUCKET = os.getenv("MINIO_BUCKET", "mlflow")
ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "miniopassword")


def main() -> None:
    print(f"Connecting to MinIO at {ENDPOINT}...")
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET_KEY,
    )

    for attempt in range(1, 31):
        try:
            s3.list_buckets()
            break
        except (EndpointConnectionError, ClientError) as exc:
            print(f"Waiting for MinIO ({attempt}/30): {type(exc).__name__}")
            time.sleep(2)
    else:
        sys.exit(f"MinIO at {ENDPOINT} did not respond in time.")

    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET in existing:
        print(f"Bucket '{BUCKET}' already exists.")
        return

    s3.create_bucket(Bucket=BUCKET)
    print(f"Bucket '{BUCKET}' created successfully.")


if __name__ == "__main__":
    main()

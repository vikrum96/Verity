"""
upload_chunks.py

Uploads transcript chunks from the validation set to S3 as static documents.
Each chunk is stored as chunks/<doc_id>.txt in the S3 bucket.

Run once at project setup before starting the pipeline.

Usage:
    python scripts/upload_chunks.py

Environment variables required:
    S3_BUCKET=verity-documents-<your_account_id>
"""

import os
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ.get("S3_BUCKET")
REGION = os.environ.get("AWS_REGION", "us-east-1")
VALIDATION_SET_PATH = "data/validation_set.csv"


def upload_chunks(df: pd.DataFrame):
    """Upload each chunk in the validation set to S3 as a plain text file."""
    s3 = boto3.client("s3", region_name=REGION)
    uploaded = 0
    skipped = 0

    for _, row in df.iterrows():
        doc_id = row["doc_id"]
        source_text = row["source_text"]

        # Skip synthetic pairs — they don't have real source documents
        if str(doc_id).startswith("synth_"):
            skipped += 1
            continue

        key = f"chunks/{doc_id}.txt"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=source_text.encode("utf-8"),
            ContentType="text/plain"
        )
        print(f"Uploaded {key}")
        uploaded += 1

    print(f"\nDone. {uploaded} chunks uploaded, {skipped} synthetic pairs skipped.")


def main():
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET environment variable is not set.")

    print(f"Loading validation set from {VALIDATION_SET_PATH}...")
    df = pd.read_csv(VALIDATION_SET_PATH)
    print(f"Loaded {len(df)} rows.")

    print(f"Uploading chunks to s3://{S3_BUCKET}/chunks/...")
    upload_chunks(df)


if __name__ == "__main__":
    main()

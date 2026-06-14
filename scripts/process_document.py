"""
process_document.py

Standalone document processing script. Loads a transcript chunk from S3,
runs it through DistilBART, and writes the result to DynamoDB.

Used for testing the core pipeline logic before the FastAPI wrapper is added.

Usage:
    python scripts/process_document.py --doc_id <doc_id> --run_id <run_id>

Environment variables required:
    AWS_PROFILE=verity-dev  (or set via aws configure)
    S3_BUCKET=verity-documents-<your_account_id>
"""

import argparse
import os
import time
import boto3
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ.get("S3_BUCKET")
DYNAMODB_TABLE = os.environ.get("PIPELINE_OUTPUTS_TABLE")
REGION = os.environ.get("AWS_REGION")
MODEL_VERSION = "sshleifer/distilbart-cnn-12-6"


def load_summarizer():
    """Load DistilBART summarization pipeline."""
    print("Loading DistilBART...")
    summarizer = pipeline(
        "summarization",
        model=MODEL_VERSION,
        device=-1
    )
    print("Model loaded.")
    return summarizer


def fetch_chunk_from_s3(doc_id: str) -> str:
    """
    Fetch a transcript chunk from S3 by doc_id.
    Expects the chunk to be stored as a plain text file at chunks/<doc_id>.txt
    """
    s3 = boto3.client("s3", region_name=REGION)
    key = f"chunks/{doc_id}.txt"
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return response["Body"].read().decode("utf-8")


def summarize(summarizer, text: str) -> tuple[str, int]:
    """
    Run text through DistilBART. Returns (summary, latency_ms).
    """
    start = time.time()
    result = summarizer(
        text,
        max_length=130,
        min_length=30,
        truncation=True,
        do_sample=False
    )
    latency_ms = int((time.time() - start) * 1000)
    return result[0]["summary_text"], latency_ms


def write_to_dynamodb(doc_id: str, run_id: str, source_text: str, summary: str, latency_ms: int):
    """Write pipeline output record to DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    item = {
        "doc_id": doc_id,
        "run_id": run_id,
        "source_text": source_text,
        "summary": summary,
        "latency_ms": latency_ms,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": MODEL_VERSION,
    }

    table.put_item(Item=item)
    print(f"Written to DynamoDB: doc_id={doc_id}, run_id={run_id}, latency={latency_ms}ms")


def main():
    parser = argparse.ArgumentParser(description="Process a document chunk through the Verity pipeline.")
    parser.add_argument("--doc_id", required=True, help="Document ID to process")
    parser.add_argument("--run_id", required=True, help="Run ID for this pipeline run")
    args = parser.parse_args()

    if not S3_BUCKET:
        raise ValueError("S3_BUCKET environment variable is not set.")

    summarizer = load_summarizer()

    print(f"Fetching chunk {args.doc_id} from S3...")
    source_text = fetch_chunk_from_s3(args.doc_id)
    print(f"Fetched {len(source_text.split())} words.")

    print("Summarizing...")
    summary, latency_ms = summarize(summarizer, source_text)
    print(f"Summary: {summary}")

    write_to_dynamodb(args.doc_id, args.run_id, source_text, summary, latency_ms)
    print("Done.")


if __name__ == "__main__":
    main()

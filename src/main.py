"""
main.py

FastAPI application for the Verity summarization pipeline.

Routes:
    GET  /health              — ECS health check
    POST /summarize           — Summarize a document chunk from S3
    GET  /results/{doc_id}    — Retrieve all pipeline outputs for a document
    GET  /runs/{run_id}       — Retrieve all pipeline outputs for a run
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

load_dotenv()

S3_BUCKET = os.environ.get("S3_BUCKET")
PIPELINE_OUTPUTS_TABLE = os.environ.get("PIPELINE_OUTPUTS_TABLE")
REGION = os.environ.get("AWS_REGION")
MODEL_VERSION = "sshleifer/distilbart-cnn-12-6"

# Summarizer is loaded once at startup and reused across requests
summarizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer
    print("Loading DistilBART...")
    summarizer = pipeline(
        "summarization",
        model=MODEL_VERSION,
        device=-1
    )
    print("Model loaded.")
    yield
    summarizer = None


app = FastAPI(title="Verity", lifespan=lifespan)

class SummarizeRequest(BaseModel):
    doc_id: str
    run_id: str

class SummarizeResponse(BaseModel):
    doc_id: str
    run_id: str
    summary: str
    latency_ms: int


@app.get("/health")
def health():
    """ECS health check endpoint."""
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_document(request: SummarizeRequest):
    """
    Load a transcript chunk from S3 by doc_id, run DistilBART inference,
    write the result to DynamoDB, and return the summary.
    """
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET environment variable not set.")

    # Fetch chunk from S3
    try:
        s3 = boto3.client("s3", region_name=REGION)
        key = f"chunks/{request.doc_id}.txt"
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        source_text = response["Body"].read().decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch chunk {request.doc_id}: {e}")

    # Run inference
    try:
        start = time.time()
        result = summarizer(
            source_text,
            max_length=130,
            min_length=30,
            truncation=True,
            do_sample=False
        )
        latency_ms = int((time.time() - start) * 1000)
        summary = result[0]["summary_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Write to DynamoDB
    try:
        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        table = dynamodb.Table(PIPELINE_OUTPUTS_TABLE)
        table.put_item(Item={
            "doc_id": request.doc_id,
            "run_id": request.run_id,
            "source_text": source_text,
            "summary": summary,
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_version": MODEL_VERSION,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DynamoDB write failed: {e}")

    return SummarizeResponse(
        doc_id=request.doc_id,
        run_id=request.run_id,
        summary=summary,
        latency_ms=latency_ms,
    )


@app.get("/results/{doc_id}")
def get_results(doc_id: str):
    """
    Retrieve all pipeline outputs for a given document across all runs.
    """
    try:
        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        table = dynamodb.Table(PIPELINE_OUTPUTS_TABLE)
        response = table.query(
            KeyConditionExpression=Key("doc_id").eq(doc_id)
        )
        items = response.get("Items", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DynamoDB query failed: {e}")

    if not items:
        raise HTTPException(status_code=404, detail=f"No results found for doc_id {doc_id}")

    return {"doc_id": doc_id, "results": items}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """
    Retrieve all pipeline outputs for a given run.
    Used by the Lambda evaluation runner to batch-fetch outputs for evaluation.
    """
    try:
        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        table = dynamodb.Table(PIPELINE_OUTPUTS_TABLE)
        # Scan with filter — acceptable at this data volume
        # Production upgrade: add a GSI on run_id for efficient querying
        response = table.scan(FilterExpression=Key("run_id").eq(run_id))
        items = response.get("Items", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DynamoDB scan failed: {e}")

    if not items:
        raise HTTPException(status_code=404, detail=f"No results found for run_id {run_id}")

    return {"run_id": run_id, "results": items}

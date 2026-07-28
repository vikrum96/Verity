# Verity — LLM Evaluation and Monitoring System

An end-to-end LLM evaluation platform for document summarization. The summarization is not the point — the evaluation, calibration, and regression detection are.

## What it does

Verity runs a document summarization pipeline, evaluates outputs across five failure categories using a calibrated LLM judge, tracks scores over time, and alerts when model quality regresses.

- **Summarization** — DistilBART processes earnings call transcript chunks served via FastAPI on AWS ECS Fargate
- **Evaluation** — Gemini judge scores each output across factual consistency, completeness, coherence, hallucination, and instruction following
- **Calibration** — judge calibrated against a 57-example human-labeled validation set; agreement rates documented below
- **Monitoring** — run-over-run score deltas computed via Lambda, metrics pushed to CloudWatch, SNS alerts on consecutive regressions

## Status

Under active development. Serving infrastructure complete and deployed. Evaluation layer in progress.

| Component | Status |
|---|---|
| FastAPI serving layer (ECS Fargate) | Complete |
| DynamoDB pipeline outputs + evaluation results tables | Complete |
| S3 document storage | Complete |
| CI/CD via GitHub Actions | Complete |
| Gemini judge + schema validation | Complete |
| Judge calibration against validation set | Complete |
| Lambda evaluation runner + EventBridge | Complete |
| Behavioral profiling + regression detection | Complete |
| CloudWatch dashboard + SNS alerts | In progress |

## Stack

Python, HuggingFace Transformers, Gemini, FastAPI, Docker, AWS ECS Fargate, Lambda, DynamoDB, EventBridge, CloudWatch, SNS, S3, GitHub Actions

## Dataset

S&P 500 earnings call transcripts (`kurry/sp500_earnings_transcripts` via HuggingFace Datasets).

---

*Full README with architecture diagram, calibration results, and CloudWatch dashboard coming soon.*

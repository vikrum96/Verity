# Verity — Design Document
## LLM Evaluation and Monitoring System

*Written before any code was written. This document is the architectural contract for the build phase.*

---

## 1. Problem Statement

Summarization models deployed in production degrade in ways that are hard to detect without structured evaluation. A model can produce outputs that are fluent, plausible, and wrong — and without a calibrated evaluation layer, that degradation is invisible until a user notices it. Verity is a system that makes model quality observable: it runs a document summarization pipeline, evaluates outputs across five failure categories using an LLM judge, tracks scores over time, and alerts when quality regresses.

The summarization model is not the point. The evaluation, calibration, and regression detection are.

---

## 2. Dataset

**Source:** `kurry/sp500_earnings_transcripts` via HuggingFace Datasets. A curated collection of over 33,000 earnings call transcripts from 685 S&P 500 companies spanning 2005–2025, pre-cleaned with no HTML artifacts or exhibit noise. Transcripts span multiple companies, quarters, and sectors, introducing realistic variation in document structure, topic density, and language style.

- The kurry dataset is verified earnings call content, eliminating the filtering problem entirely.

**Chunking strategy:** Transcripts are chunked at ~512 tokens with 50-token overlap before being passed to the summarization model.

**Why 512 tokens:**
DistilBART has a hard maximum input of 1024 tokens — anything beyond that is silently truncated, which introduces completeness failures at the model level before evaluation begins. 512 tokens is half the maximum context, giving the model sufficient surrounding text to produce coherent summaries without risking truncation. For earnings call transcripts specifically, 512 tokens maps naturally to 3-4 conversational exchanges, which is a semantically coherent unit.

**Why 50-token overlap:**
Prevents sentences that straddle chunk boundaries from being lost entirely. The last 50 tokens of chunk N become the first 50 tokens of chunk N+1, preserving cross-boundary context at low cost.

**Static dataset:** Transcripts are pre-loaded into S3 at project setup and treated as a fixed dataset. No live ingestion. This scopes the project to evaluation and monitoring rather than data pipeline engineering, which is where the resume signal is.

---

## 3. Summarization Model

**Model:** `sshleifer/distilbart-cnn-12-6` via HuggingFace Transformers pipeline.

**Why DistilBART:**
This is a deliberate choice of a known-imperfect model. DistilBART produces realistic summarization failures — factual inconsistencies, completeness gaps, occasional hallucinations — which is exactly what an evaluation system needs to be worth building. A near-perfect model would produce near-perfect scores on every run, making behavioral profiling and regression detection trivially uninteresting. The point of Verity is the evaluation layer; DistilBART is the vehicle that produces failures worth evaluating.

**Why not a larger or more capable model:**
A more capable model (BART-large, PEGASUS, a GPT-class model) would reduce the frequency and severity of failures, undermining the evaluation system's ability to demonstrate meaningful score variation across runs. It would also increase inference cost and latency on ECS without adding anything to the resume signal.

---

## 4. Infrastructure

### 4.1 Serving Layer — ECS Fargate + FastAPI

The summarization pipeline runs as a containerized FastAPI application on ECS Fargate.

**Why ECS Fargate over alternatives:**
- **vs Lambda for serving:** DistilBART is a 300MB+ model. Lambda has a 512MB deployment package limit and cold start latency that makes model loading impractical. ECS has no such constraint.
- **vs EC2:** Fargate is serverless container execution — no instance management, no AMI selection, no patching. For a portfolio project the operational simplicity is the right tradeoff.
- **vs SageMaker:** SageMaker endpoints are purpose-built for ML serving but carry significant configuration overhead and higher baseline cost. ECS with a HuggingFace pipeline is simpler to reason about and sufficient for this use case.

**Task sizing:** 0.25 vCPU, 1GB RAM. Sufficient for DistilBART inference at low concurrency.

**Cost discipline:** ECS is spun down when not in active use. At $0.013/hour for this task size, total ECS cost across the project is estimated at $1–3.

**API routes:**

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | ECS health check endpoint. Returns 200 OK. Required for Fargate task health monitoring. |
| `/summarize` | POST | Accepts a doc_id, loads the corresponding transcript chunk from S3, runs DistilBART inference, writes input/output/latency to DynamoDB pipeline_outputs table, returns summary. |
| `/results/{doc_id}` | GET | Retrieves all pipeline outputs for a given document. Used for debugging and manual inspection. |
| `/runs/{run_id}` | GET | Retrieves all pipeline outputs for a given run. Used by the Lambda evaluation runner to batch-fetch outputs for a run without needing individual doc_ids. |

**Why four routes and not three:**
`/runs/{run_id}` is added over the baseline three because the Lambda evaluation runner needs to fetch all outputs from a run in a single call. Without this route, Lambda would need to either scan DynamoDB directly or track individual doc_ids externally — both are messier than a single API call keyed on run_id.

### 4.2 Storage — S3

Raw EDGAR transcript chunks are stored in S3 as a static dataset, pre-loaded at project setup. The `/summarize` endpoint reads from S3 by doc_id at inference time.

**Why S3 for documents:**
Documents are large, infrequently accessed, and read-only after initial load. S3 is the natural fit — cheap, durable, and trivially integrated with ECS via the boto3 SDK. Storing documents in DynamoDB would be wasteful (DynamoDB item size limit is 400KB; transcript chunks can exceed this) and storing them on the ECS container filesystem would make them non-persistent across task restarts.

### 4.3 Database — DynamoDB

Two tables with on-demand billing.

**Why DynamoDB over PostgreSQL / RDS:**
PostgreSQL would provide richer query capabilities — joins, window functions, aggregations — which would be useful for behavioral profiling. The reason to use DynamoDB is operational cost. RDS requires a persistent instance running 24/7, adding $15–25/month minimum regardless of usage. DynamoDB on-demand has zero idle cost — cost is incurred only on reads and writes, which at this project's volume amounts to pennies. For simple, predictable access patterns with infrequent writes, DynamoDB is the correct tradeoff. PostgreSQL is documented as the production upgrade path in the README.

**Table 1: `pipeline_outputs`**

| Field | Type | Notes |
|---|---|---|
| `doc_id` | String (PK) | Unique identifier for the source document chunk |
| `run_id` | String (SK) | Identifies the pipeline run. Composite key enables multiple runs per document. |
| `source_text` | String | Raw transcript chunk passed to DistilBART |
| `summary` | String | Model output |
| `latency_ms` | Number | Inference latency in milliseconds |
| `timestamp` | String | ISO 8601 timestamp of the summarization call |
| `model_version` | String | Model identifier for tracking across potential future model changes |

**Table 2: `evaluation_results`**

| Field | Type | Notes |
|---|---|---|
| `eval_id` | String (PK) | Unique identifier for this evaluation record |
| `doc_id` | String (GSI PK) | Foreign key to pipeline_outputs. GSI enables efficient cross-run queries by document. |
| `run_id` | String (GSI SK) | Links evaluation back to the pipeline run |
| `eval_run_tag` | String | Groups all evaluation records from a single Lambda invocation. Load-bearing for behavioral profiling — delta calculation groups by eval_run_tag. |
| `factual_consistency` | Number | Gemini score: 0, 0.5, or 1 |
| `completeness` | Number | Gemini score: 0, 0.5, or 1 |
| `coherence` | Number | Gemini score: 0, 0.5, or 1 |
| `hallucination` | Number | Gemini score: 0, 0.5, or 1 |
| `instruction_following` | Number | Gemini score: 0, 0.5, or 1 |
| `judge_raw_response` | String | Raw Gemini response stored for debugging and rubric revision |
| `timestamp` | String | ISO 8601 timestamp of the evaluation call |

**Why two tables and not one:**
Pipeline outputs and evaluation results have different write patterns (ECS writes outputs; Lambda writes evaluations, potentially hours later), different read patterns, and different access lifetimes. Combining them into one table would require either heavy denormalization or a wide table with sparse attributes. Two tables with a clean doc_id/run_id relationship is simpler to reason about and maintain.

**Why `eval_run_tag` is load-bearing:**
Behavioral profiling computes per-category score deltas between consecutive runs. To do this, it needs to group all evaluation records from a single Lambda invocation together. `eval_run_tag` is that grouping key — without it, delta calculation cannot determine which records belong to the same evaluation run.

### 4.4 Evaluation Runner — Lambda + EventBridge

The evaluation pipeline runs as a Lambda function triggered by an EventBridge rule. In practice, the cron rule is disabled by default and runs are triggered manually via direct Lambda invocation or a GitHub Actions workflow dispatch.

**Why Lambda for the evaluation runner:**
The evaluation runner is a short-lived batch job — fetch outputs from a run, send each to Gemini, write scores to DynamoDB, compute deltas, push metrics to CloudWatch. This is a textbook Lambda use case: event-driven, stateless, short duration, infrequent execution. Running it on ECS would mean either keeping the ECS service running continuously (cost) or spinning up a separate task per run (operational complexity). Lambda is simpler and cheaper for this workload.

**Why EventBridge and not a cron job on EC2 or a GitHub Actions schedule:**
EventBridge is the AWS-native scheduling primitive. Using it demonstrates knowledge of the AWS event-driven architecture pattern, which is directly relevant to MLOps roles. A GitHub Actions schedule would work but keeps the scheduling logic outside AWS, which weakens the infrastructure story. An EC2 cron job requires a persistent instance. EventBridge + Lambda is the production-correct pattern.

**Why runs are triggered manually rather than on an actual nightly schedule:**
This is a portfolio project. Running the evaluator on a true nightly schedule would accumulate Gemini API calls and DynamoDB writes continuously, adding cost and complexity with no benefit. The EventBridge rule is configured and documented — demonstrating the capability — but the cron is disabled by default. Manual invocation is used to generate the 3+ runs of data required for the CloudWatch dashboard.

### 4.5 Monitoring — CloudWatch + SNS

**CloudWatch metrics tracked:**

| Metric | Type | Rationale |
|---|---|---|
| `SummarizationLatencyMs` | Infrastructure | Latency spikes distinguish infrastructure problems from model quality regressions |
| `ECSTaskCount` | Infrastructure | Task count dropping to zero means the system is down and requests are silently failing |
| `JudgeFailureRate` | Data quality | Gemini errors or schema validation failures mean evaluation scores for that run are incomplete or untrustworthy |
| `Score_FactualConsistency` | Evaluation | Per-run average score; feeds behavioral profiling and dashboard |
| `Score_Completeness` | Evaluation | Per-run average score; feeds behavioral profiling and dashboard |
| `Score_Coherence` | Evaluation | Per-run average score; feeds behavioral profiling and dashboard |
| `Score_Hallucination` | Evaluation | Per-run average score; feeds behavioral profiling and dashboard |
| `Score_InstructionFollowing` | Evaluation | Per-run average score; feeds behavioral profiling and dashboard |
| `Regression_[Category]` | Behavioral | Binary flag set when a category score drops more than 5% from the previous run |

**Why track regression flags as separate metrics rather than recomputing from score history:**
Storing regression flags explicitly makes the SNS alerting logic simple — a CloudWatch alarm watches for consecutive 1s on a high-severity regression metric and fires. Recomputing from score history every time would require a Lambda function just to evaluate the alert condition, adding unnecessary complexity.

**SNS alerts configured for:**
- Consecutive regressions on Hallucination or Factual Consistency (high severity — immediate alert)
- Judge failure rate exceeding threshold (data quality — scores from this run may be invalid)
- Summarization latency exceeding threshold (infrastructure health)
- ECS task count dropping to zero (system down)

**Why SNS over CloudWatch alarms directly:**
CloudWatch alarms can send emails directly, but SNS decouples the alerting logic from the notification channel. In production you'd route different severities to different channels (PagerDuty, Slack, email). Using SNS demonstrates awareness of that pattern even at portfolio scale.

### 4.6 CI/CD — GitHub Actions

Automated container build and push to ECR on every push to main. ECS service update triggered automatically after successful push.

**Why GitHub Actions over alternatives (CodePipeline, CircleCI):**
GitHub Actions is where the code lives. No additional service integration required, no extra cost, and GitHub Actions CI/CD is the most widely recognized pattern in job postings for the roles being targeted.

---

## 5. Evaluation Rubric — Draft

The Gemini judge receives the source transcript chunk, the DistilBART summary, and a structured rubric prompt. It returns a JSON object with a score (0, 0.5, or 1) and a one-sentence justification for each of the five failure categories defined in `failure_mode_taxonomy.md`.

**Schema validation is enforced on every Gemini response.** Responses that do not conform to the expected schema are logged as judge failures, incremented against the `JudgeFailureRate` metric, and excluded from scoring for that run.

**Draft rubric prompt structure:**
```
You are an evaluation judge for a document summarization system.

Given the source document and the model-generated summary below, score the summary on each of the following five dimensions. For each dimension, return a score of 0 (fail), 0.5 (marginal), or 1 (pass), and a one-sentence justification.

Dimensions:
1. Factual Consistency: Does the summary accurately reflect the facts in the source? A score of 0 means the summary contradicts the source. A score of 0.5 means a claim is imprecisely stated but directionally correct. A score of 1 means all claims are accurately grounded.
2. Completeness: Does the summary capture the material substance of the source, including primary conclusions, key supporting evidence, and significant qualifications? A score of 0 means critical information is absent. A score of 0.5 means secondary detail is missing but primary substance is present. A score of 1 means the summary is materially complete.
3. Coherence: Is the summary internally consistent and logically structured, independent of the source? A score of 0 means the summary is self-contradictory. A score of 0.5 means it is structurally awkward but not misleading. A score of 1 means it is internally consistent and well-structured.
4. Hallucination: Does the summary reference any entity, figure, or fact not present in the source? A score of 0 means at least one reference has no basis in the source. A score of 0.5 means a reference is loosely inferable but not explicit. A score of 1 means all references are clearly grounded.
5. Instruction Following: Does the summary conform to the format and constraints specified in the prompt? A score of 0 means a material format constraint is violated. A score of 0.5 means the summary partially conforms. A score of 1 means full conformance.

Return only a JSON object. No preamble, no explanation outside the JSON.

Source document:
{source_text}

Model summary:
{summary}
```

**This rubric is a draft.** It will be revised during Week 7 calibration based on systematic disagreements between Gemini scores and human-labeled ground truth. The calibration target is 75% agreement per category before the judge is integrated into the scheduled pipeline.

---

## 6. Calibration Plan

**Process:**
1. Run the Gemini judge against the 40–50 example human-labeled validation set (`data/validation_set.csv`)
2. Compute per-category agreement rate, false positive rate, and false negative rate
3. Identify categories where agreement is below 75%
4. Revise the rubric on systematic disagreements — not on individual edge cases
5. Re-run against the validation set
6. Repeat until all categories stabilize above 75% agreement
7. Commit the final rubric and calibration results table to the repo

**What counts as a systematic disagreement:**
If Gemini scores a category differently from the human label on more than 25% of examples, and the errors cluster around a specific type of case (e.g., Gemini consistently passes completeness when supporting evidence is missing), that is a rubric problem, not a model quirk. The rubric definition for that category is revised to be more explicit about the edge case.

**Calibration results go in:**
- `docs/calibration_results.md` — full table with per-category agreement, FP, FN rates
- `README.md` — summary table with final numbers
- Resume bullet — specific numbers only (e.g., "83% average judge-human agreement across five failure categories")

**What to do if a category does not stabilize above 75%:**
Do not move forward. Iterate on the rubric. A calibration number below 75% means the judge is not reliably detecting that failure mode, and behavioral profiling built on unreliable scores produces meaningless regression flags. The calibration gate is not optional.

---

## 7. What Would Be Improved in Production

These are honest tradeoffs made for portfolio scope that would change in a production system:

- **PostgreSQL over DynamoDB** — richer query capabilities for behavioral analytics, window functions for trend analysis, proper foreign key constraints
- **Live ingestion over static dataset** — a real pipeline would ingest documents continuously rather than from a pre-loaded S3 bucket
- **A more capable judge** — Gemini 1.5 Pro (free tier) is used for cost reasons; a production system would use a larger model with higher reliability and lower hallucination rate as the judge itself
- **Multi-model evaluation** — comparing DistilBART against BART-large or PEGASUS across the same documents would produce more interesting behavioral profiling data
- **Human-in-the-loop review** — systematic disagreements between the judge and human labels would trigger a review queue rather than a rubric revision cycle

---

*This document will be updated if architectural decisions change during the build phase. All changes tracked via Git commit history.*

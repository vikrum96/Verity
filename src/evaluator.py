"""
evaluator.py

Lambda handler for the evaluation pipeline. Fetches pipeline outputs
for a given run, evaluates each pair using the Gemini judge, writes results
to DynamoDB, computes behavioral deltas, pushes metrics to CloudWatch, and
fires SNS alerts on consecutive regressions.

Triggered manually or via EventBridge cron rule (disabled by default).
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from dotenv import load_dotenv

load_dotenv()

from judge import evaluate_pair

REGION = os.environ.get("AWS_REGION", "us-east-1")
PIPELINE_OUTPUTS_TABLE = os.environ.get("PIPELINE_OUTPUTS_TABLE", "verity-pipeline-outputs")
EVALUATION_RESULTS_TABLE = os.environ.get("EVALUATION_RESULTS_TABLE", "verity-evaluation-results")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
CLOUDWATCH_NAMESPACE = "Verity"

CATEGORIES = [
    "factual_consistency",
    "completeness",
    "coherence",
    "hallucination",
    "instruction_following",
]

# Categories where consecutive regressions trigger immediate SNS alert
HIGH_SEVERITY_CATEGORIES = {"hallucination", "factual_consistency"}

# Score drop threshold that constitutes a regression
REGRESSION_THRESHOLD = 0.05


def get_dynamodb():
    return boto3.resource("dynamodb", region_name=REGION)


def get_cloudwatch():
    return boto3.client("cloudwatch", region_name=REGION)


def get_sns():
    return boto3.client("sns", region_name=REGION)


def fetch_pipeline_outputs(run_id: str) -> list[dict]:
    """Fetch all pipeline outputs for a given run from DynamoDB."""
    dynamodb = get_dynamodb()
    table = dynamodb.Table(PIPELINE_OUTPUTS_TABLE)

    response = table.scan(
        FilterExpression=boto3.dynamodb.conditions.Attr("run_id").eq(run_id)
    )
    return response.get("Items", [])


def write_evaluation_result(doc_id: str, run_id: str, eval_run_tag: str, scores: dict, raw_response: str):
    """Write evaluation result for a single document to DynamoDB."""
    dynamodb = get_dynamodb()
    table = dynamodb.Table(EVALUATION_RESULTS_TABLE)

    item = {
        "eval_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "run_id": run_id,
        "eval_run_tag": eval_run_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_raw_response": raw_response,
    }

    for category in CATEGORIES:
        # DynamoDB doesn't support float — store as Decimal
        item[category] = Decimal(str(scores[category]["score"]))

    table.put_item(Item=item)


def fetch_previous_run_scores(eval_run_tag: str) -> dict | None:
    """
    Fetch average scores from the most recent previous evaluation run.
    Returns None if no previous run exists.
    """
    dynamodb = get_dynamodb()
    table = dynamodb.Table(EVALUATION_RESULTS_TABLE)

    # Scan for all eval records to find previous runs
    response = table.scan()
    items = response.get("Items", [])

    # Group by eval_run_tag, exclude current run
    runs = {}
    for item in items:
        tag = item.get("eval_run_tag")
        if tag and tag != eval_run_tag:
            if tag not in runs:
                runs[tag] = []
            runs[tag].append(item)

    if not runs:
        return None

    # Get the most recent previous run by timestamp
    latest_tag = max(runs.keys(), key=lambda t: t)
    previous_items = runs[latest_tag]

    # Compute average scores for each category
    averages = {}
    for category in CATEGORIES:
        scores = [float(item[category]) for item in previous_items if category in item]
        averages[category] = sum(scores) / len(scores) if scores else 0.0

    return averages


def compute_deltas(current_scores: dict, previous_scores: dict) -> dict:
    """Compute per-category score deltas between current and previous run."""
    return {
        category: current_scores[category] - previous_scores[category]
        for category in CATEGORIES
    }


def push_cloudwatch_metrics(
    eval_run_tag: str,
    avg_scores: dict,
    regression_flags: dict,
    judge_failure_count: int,
    total_pairs: int,
):
    """Push all 13 metrics to CloudWatch."""
    cloudwatch = get_cloudwatch()
    timestamp = datetime.now(timezone.utc)

    metric_data = []

    # Per-category average scores
    for category in CATEGORIES:
        metric_name = f"Score_{category.replace('_', ' ').title().replace(' ', '')}"
        metric_data.append({
            "MetricName": metric_name,
            "Value": avg_scores[category],
            "Unit": "None",
            "Timestamp": timestamp,
            "Dimensions": [{"Name": "EvalRunTag", "Value": eval_run_tag}],
        })

    # Per-category regression flags (1 = regression detected, 0 = no regression)
    for category in CATEGORIES:
        metric_data.append({
            "MetricName": f"Regression_{category.replace('_', ' ').title().replace(' ', '')}",
            "Value": 1.0 if regression_flags.get(category, False) else 0.0,
            "Unit": "None",
            "Timestamp": timestamp,
            "Dimensions": [{"Name": "EvalRunTag", "Value": eval_run_tag}],
        })

    # Judge failure rate
    judge_failure_rate = judge_failure_count / total_pairs if total_pairs > 0 else 0.0
    metric_data.append({
        "MetricName": "JudgeFailureRate",
        "Value": judge_failure_rate,
        "Unit": "None",
        "Timestamp": timestamp,
        "Dimensions": [{"Name": "EvalRunTag", "Value": eval_run_tag}],
    })

    # CloudWatch accepts max 20 metrics per call
    for i in range(0, len(metric_data), 20):
        cloudwatch.put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricData=metric_data[i:i+20],
        )


def send_sns_alert(category: str, delta: float, eval_run_tag: str):
    """Send SNS alert for a high-severity consecutive regression."""
    if not SNS_TOPIC_ARN:
        print(f"SNS_TOPIC_ARN not set — skipping alert for {category}")
        return

    sns = get_sns()
    message = (
        f"Verity regression alert\n\n"
        f"Category: {category}\n"
        f"Score drop: {abs(delta):.3f}\n"
        f"Eval run: {eval_run_tag}\n"
        f"Severity: {'HIGH' if category in HIGH_SEVERITY_CATEGORIES else 'MEDIUM'}\n"
    )

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"Verity: Regression detected in {category}",
        Message=message,
    )
    print(f"SNS alert sent for {category}")


def handler(event, context):
    """
    Lambda entry point.

    Expected event format:
    {
        "run_id": "run_001"
    }
    """
    run_id = event.get("run_id")
    if not run_id:
        raise ValueError("Event must contain 'run_id'")

    eval_run_tag = f"eval_{run_id}_{int(time.time())}"
    print(f"Starting evaluation run: {eval_run_tag}")

    # Fetch pipeline outputs for this run
    outputs = fetch_pipeline_outputs(run_id)
    if not outputs:
        raise ValueError(f"No pipeline outputs found for run_id: {run_id}")

    print(f"Found {len(outputs)} documents to evaluate.")

    # Run evaluation
    all_scores = []
    judge_failures = 0

    for output in outputs:
        doc_id = output["doc_id"]
        source_text = output.get("source_text", "")
        summary = output.get("summary", "")

        print(f"Evaluating {doc_id}...")
        result = evaluate_pair(source_text, summary)

        if result is None:
            print(f"Judge failure for {doc_id} — skipping.")
            judge_failures += 1
            continue

        write_evaluation_result(
            doc_id=doc_id,
            run_id=run_id,
            eval_run_tag=eval_run_tag,
            scores=result,
            raw_response=json.dumps(result),
        )

        all_scores.append(result)

    if not all_scores:
        raise RuntimeError("All judge evaluations failed — no results to process.")

    # Compute average scores for this run
    avg_scores = {}
    for category in CATEGORIES:
        scores = [r[category]["score"] for r in all_scores]
        avg_scores[category] = sum(scores) / len(scores)

    print(f"Average scores: {avg_scores}")

    # Behavioral profiling — compare against previous run
    previous_scores = fetch_previous_run_scores(eval_run_tag)
    regression_flags = {}

    if previous_scores:
        deltas = compute_deltas(avg_scores, previous_scores)
        print(f"Score deltas: {deltas}")

        for category, delta in deltas.items():
            regression = delta < -REGRESSION_THRESHOLD
            regression_flags[category] = regression

            if regression:
                print(f"Regression detected in {category}: {delta:.3f}")
                if category in HIGH_SEVERITY_CATEGORIES:
                    send_sns_alert(category, delta, eval_run_tag)
    else:
        print("No previous run found — skipping delta calculation.")
        regression_flags = {category: False for category in CATEGORIES}

    # Push CloudWatch metrics
    push_cloudwatch_metrics(
        eval_run_tag=eval_run_tag,
        avg_scores=avg_scores,
        regression_flags=regression_flags,
        judge_failure_count=judge_failures,
        total_pairs=len(outputs),
    )

    print(f"Evaluation complete. Judge failures: {judge_failures}/{len(outputs)}")

    return {
        "eval_run_tag": eval_run_tag,
        "pairs_evaluated": len(all_scores),
        "judge_failures": judge_failures,
        "avg_scores": avg_scores,
        "regression_flags": regression_flags,
    }
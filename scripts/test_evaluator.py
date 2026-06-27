"""
test_evaluator.py

Runs the Lambda evaluator handler locally against a real pipeline run.
Tests the full evaluation pipeline end to end before Lambda deployment.

Usage:
    python scripts/test_evaluator.py --run_id <run_id>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import handler


def main():
    parser = argparse.ArgumentParser(description="Test the evaluator handler locally.")
    parser.add_argument("--run_id", required=True, help="Run ID to evaluate")
    args = parser.parse_args()

    print(f"Testing evaluator with run_id: {args.run_id}\n")

    event = {"run_id": args.run_id}
    result = handler(event, context=None)

    print("\nHandler result:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

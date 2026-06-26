"""
test_judge.py

Runs the Gemini judge on the first 10 pairs from the validation set.
Used to verify judge behavior before running full calibration.

Usage:
    python scripts/test_judge.py
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import evaluate_pair

VALIDATION_SET_PATH = "data/validation_set.csv"
TEST_SAMPLE_SIZE = 10


def main():
    df = pd.read_csv(VALIDATION_SET_PATH)

    # Skip synthetic pairs for this test
    organic = df[~df["doc_id"].astype(str).str.startswith("synth_")]
    sample = organic.head(TEST_SAMPLE_SIZE)

    print(f"Testing judge on {len(sample)} pairs...\n")

    passed = 0
    failed = 0

    for i, row in sample.iterrows():
        print(f"Pair {i+1}/{len(sample)} | doc_id: {row['doc_id']}")
        result = evaluate_pair(row["source_text"], row["summary"])

        if result is None:
            print("  JUDGE FAILURE — returned None\n")
            failed += 1
            continue

        passed += 1
        for category, values in result.items():
            print(f"  {category}: {values['score']} — {values['justification']}")
        print()

    print(f"Results: {passed} passed, {failed} failed out of {len(sample)} pairs.")


if __name__ == "__main__":
    main()

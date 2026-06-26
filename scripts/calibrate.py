"""
calibrate.py

Runs the Gemini judge against all 57 pairs in the validation set and computes
per-category agreement, false positive, and false negative rates against human
labels. Outputs results to docs/calibration_results.json.

Usage:
    python scripts/calibrate.py

Expected runtime: ~13 minutes (57 pairs x 13s rate limit delay)
"""

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import evaluate_pair

VALIDATION_SET_PATH = "data/validation_set.csv"
OUTPUT_PATH = "docs/calibration_results.json"

CATEGORIES = [
    "factual_consistency",
    "completeness",
    "coherence",
    "hallucination",
    "instruction_following",
]


def scores_agree(human_score: float, gemini_score: float) -> bool:
    """
    Two scores agree if they are exactly equal.
    0.5 is treated as its own distinct label — not a rounding of 0 or 1.
    """
    return human_score == gemini_score


def is_false_positive(human_score: float, gemini_score: float) -> bool:
    """
    False positive: Gemini passes (1) when human labeled fail (0 or 0.5).
    Gemini is more lenient than the human label.
    """
    return gemini_score == 1 and human_score < 1


def is_false_negative(human_score: float, gemini_score: float) -> bool:
    """
    False negative: Gemini fails (0) when human labeled pass (1 or 0.5).
    Gemini is more strict than the human label.
    """
    return gemini_score == 0 and human_score > 0


def main():
    df = pd.read_csv(VALIDATION_SET_PATH)
    print(f"Loaded {len(df)} pairs from validation set.")
    print(f"Estimated runtime: ~{len(df) * 13 // 60} minutes\n")

    results = []
    judge_failures = []

    for i, row in df.iterrows():
        doc_id = str(row["doc_id"])
        is_synthetic = doc_id.startswith("synth_")
        print(f"[{i+1}/{len(df)}] doc_id: {doc_id} {'(synthetic)' if is_synthetic else ''}")

        gemini_result = evaluate_pair(row["source_text"], row["summary"])

        if gemini_result is None:
            print(f"  JUDGE FAILURE — skipping\n")
            judge_failures.append(doc_id)
            continue

        pair_result = {
            "doc_id": doc_id,
            "is_synthetic": is_synthetic,
            "scores": {}
        }

        for category in CATEGORIES:
            human_score = float(row[category])
            gemini_score = float(gemini_result[category]["score"])
            justification = gemini_result[category]["justification"]

            pair_result["scores"][category] = {
                "human": human_score,
                "gemini": gemini_score,
                "agree": scores_agree(human_score, gemini_score),
                "false_positive": is_false_positive(human_score, gemini_score),
                "false_negative": is_false_negative(human_score, gemini_score),
                "gemini_justification": justification,
            }

        results.append(pair_result)
        print(f"  Done.\n")

    # Compute per-category summary statistics
    summary = {}
    for category in CATEGORIES:
        category_results = [
            r["scores"][category]
            for r in results
            if category in r["scores"]
        ]

        total = len(category_results)
        if total == 0:
            continue

        agreements = sum(1 for r in category_results if r["agree"])
        false_positives = sum(1 for r in category_results if r["false_positive"])
        false_negatives = sum(1 for r in category_results if r["false_negative"])

        summary[category] = {
            "total_pairs": total,
            "agreement_count": agreements,
            "agreement_rate": round(agreements / total, 3),
            "false_positive_count": false_positives,
            "false_positive_rate": round(false_positives / total, 3),
            "false_negative_count": false_negatives,
            "false_negative_rate": round(false_negatives / total, 3),
            "passes_threshold": agreements / total >= 0.75,
        }

    # Separate organic vs synthetic summary
    organic_results = [r for r in results if not r["is_synthetic"]]
    synthetic_results = [r for r in results if r["is_synthetic"]]

    output = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_pairs_evaluated": len(results),
        "judge_failures": len(judge_failures),
        "judge_failure_doc_ids": judge_failures,
        "organic_pairs_evaluated": len(organic_results),
        "synthetic_pairs_evaluated": len(synthetic_results),
        "category_summary": summary,
        "pair_results": results,
    }

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCalibration complete. Results saved to {OUTPUT_PATH}")
    print(f"Judge failures: {len(judge_failures)}/{len(df)}")
    print("\nPer-category summary:")
    for category, stats in summary.items():
        status = "PASS" if stats["passes_threshold"] else "FAIL — revise rubric"
        print(f"  {category}: {stats['agreement_rate']*100:.1f}% agreement | "
              f"FP: {stats['false_positive_rate']*100:.1f}% | "
              f"FN: {stats['false_negative_rate']*100:.1f}% | {status}")


if __name__ == "__main__":
    main()

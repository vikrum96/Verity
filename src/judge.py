"""
judge.py

Gemini judge module for evaluating DistilBART summaries across five failure
categories. Designed to be tested standalone before integration into the
Lambda evaluation pipeline.

Usage:
    from src.judge import evaluate_pair
    result = evaluate_pair(source_text, summary)
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
# 13s between calls stays safely under 5 RPM free tier limit
RATE_LIMIT_DELAY_SECONDS = 13

RUBRIC_PROMPT_TEMPLATE = """You are an evaluation judge for a document summarization system.

Given the source document and the model-generated summary below, score the summary on each of the following five dimensions. For each dimension, return a score of 0 (fail), 0.5 (marginal), or 1 (pass), and a one-sentence justification.

Scoring dimensions:
1. factual_consistency: Does the summary accurately reflect the facts in the source? Score 0 if the summary states something that directly contradicts the source — a wrong figure, a misattributed statement, or a claim that inverts what the source says. Score 0.5 if a claim is imprecisely stated but directionally correct, such as an approximated figure or a dropped qualifier. Score 1 if all claims are accurately grounded in the source. Important distinctions: repeated or duplicated names are a coherence failure, not a factual consistency failure — score these as 1 for factual consistency. Omitting information is a completeness failure, not a factual consistency failure — only penalize factual consistency when something stated is wrong, not when something true is missing.
2. completeness: Does the summary capture the material substance of the source? Score 0 if the summary fails to capture the primary purpose of the chunk — for example, a boilerplate-only summary of a chunk that contains actual financial results, or a summary that captures only logistical details when the chunk contains substantive business performance content. Score 0.5 if the summary captures the primary financial results or strategic conclusions but omits supporting detail — for example, it states overall revenue growth but omits segment-level breakdown, geographic performance, or key qualifications. Score 1 if the summary captures primary conclusions, key supporting evidence, and significant qualifications. For earnings call transcripts specifically: primary substance consists of financial results (revenue, EPS, margins), segment performance, geographic highlights, and forward guidance. Logistical boilerplate such as operator instructions, IR introductions, and disclaimer language is not primary substance regardless of how much of the chunk it occupies.
3. coherence: Is the summary internally consistent and logically structured, independent of the source? Score 0 if the summary is self-contradictory. Score 0.5 if it is structurally awkward but not misleading. Score 1 if it is internally consistent and well-structured.
4. hallucination: Does the summary reference any entity, figure, or fact not present in the source? Score 0 if at least one reference has no basis in the source. Score 0.5 if a reference is loosely inferable but not explicitly stated. Score 1 if all references are clearly grounded in the source.
5. instruction_following: Does the summary conform to the format and constraints specified in the prompt? Score 0 if a material format constraint is violated. Score 0.5 if the summary partially conforms. Score 1 if the output fully conforms.

Return ONLY a JSON object with this exact schema. No preamble, no explanation outside the JSON:
{{
  "factual_consistency": {{"score": <0, 0.5, or 1>, "justification": "<one sentence>"}},
  "completeness": {{"score": <0, 0.5, or 1>, "justification": "<one sentence>"}},
  "coherence": {{"score": <0, 0.5, or 1>, "justification": "<one sentence>"}},
  "hallucination": {{"score": <0, 0.5, or 1>, "justification": "<one sentence>"}},
  "instruction_following": {{"score": <0, 0.5, or 1>, "justification": "<one sentence>"}}
}}

Source document:
{source_text}

Model summary:
{summary}"""


VALID_SCORES = {0, 0.5, 1}
REQUIRED_CATEGORIES = {
    "factual_consistency",
    "completeness",
    "coherence",
    "hallucination",
    "instruction_following",
}


def validate_response(response: dict) -> bool:
    """
    Validate that the Gemini response conforms to the expected schema.
    Returns True if valid, False otherwise.
    """
    if not isinstance(response, dict):
        return False
    if set(response.keys()) != REQUIRED_CATEGORIES:
        return False
    for category, value in response.items():
        if not isinstance(value, dict):
            return False
        if "score" not in value or "justification" not in value:
            return False
        if value["score"] not in VALID_SCORES:
            return False
        if not isinstance(value["justification"], str) or not value["justification"].strip():
            return False
    return True


def evaluate_pair(source_text: str, summary: str, retries: int = 3) -> dict | None:
    """
    Evaluate a source/summary pair using the Gemini judge.

    Returns a validated evaluation dict on success, None on failure.
    Retries up to `retries` times with exponential backoff on API errors.

    Return format:
    {
        "factual_consistency": {"score": float, "justification": str},
        "completeness": {"score": float, "justification": str},
        "coherence": {"score": float, "justification": str},
        "hallucination": {"score": float, "justification": str},
        "instruction_following": {"score": float, "justification": str},
    }
    """
    prompt = RUBRIC_PROMPT_TEMPLATE.format(source_text=source_text, summary=summary)

    for attempt in range(retries):
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                # Temperature 0 ensures deterministic scoring across runs, which is required for calibration stability.
                # Without it, scores drift between runs and agreement rates fluctuate even when the rubric hasn't changed.
                # Note: full determinism is not guaranteed by the API even at temperature 0.
                config={"temperature": 0}
            )
            raw_text = response.text.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(lines[1:-1])

            parsed = json.loads(raw_text)

            if not validate_response(parsed):
                print(f"Schema validation failed on attempt {attempt+1}. Raw response: {raw_text[:200]}")
                continue

            return parsed

        except ClientError as e:
            print(f"API error on attempt {attempt + 1}: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error on attempt {attempt+1}: {e}")
        except Exception as e:
            print(f"API error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                wait = 30 if ("429" in str(e) or "503" in str(e)) else 2 ** attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    print("All retry attempts exhausted. Returning None.")
    return None
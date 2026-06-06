"""
generate_summaries.py

Loads earnings call transcripts from the kurry/sp500_earnings_transcripts
HuggingFace dataset and runs them through DistilBART to generate summaries
for manual scoring.

Output: data/summaries_raw.csv
Columns: doc_id, company, symbol, year, quarter, chunk_index, source_text, summary

Run this locally once to generate the validation set inputs.
Requires: pip install transformers==4.44.0 torch datasets pandas
"""

import hashlib
import pandas as pd
from datasets import load_dataset
from transformers import pipeline

# Config

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50

# Number of transcripts to sample — 10 transcripts x ~5 chunks each = ~50 pairs
SAMPLE_SIZE = 10

# Cap chunks per transcript to keep variety across companies
CHUNKS_PER_TRANSCRIPT = 5

OUTPUT_PATH = "data/summaries_raw.csv"


def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Split text into chunks of approximately chunk_tokens words with overlap."""
    words = text.split()
    chunk_words = int(chunk_tokens / 1.3)   # ≈ 394 words per chunk
    overlap_words = int(overlap_tokens / 1.3)  # ≈ 38 words overlap

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_words - overlap_words

    return chunks


def load_summarizer():
    """Load DistilBART summarization pipeline (sshleifer/distilbart-cnn-12-6)."""
    print("Loading DistilBART...")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # CPU; set to 0 if GPU
    )
    print("Model loaded.")
    return summarizer


def summarize_chunk(summarizer, text: str) -> str | None:
    """Run a single chunk through DistilBART. Returns summary or None on error."""
    try:
        # max_length=130, min_length=30 are standard DistilBART settings
        # truncation=True handles chunks that slightly exceed the 1024 token limit
        result = summarizer(
            text,
            max_length=130,
            min_length=30,
            truncation=True,
            do_sample=False
        )
        return result[0]["summary_text"]
    except Exception as e:
        print(f"Summarization error: {e}")
        return None


def make_doc_id(symbol: str, year: int, quarter: int, chunk_index: int) -> str:
    """Generate a stable doc_id from symbol, year, quarter, and chunk index."""
    raw = f"{symbol}_{year}_Q{quarter}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def main():
    print("Loading dataset...")
    dataset = load_dataset("kurry/sp500_earnings_transcripts", split="train")
    print(f"Dataset loaded. {len(dataset)} transcripts available.")

    # Sample a fixed subset — skip transcripts with no content
    sample = []
    for item in dataset:
        if item.get("content", "").strip():
            sample.append(item)
        if len(sample) >= SAMPLE_SIZE:
            break

    print(f"Sampled {len(sample)} transcripts with content.")

    summarizer = load_summarizer()
    records = []

    for idx, item in enumerate(sample):
        symbol = item.get("symbol", f"unknown_{idx}")
        company = item.get("company_name", symbol)
        year = item.get("year", "")
        quarter = item.get("quarter", "")
        text = item["content"]

        print(f"\n{symbol} — {company} Q{quarter} {year}")
        chunks = chunk_text(text)
        print(f"Extracted {len(chunks)} chunks.")

        chunks_used = 0
        for i, chunk in enumerate(chunks):
            if chunks_used >= CHUNKS_PER_TRANSCRIPT:
                break

            print(f"Summarizing chunk {i + 1}/{len(chunks)}...", end=" ")
            summary = summarize_chunk(summarizer, chunk)

            if summary:
                doc_id = make_doc_id(symbol, year, quarter, i)
                records.append({
                    "doc_id": doc_id,
                    "company": company,
                    "symbol": symbol,
                    "year": year,
                    "quarter": quarter,
                    "chunk_index": i,
                    "source_text": chunk,
                    "summary": summary,
                })
                chunks_used += 1
                print("done.")
            else:
                print("failed, skipping.")

        print(f"Collected {chunks_used} chunks for {symbol}.")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. {len(df)} source/summary pairs saved to {OUTPUT_PATH}.")

if __name__ == "__main__":
    main()
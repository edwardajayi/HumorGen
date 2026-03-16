#!/usr/bin/env python3
"""
Multi-Model Joke Generation Script for Evaluation
Appends Llama 3.3 and Kimi jokes to benchmark_results_v3_updated.json
"""

import os
import json
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Same prompt as src/run_comparison.py (copied to avoid GPU imports)
SYSTEM_PROMPT = """You generate original jokes based on the given prompt.
Strong jokes often rely on surprise or unexpected connections.
Different humor styles are allowed.

STRICT RULES:
1. ENGLISH ONLY - No other languages, no Unicode characters outside ASCII
2. MAXIMUM 900 CHARACTERS - Ensure the joke is complete and ends properly.
3. REPETITION - Avoid unnecessary repetition.
4. SELF-CONTAINED - The joke must make sense without additional context

QUALITY STANDARDS:
 - Use surprise, wordplay, or unexpected connections
 - Avoid obvious or predictable punchlines
 - Make it quotable and shareable

Output ONLY the joke text, nothing else."""

# --- Config ---
MODELS = {
    "llama_3_3": "llama-3.3-70b-versatile",
    "kimi_k2": "moonshotai/kimi-k2-instruct-0905",
}
_REPO_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = str(_REPO_ROOT / "results" / "benchmark_results_v3_updated.json")

def get_model(name):
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=name, temperature=0.7, max_retries=3)

def generate(model, prompt):
    try:
        res = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f"Create a joke about: {prompt}")])
        return res.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def main():
    data = [json.loads(line) for line in open(FILE_PATH) if line.strip()]
    print(f"Loaded {len(data)} rows")

    models = {k: get_model(v) for k, v in MODELS.items()}
    print(f"Models ready: {list(models.keys())}")

    temp = FILE_PATH + ".tmp"
    with open(temp, 'w') as f:
        for row in tqdm(data, desc="Generating"):
            for key, model in models.items():
                col = f"{key}_joke"
                if col not in row or not row[col] or "ERROR" in row[col]:
                    row[col] = generate(model, row['prompt'])
            f.write(json.dumps(row) + "\n")
            f.flush()
    
    os.replace(temp, FILE_PATH)
    print(f"Done! Saved to {FILE_PATH}")

if __name__ == "__main__":
    main()

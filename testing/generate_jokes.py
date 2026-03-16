#!/usr/bin/env python3
"""
Generate Jokes from Trained Models + Groq API for Evaluation

Models: base Qwen 7B, DPO 7B, GRPO 7B, Qwen3-32B (Groq), Kimi-K2 (Groq), GPT-OSS-120B (Groq)
Output: JSONL with id, headline, base_joke, dpo_joke, grpo_joke, qwen3_32b_joke, kimi_k2_joke, gpt_oss_joke

Usage:
    python generate_jokes.py --num 20
    python generate_jokes.py --num 20 --local-only
    python generate_jokes.py --num 20 --api-only
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import logging
import torch
from tqdm import tqdm
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Paths (repo root = parent of testing/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = _PROJECT_ROOT
TEST_DATA = os.path.join(PROJECT_ROOT, "testing", "test_data.tsv")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "testing", "generated_jokes.jsonl")

BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
DPO_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_DPO_7B", "checkpoint-1550")
GRPO_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_GRPO_7B", "checkpoint-6250")

# Import model utils for Unsloth loading
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "codes"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
try:
    from src.model_utils import get_model_unsloth
except ImportError:
    from model_utils import get_model_unsloth

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# =====================================================================
# Prompts
# =====================================================================
SYSTEM_PROMPT = "You are a joke generator. Given a headline or topic, generate a funny joke. Output ONLY the joke text. No thinking tags, no reasoning, no explanation, no extra words."


# =====================================================================
# Groq API Key Rotation (keys from GROQ_API_KEY or comma-separated GROQ_API_KEYS)
# =====================================================================
def _get_groq_api_keys():
    raw = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError("Set GROQ_API_KEY or GROQ_API_KEYS in environment for Groq models.")
    return keys

GROQ_MODELS = {
    "qwen3_32b":  "qwen/qwen3-32b",
    "kimi_k2":    "moonshotai/kimi-k2-instruct-0905",
    "gpt_oss":    "openai/gpt-oss-120b",
}

OPENAI_MODELS = {
    "gpt4o": "gpt-5",
}


class GroqKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_idx = 0
        self.clients = [Groq(api_key=key) for key in keys]

    def get_client(self):
        client = self.clients[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.clients)
        return client


# =====================================================================
# Data Loading
# =====================================================================
def load_headlines(path, num):
    """Load N headlines from the test TSV."""
    headlines = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            headlines.append({
                "id": row["id"].strip(),
                "headline": row["headline"].strip()
            })
            if len(headlines) >= num:
                break
    return headlines


# =====================================================================
# Local Model Generation
# =====================================================================
def generate_local_joke(model, tokenizer, headline):
    """Generate a joke using a local model."""
    prompt = f'Write a funny joke about: {headline}'
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response


def load_model_with_adapter(adapter_path):
    """Load a LoRA adapter on top of the base model."""
    from peft import PeftModel
    from unsloth import FastLanguageModel

    logging.info(f"  Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    logging.info(f"  Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# Groq API Generation
# =====================================================================
def generate_groq_joke(key_manager, model_name, headline):
    """Generate a joke via Groq API with key rotation on rate limit."""
    prompt = f'Write a funny joke about: {headline}'
    max_retries = len(key_manager.keys)

    for attempt in range(max_retries):
        client = key_manager.get_client()
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "exhausted" in err_msg or "429" in err_msg:
                logging.warning(f"Rate limit on key {key_manager.current_idx}, rotating...")
                time.sleep(2)
                continue
            else:
                logging.error(f"Error {model_name}: {e}")
                return f"[ERROR: {e}]"

    logging.error(f"All keys exhausted for {model_name}.")
    return "[ERROR: All keys exhausted]"


def generate_openai_joke(model_name, headline):
    """Generate a joke via OpenAI API (via CMU AI Gateway)."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        return "[ERROR: No OpenAI API key]"
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = f'Write a funny joke about: {headline}'
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error OpenAI {model_name}: {e}")
        return f"[ERROR: {e}]"


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate jokes from trained + API models")
    parser.add_argument("--num", type=int, default=20, help="Number of headlines to evaluate")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--local-only", action="store_true", help="Only run local models (GPU required)")
    parser.add_argument("--api-only", action="store_true", help="Only run Groq API models (no GPU needed)")
    args = parser.parse_args()

    # Load headlines
    headlines = load_headlines(TEST_DATA, args.num)
    logging.info(f"Loaded {len(headlines)} headlines for evaluation")

    # Load existing results if resuming
    results = {}
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    results[e["id"]] = e
        logging.info(f"Resuming: {len(results)} entries already have some results.")

    # Initialize missing entries
    for h in headlines:
        if h["id"] not in results:
            results[h["id"]] = {"id": h["id"], "headline": h["headline"]}

    run_local = not args.api_only
    run_api = not args.local_only

    # ========================
    # LOCAL MODELS (GPU)
    # ========================
    if run_local:
        local_models = [
            ("base_joke", None, "BASE MODEL"),
            ("dpo_joke", DPO_ADAPTER, "DPO MODEL"),
            ("grpo_joke", GRPO_ADAPTER, "GRPO MODEL"),
        ]

        for i, (col, adapter, label) in enumerate(local_models):
            # Skip if all headlines already have this column
            remaining = [h for h in headlines if col not in results[h["id"]] or not results[h["id"]][col]]
            if not remaining:
                logging.info(f"[{i+1}/3] {label}: All {len(headlines)} already generated. Skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"[{i+1}/3] Generating with {label} ({len(remaining)} headlines)")
            print(f"{'='*60}")

            if adapter:
                model, tokenizer = load_model_with_adapter(adapter)
            else:
                from unsloth import FastLanguageModel
                model, tokenizer = get_model_unsloth(BASE_MODEL, max_seq_length=4096, load_in_4bit=True)
                FastLanguageModel.for_inference(model)

            for h in tqdm(remaining, desc=label):
                joke = generate_local_joke(model, tokenizer, h["headline"])
                results[h["id"]][col] = joke

            unload_model(model, tokenizer)

            # Save after each model (checkpoint)
            _save_results(args.output, headlines, results)

    # ========================
    # GROQ API MODELS
    # ========================
    if run_api:
        key_manager = GroqKeyManager(_get_groq_api_keys())
        groq_items = list(GROQ_MODELS.items())

        for i, (col_key, model_name) in enumerate(groq_items):
            col = f"{col_key}_joke"
            remaining = [h for h in headlines if col not in results[h["id"]] or not results[h["id"]][col] or "ERROR" in str(results[h["id"]].get(col, ""))]
            if not remaining:
                logging.info(f"[API {i+1}/3] {model_name}: All done. Skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"[API {i+1}/3] Generating with {model_name} ({len(remaining)} headlines)")
            print(f"{'='*60}")

            for h in tqdm(remaining, desc=model_name):
                joke = generate_groq_joke(key_manager, model_name, h["headline"])
                results[h["id"]][col] = joke
                time.sleep(1)  # rate limit buffer

            # Save after each model
            _save_results(args.output, headlines, results)

        # ========================
        # OPENAI API MODELS
        # ========================
        for i, (col_key, model_name) in enumerate(OPENAI_MODELS.items()):
            col = f"{col_key}_joke"
            remaining = [h for h in headlines if col not in results[h["id"]] or not results[h["id"]][col] or "ERROR" in str(results[h["id"]].get(col, ""))]
            if not remaining:
                logging.info(f"[OpenAI {i+1}] {model_name}: All done. Skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"[OpenAI {i+1}] Generating with {model_name} ({len(remaining)} headlines)")
            print(f"{'='*60}")

            for h in tqdm(remaining, desc=model_name):
                joke = generate_openai_joke(model_name, h["headline"])
                results[h["id"]][col] = joke
                time.sleep(1)

            _save_results(args.output, headlines, results)

    # Final save
    _save_results(args.output, headlines, results)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DONE! {len(headlines)} headlines × 6 models")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    # Print a sample
    sample = results[headlines[0]["id"]]
    print(f"\n--- Sample ({sample['id']}: {sample['headline'][:60]}...) ---")
    for col in ["base_joke", "dpo_joke", "grpo_joke", "qwen3_32b_joke", "kimi_k2_joke", "gpt_oss_joke", "gpt4o_joke"]:
        val = sample.get(col, "N/A")
        print(f"  {col}: {str(val)[:120]}...")


def _save_results(path, headlines, results):
    """Save results preserving headline order."""
    with open(path, 'w') as f:
        for h in headlines:
            f.write(json.dumps(results[h["id"]]) + '\n')
    logging.info(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    main()

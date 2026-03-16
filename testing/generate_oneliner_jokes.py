#!/usr/bin/env python3
"""
Case C (Issue 4): Short-Form One-Liners.
Zero-shot generation of one-liner jokes using HumorGen-SFT-7B and HumorGen-DPO-7B.

PROMPTS (tighter, less repetitive; + post-processing for strict format):
  System: "You generate funny one-liner jokes. Given a topic, write exactly one short,
           punchy sentence under 18 words. Do not include a second sentence, explanation,
           labels, quotation marks, or extra text. Output only the joke."
  User:   "Write a one-liner joke about {topic}."
  Decoding: temperature=0.6, max_new_tokens=50. Optional seed for reproducibility.
  Post-process: strip labels/quotes, first sentence only. No truncation (never cut a joke).
  Rows where generation failed (after retries) are excluded from the output JSONL so the appendix only shows successful one-liners.
"""
import csv
import gc
import json
import os
import re
import torch
from tqdm import tqdm

# Validation / post-processing (no truncation: we never cut a joke; first sentence only, any length)
MAX_SENTENCES = 1
MAX_RETRIES = 2  # regenerate up to 2 times if output invalid (reduces "(generation failed)")
FAILED_PLACEHOLDER = "(generation failed)"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Shared OOD topics so we can compare long vs short on same prompts (see OOD_GENERATION_PLAN.md)
ONELINER_DATA = os.path.join(PROJECT_ROOT, "data", "datasets", "ood_prompts.tsv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "testing", "ONELINER_JOKES.jsonl")

BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
SFT_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_SFT_7B", "checkpoint-900")
DPO_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_DPO_7B", "checkpoint-1550")

# Prompt asks for one short sentence; we keep first sentence only and never truncate.
SYSTEM_PROMPT = (
    "You generate funny one-liner jokes. "
    "Given a topic, write exactly one short, punchy sentence under 18 words. "
    "Do not include a second sentence, explanation, labels, quotation marks, or extra text. "
    "Output only the joke."
)


def clean_oneliner(text):
    """Strip labels, surrounding quotes, and take first sentence only."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove leading label (e.g. "Joke:", "One-liner:", "A:")
    text = re.sub(r"^(Joke|One-liner|One liner|A)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # Strip surrounding quotes (single or double)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    # Take first sentence only (split on . ! ?)
    for sep in [". ", "! ", "? "]:
        if sep in text:
            idx = text.index(sep) + 1
            text = text[:idx].strip()
            break
    return text


def is_valid_oneliner(text):
    """True if single sentence and no newline. No word limit — we never truncate."""
    if not text or "\n" in text:
        return False
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) <= MAX_SENTENCES


def load_prompts(path):
    prompts = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prompts.append({"id": row["id"].strip(), "topic": row["topic"].strip()})
    return prompts


def load_model_with_adapter(adapter_path):
    from peft import PeftModel
    from unsloth import FastLanguageModel

    print(f"Loading adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_oneliner(model, tokenizer, topic, seed=None):
    """Generate one one-liner; clean and validate; optionally retry once if invalid."""
    user_prompt = f"Write a one-liner joke about {topic}."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    for attempt in range(MAX_RETRIES + 1):
        if seed is not None:
            torch.manual_seed(seed + attempt)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        response = clean_oneliner(response)
        if not response:
            continue
        if is_valid_oneliner(response):
            return response  # first sentence only, any length — never truncated

    return response if response else FAILED_PLACEHOLDER


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate OOD one-liner jokes (SFT + DPO).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print(f"Loading {ONELINER_DATA}")
    prompts = load_prompts(ONELINER_DATA)
    print(f"Loaded {len(prompts)} topics.")
    print(f"Prompt: System = (tighter, under 18 words); User = 'Write a one-liner joke about {{topic}}.'")
    print("Post-process: strip labels/quotes, first sentence only. No truncation (never cut a joke).")

    results = {p["id"]: dict(p) for p in prompts}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    skipped = 0
    with open(OUTPUT_PATH, "w") as f:
        # 1. SFT
        print("\nStarting SFT generation (one-liners)...")
        model, tokenizer = load_model_with_adapter(SFT_ADAPTER)
        for p_id in tqdm(results, desc="SFT one-liners"):
            results[p_id]["sft_joke"] = generate_oneliner(
                model, tokenizer, results[p_id]["topic"], seed=args.seed
            )
        unload_model(model, tokenizer)

        # 2. DPO — write each row as soon as DPO for that topic is done (so you see progress)
        print("\nStarting DPO generation (one-liners)...")
        model, tokenizer = load_model_with_adapter(DPO_ADAPTER)
        for p_id in tqdm(results, desc="DPO one-liners"):
            results[p_id]["dpo_joke"] = generate_oneliner(
                model, tokenizer, results[p_id]["topic"], seed=args.seed
            )
            res = results[p_id]
            if res.get("sft_joke") == FAILED_PLACEHOLDER or res.get("dpo_joke") == FAILED_PLACEHOLDER:
                skipped += 1
                print(f"  [Skip] {res.get('id', '?')}: generation failed (excluded from output).")
                continue
            f.write(json.dumps(res) + "\n")
            f.flush()
        unload_model(model, tokenizer)

    if skipped:
        print(f"Skipped {skipped} row(s) with failed generations.")
    print(f"\nSaved results to {OUTPUT_PATH} (one row per topic as completed).")
    print("Done.")


if __name__ == "__main__":
    main()

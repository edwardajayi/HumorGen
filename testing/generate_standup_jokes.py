#!/usr/bin/env python3
"""
Case A (Issue 4): Long-Form Comedy.
Zero-shot generation of stand-up style long-form jokes using HumorGen-SFT-7B and HumorGen-DPO-7B.
Prompt: 3–5 sentences, setup + punchline, no labels/quotes. Validation: min 40 words, max 100; retry if too short.
"""
import csv
import gc
import json
import os
import re
import torch
from tqdm import tqdm

# Validation: long-form = 3–5 sentences, ~40–100 words. Retry once if too short.
MIN_LONG_WORDS = 40
MIN_SENTENCES = 2
MAX_LONG_WORDS = 100  # cap so output doesn't ramble
MAX_RETRIES = 1

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Shared OOD topics so we can compare long vs short on same prompts (see OOD_GENERATION_PLAN.md)
STANDUP_DATA = os.path.join(PROJECT_ROOT, "data", "datasets", "ood_prompts.tsv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "testing", "STANDUP_JOKES.jsonl")

BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
SFT_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_SFT_7B", "checkpoint-900")
DPO_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_DPO_7B", "checkpoint-1550")

# Stand-up style: 3–5 sentences, setup + punchline, concise. No labels/quotes/rambling.
SYSTEM_PROMPT = (
    "You generate funny stand-up style jokes. "
    "Given a topic, write one short comedic bit in 3 to 5 sentences. "
    "It should have a clear setup and end with a punchline or twist. "
    "Keep it concise, natural, and funny, not rambling. "
    "Do not include labels, quotation marks, thinking tags, reasoning, or extra text. "
    "Output only the joke."
)


def clean_longform(text):
    """Strip labels (e.g. 'Joke:', 'Here's a joke:') and surrounding quotes."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove leading label
    text = re.sub(
        r"^(Here'?s a joke|Joke|Stand-?up joke|Bit)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    return text


def truncate_to_max_words(text, max_words=MAX_LONG_WORDS):
    """Truncate to max_words at a sentence boundary if possible."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Cut at last sentence end
    for sep in [". ", "! ", "? "]:
        last = truncated.rfind(sep)
        if last != -1:
            return truncated[: last + 1].strip()
    return truncated


def count_sentences(text):
    """Approximate sentence count (split on . ! ?)."""
    if not text or not text.strip():
        return 0
    parts = re.split(r"[.!?]+", text)
    return len([p for p in parts if p.strip()])


def generate_bit(model, tokenizer, topic, seed=None):
    """Generate one long-form joke; clean, validate (40–100 words, 2+ sentences), retry if too short."""
    user_prompt = f"Write a funny stand-up style joke about {topic}."
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
                max_new_tokens=160,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        response = clean_longform(response)
        if not response:
            continue
        word_count = len(response.split())
        sent_count = count_sentences(response)
        if word_count > MAX_LONG_WORDS:
            response = truncate_to_max_words(response)
            word_count = len(response.split())
        if word_count >= MIN_LONG_WORDS and sent_count >= MIN_SENTENCES:
            return response
        # Too short or too few sentences; retry once

    return response


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


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate OOD long-form jokes (SFT + DPO).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print(f"Loading {STANDUP_DATA}")
    prompts = load_prompts(STANDUP_DATA)
    print(f"Loaded {len(prompts)} topics.")
    print(f"Prompt: stand-up style, 3–5 sentences, setup + punchline. Validation: {MIN_LONG_WORDS}–{MAX_LONG_WORDS} words, ≥{MIN_SENTENCES} sentences, max_new_tokens=160.")

    results = {p["id"]: dict(p) for p in prompts}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        # 1. SFT
        print("\nStarting SFT generation (long-form)...")
        model, tokenizer = load_model_with_adapter(SFT_ADAPTER)
        for p_id in tqdm(results, desc="SFT long-form"):
            results[p_id]["sft_joke"] = generate_bit(
                model, tokenizer, results[p_id]["topic"], seed=args.seed
            )
        unload_model(model, tokenizer)

        # 2. DPO — write each row as soon as DPO for that topic is done (so you see progress)
        print("\nStarting DPO generation (long-form)...")
        model, tokenizer = load_model_with_adapter(DPO_ADAPTER)
        for p_id in tqdm(results, desc="DPO long-form"):
            results[p_id]["dpo_joke"] = generate_bit(
                model, tokenizer, results[p_id]["topic"], seed=args.seed
            )
            f.write(json.dumps(results[p_id]) + "\n")
            f.flush()
        unload_model(model, tokenizer)

    print(f"\nSaved results to {OUTPUT_PATH} (one row per topic as completed).")
    print("Done.")


if __name__ == "__main__":
    main()

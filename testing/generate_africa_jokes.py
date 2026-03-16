#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import sys
import torch
from tqdm import tqdm

# Paths (repo root = parent of testing/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AFRICA_DATA = os.path.join(PROJECT_ROOT, "data", "datasets", "africa_headlines.tsv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "testing", "AFRICA_JOKES.jsonl")

BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
SFT_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_SFT_7B", "checkpoint-900")
DPO_ADAPTER = os.path.join(PROJECT_ROOT, "models", "HumorGen_DPO_7B", "checkpoint-1550")

SYSTEM_PROMPT = "You are a joke generator. Given a headline or topic, generate a funny joke. Output ONLY the joke text. No thinking tags, no reasoning, no explanation, no extra words."

def load_headlines(path):
    headlines = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            headlines.append({"id": row["id"].strip(), "headline": row["headline"].strip()})
    return headlines

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

def generate_joke(model, tokenizer, headline):
    prompt = f'Write a funny joke about: {headline}'
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
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

def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    print(f"Loading {AFRICA_DATA}")
    headlines = load_headlines(AFRICA_DATA)
    print(f"Loaded {len(headlines)} headlines.")

    results = {h["id"]: h for h in headlines}

    # 1. SFT Generation
    print("\nStarting SFT generation...")
    model, tokenizer = load_model_with_adapter(SFT_ADAPTER)
    for h_id in tqdm(results, desc="SFT Jokes"):
        results[h_id]["sft_joke"] = generate_joke(model, tokenizer, results[h_id]["headline"])
    unload_model(model, tokenizer)

    # 2. DPO Generation
    print("\nStarting DPO generation...")
    model, tokenizer = load_model_with_adapter(DPO_ADAPTER)
    for h_id in tqdm(results, desc="DPO Jokes"):
        results[h_id]["dpo_joke"] = generate_joke(model, tokenizer, results[h_id]["headline"])
    unload_model(model, tokenizer)

    print(f"\nSaving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        for res in results.values():
            f.write(json.dumps(res) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()

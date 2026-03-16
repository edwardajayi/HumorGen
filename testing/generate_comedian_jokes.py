#!/usr/bin/env python3
import json
import os
import torch
import gc

# Paths (repo root = parent of testing/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA = os.path.join(PROJECT_ROOT, "testing", "test_data.tsv")
OUTPUT_TEMP = os.path.join(PROJECT_ROOT, "testing", "comedian_jokes.jsonl")

ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "HumorGen-Com_7B")

SYSTEM_PROMPT = """You generate original jokes based on the given prompt.
Strong jokes often rely on surprise or unexpected connections.
Different humor styles are allowed."""

def load_headlines(path, num):
    import csv
    headlines = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            headlines.append({"id": row["id"].strip(), "headline": row["headline"].strip()})
            if len(headlines) >= num: break
    return headlines

def generate_joke(model, tokenizer, headline):
    prompt = f'Write a funny joke about: {headline}'
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

def main():
    import argparse
    from tqdm import tqdm
    import gc
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50) # Assuming 50 is the standard evaluation size based on previous logs 
    args = parser.parse_args()

    headlines = load_headlines(TEST_DATA, args.num)
    results = {}
    
    if os.path.exists(OUTPUT_TEMP):
        with open(OUTPUT_TEMP, 'r') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    results[d["id"]] = d
        print(f"Resuming: {len(results)} existing entries.")

    for h in headlines:
        if h["id"] not in results:
            results[h["id"]] = {"id": h["id"], "headline": h["headline"]}

    remaining = [h for h in headlines if "comedian_joke" not in results[h["id"]] or not results[h["id"]]["comedian_joke"]]
    
    if remaining:
        print(f"Generating with Comedian Model... ({len(remaining)} remaining)")
        
        # Load the base model + the new adapter
        from unsloth import FastLanguageModel
        from peft import PeftModel
        
        # We know from model_training.md that it started from HumorGen_SFT_7B
        BASE_MODEL = os.path.join(PROJECT_ROOT, "models", "HumorGen_SFT_7B")
        
        # Check if Base Model exists, else use the raw absolute path or fallback to base
        if not os.path.exists(BASE_MODEL):
            # If the SFT model was overwritten/merged, we just load the unsloth base and apply the adapter directly
            BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
            print(f"SFT model path not found, falling back to base: {BASE_MODEL}")
            
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=1024, load_in_4bit=True)
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        FastLanguageModel.for_inference(model)

        for h in tqdm(remaining):
            joke = generate_joke(model, tokenizer, h["headline"])
            results[h["id"]]["comedian_joke"] = joke
            
        del model; del tokenizer; gc.collect(); torch.cuda.empty_cache()
    else:
        print("Comedian Baseline: All generated. Skipping.")

    with open(OUTPUT_TEMP, 'w') as f:
        for h in headlines:
            f.write(json.dumps(results[h["id"]]) + '\n')
    print(f"Done. Results saved to {OUTPUT_TEMP}")

if __name__ == "__main__":
    main()

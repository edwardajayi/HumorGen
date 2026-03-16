"""
HumorGen V3 Unified Evaluation
Merges V3 Models (SFT, DPO, GRPO) with Baselines (Qwen 7B, 32B, GPT).
"""

import os
import sys
import json
import torch
import gc
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

# Load env immediately for LTI Gateway
load_dotenv()

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.model_utils import (
    get_model_robust, 
    get_model_unsloth,
    get_model_4bit,
    cleanup_gpu,
    nuke_gpu
)
from src.run_comparison import generate_gpt, generate_joke
from peft import PeftModel

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = _REPO_ROOT
OUTPUT_FILE = BASE_DIR / "results" / "benchmark_results_v3_full.json"
TEST_PROMPTS_FILE = BASE_DIR / "results" / "test_prompts_200.jsonl"

# Define Models
MODELS = {
    # --- V3 Models (Adapters) ---
    "SFT_V3": {
        "path": f"{BASE_DIR}/models/HumorGen_SFT_7B",
        "type": "adapter",
        "base": "Qwen/Qwen2.5-7B-Instruct"
    },
    "DPO_V3": {
        "path": f"{BASE_DIR}/models/HumorGen_DPO_7B",
        "type": "adapter",
        "base": "Qwen/Qwen2.5-7B-Instruct"
    },
    "GRPO_V3": {
        "path": f"{BASE_DIR}/models/HumorGen_GRPO_7B",
        "type": "adapter",
        "base": "Qwen/Qwen2.5-7B-Instruct"
    },
    
    # --- Baselines ---
    "Qwen_7B_Instruct": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        # Use 4-bit to match baselines validation
        "type": "4bit",
        "base": None
    },
    "Qwen_32B_Instruct": {
        "path": "Qwen/Qwen2.5-32B-Instruct",
        # Use Unsloth 4-bit for 32B to ensure it fits comfortably and fast
        "type": "unsloth_4bit", 
        "base": None
    },
    "GPT-4o-mini": {
        "path": "gpt-4o-mini-2024-07-18",
        "type": "api",
        "base": None
    }
}

def load_prompts():
    prompts = []
    if TEST_PROMPTS_FILE.exists():
        print(f"Loading prompts from {TEST_PROMPTS_FILE}...")
        with open(TEST_PROMPTS_FILE, 'r') as f:
             for line in f:
                 if line.strip():
                     prompts.append(json.loads(line))
    else:
        raise FileNotFoundError(f"Test prompts file not found at {TEST_PROMPTS_FILE}. Please ensure the file exists.")
        
    # Standardize: Ensure 'id' and 'prompt' keys
    final_prompts = []
    for i, p in enumerate(prompts):
        p_text = p.get('prompt', '') or p.get('text', '')
        p_id = p.get('id', f"prompt_{i}")
        if p_text:
            final_prompts.append({"id": p_id, "prompt": p_text})
            
    print(f"Loaded {len(final_prompts)} prompts.")
    
    # Limit to first 100 prompts only
    final_prompts = final_prompts[:100]
    print(f"Using first {len(final_prompts)} prompts for evaluation.")
    
    return final_prompts

def load_results():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]
    return []

def save_full_results(results):
    with open(OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

def main():
    print("=== HumorGen V3 Unified Evaluation ===")
    
    # 1. Load Data
    prompts = load_prompts()
    existing_results = load_results()
    
    # Map existing results by ID
    results_map = {r['id']: r for r in existing_results}
    
    # Initialize missing
    for p in prompts:
        if p['id'] not in results_map:
            results_map[p['id']] = p.copy()
    
    # Convert back to list for processing
    results_list = list(results_map.values())
    
    # 2. Iterate Models
    for model_key, config in MODELS.items():
        print(f"\n--- Processing {model_key} ---")
        
        # Check if already done for all prompts
        col_name = f"{model_key}_joke"
        done_count = sum(1 for r in results_list if col_name in r and r[col_name])
        if done_count == len(results_list):
            print(f"All {len(results_list)} prompts already have {model_key} generations. Skipping.")
            continue
            
        print(f"Generating for {len(results_list) - done_count} prompts...")
        
        # Load Model Strategy
        model = None
        tokenizer = None
        
        try:
            if config["type"] == "api":
                # API Model (GPT)
                for item in tqdm(results_list):
                    if col_name in item: continue
                    try:
                        resp = generate_gpt(item['prompt'], config["path"])
                        item[col_name] = resp
                    except Exception as e:
                        print(f"API Error: {e}")
                        item[col_name] = "ERROR"
                    save_full_results(results_list) # Save incrementally
                    
            elif config["type"] == "unsloth_4bit":
                # Unsloth Model (Qwen 32B)
                model, tokenizer = get_model_unsloth(config["path"], load_in_4bit=True)
                tokenizer.pad_token = tokenizer.eos_token
                
                for item in tqdm(results_list):
                    if col_name in item: continue
                    resp = generate_joke(model, tokenizer, item['prompt'])
                    item[col_name] = resp
                    save_full_results(results_list)
                    
            elif config["type"] == "4bit":
                # Standard HF 4-bit (Qwen 7B)
                model, tokenizer = get_model_4bit(config["path"], alias="qwen7b_4bit")
                tokenizer.pad_token = tokenizer.eos_token
                
                for item in tqdm(results_list):
                    if col_name in item: continue
                    resp = generate_joke(model, tokenizer, item['prompt'])
                    item[col_name] = resp
                    save_full_results(results_list)

            elif config["type"] in ["adapter", "base"]:
                # HF Model (SFT/DPO/GRPO/Base7B)
                base_name = config["base"] if config["type"] == "adapter" else config["path"]
                
                # Load Base (BF16)
                model, tokenizer = get_model_robust(base_name, alias="base_qwen7b")
                tokenizer.pad_token = tokenizer.eos_token
                
                if config["type"] == "adapter":
                    # Load Adapter
                    if os.path.exists(config["path"]):
                        print(f"Loading adapter: {config['path']}")
                        model = PeftModel.from_pretrained(model, config["path"])
                    else:
                        print(f"ERROR: Adapter path {config['path']} not found. Skipping.")
                        continue
                
                # Generate
                for item in tqdm(results_list):
                    if col_name in item: continue
                    resp = generate_joke(model, tokenizer, item['prompt'])
                    item[col_name] = resp
                    save_full_results(results_list)
            
        except Exception as e:
            print(f"CRITICAL ERROR running {model_key}: {e}")
        
        finally:
            # Cleanup
            if model is not None:
                # If adapter, unload adapter first? No, just delete model
                pass
            cleanup_gpu(model, tokenizer, alias="base_qwen7b" if config["type"]=="adapter" else None)
            model = None
            tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            
    print(f"\nAll Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

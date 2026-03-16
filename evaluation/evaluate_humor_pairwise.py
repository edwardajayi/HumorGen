#!/usr/bin/env python3
"""
Pairwise Humor Evaluation Script
--------------------------------
Implements "Structured Pairwise" evaluation based on best practices:
1. Pairwise comparison (Subjective task)
2. Chain-of-Thought reasoning (Mandatory)
3. Position Bias Mitigation (Swap orders)
4. JSON Structured Output
5. Constrained Feature Taxonomy
6. Explicit Winner Storage, Robust Features, and API Key Rotation

Judges: 
- OpenAI: gpt-4o-mini
- Groq: llama-3.3-70b, qwen-32b (Qwen3)
"""

import os
import json
import random
import time
import argparse
from itertools import combinations
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Import LangChain components
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: Missing dependencies. Run: pip install langchain-groq langchain-openai langchain-core pydantic")
    exit(1)

# Load env from parent dir
load_dotenv("../.env")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_MAP = {
    # The Selected Judges
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "llama-3.3": "llama-3.3-70b-versatile",
    "qwen-32b": "qwen/qwen3-32b"
}

# The Research-Backed Prompt
PAIRWISE_SYSTEM_PROMPT = """You are a professional comedy critic. Your task is to compare two jokes and decide which is funnier.

You must ignore the length of the joke. Do NOT penalize long jokes. A long, narrative joke with a good payoff is just as valuable as a short, punchy one.
You must be objective and look for:
1. Surprise / Incongruity
2. Cleverness / Wordplay
3. Narrative structure (if applicable)

Output your analysis in JSON format."""

PAIRWISE_USER_TEMPLATE = """Compare these two jokes based on the prompt: "{original_prompt}"

JOKE A:
{joke_a}

JOKE B:
{joke_b}

Step-by-Step Analysis:
1. Analyze Joke A's technique and flaws.
2. Analyze Joke B's technique and flaws.
3. Compare them directly (ignoring length).
4. Decide the winner.

Return JSON exactly like this:
{{
  "reasoning": "Joke A uses a clever pun on X, while Joke B is too literal...",
  "decision": "A" or "B" or "TIE",
  "winner_features": [SELECT FROM: {allowed_features}],
  "loser_features": ["cliché", "too_long", "confusing", "weak_punchline", "offensive"]
}}"""

# Constrained Feature List for Clean Analysis
ALLOWED_FEATURES = [
    "incongruity", "wordplay", "timing", "absurdity", "surprise", 
    "irony", "dark_humor", "observational", "sarcasm", "narrative"
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PairwiseResult(BaseModel):
    judge_model: str
    prompt_id: str
    original_prompt: str
    model_a: str
    model_b: str
    joke_a: str
    joke_b: str
    swapped: bool # True if we swapped A/B positions for bias check
    decision: str
    reasoning: str
    timestamp: float

# =============================================================================
# UTILS
# =============================================================================

def get_judge_model(model_key: str):
    """Initialize the LLM judge with fresh env vars."""
    # Reload env to handle rotated keys before init
    load_dotenv("../.env", override=True)
    
    if "gpt" in model_key and "openai" not in model_key and "oss" not in model_key:
        # Standard OpenAI models
        return ChatOpenAI(
            model=MODEL_MAP[model_key], 
            temperature=0.1, # Low temp for evaluation
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None)
        )
    else:
        # Groq Models
        return ChatGroq(
            model=MODEL_MAP[model_key], 
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )

def parse_json_response(content: str) -> Dict:
    """Robust JSON parsing from LLM output."""
    try:
        # Try direct parse
        return json.loads(content)
    except:
        try:
            # Finding JSON block with improved regex
            import re
            match = re.search(r"\{[\s\S]*?\}", content)
            if match:
                return json.loads(match.group(0))
        except:
            pass
    return {"decision": "ERROR", "reasoning": content}

# =============================================================================
# EVALUATION LOGIC - IMPROVED
# =============================================================================

def evaluate_pair(judge, prompt, model_a, joke_a, model_b, joke_b) -> Dict:
    """Run a single pairwise comparison with retries + rate limiting + key rotation."""
    
    # Construct message with constrained feature list
    feature_list_str = ", ".join(ALLOWED_FEATURES)
    user_msg = PAIRWISE_USER_TEMPLATE.format(
        original_prompt=prompt,
        joke_a=joke_a,
        joke_b=joke_b,
        allowed_features=feature_list_str
    )
    
    messages = [
        SystemMessage(content=PAIRWISE_SYSTEM_PROMPT),
        HumanMessage(content=user_msg)
    ]
    
    # Retry loop with exponential backoff & Rate Limiting
    max_retries = 3
    last_error = ""
    current_judge = judge 
    
    for attempt in range(max_retries):
        try:
            # RATE LIMITING: Sleep to avoid limits (approx 30/min for Groq)
            time.sleep(2) 
            
            response = current_judge.invoke(messages)
            result = parse_json_response(response.content)
            
            # Validation: Ensure decision is A, B, or TIE
            if "decision" in result:
                decision = str(result["decision"]).upper().strip()
                if decision not in ["A", "B", "TIE"]:
                     raise ValueError(f"Invalid decision value: {decision}")
                result["decision"] = decision
            else:
                 raise ValueError("Missing 'decision' key in JSON response")

            return result
        except Exception as e:
            last_error = str(e)
            
            # --- KEY ROTATION LOGIC ---
            # If we hit an auth/rate error, reload key from .env
            if "401" in last_error or "authentication" in last_error.lower():
                print(f"\n[Auth Error] Reloading API Key from .env... (Attempt {attempt+1}/{max_retries})")
                try:
                    load_dotenv("../.env", override=True)
                except: pass
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Backoff: 1s, 2s, 4s
            
    return {"decision": "ERROR", "reasoning": f"Failed after {max_retries} retries: {last_error}"}

def run_evaluation(input_file, output_file, judge_name, limit=None, specific_models=None):
    """Main evaluation loop."""
    
    # 1. Load Data
    print(f"Loading data from {input_file}...")
    try:
        data = []
        with open(input_file, 'r') as f:
            content = f.read().strip()
            # Handle both JSON Array and JSONL filenames/content
            if content.startswith('['):
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON array.")
                    return
            else:
                for line in content.split('\n'):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except: pass
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return

    if limit:
        data = data[:limit]
        
    if not data:
        print("No data loaded.")
        return

    # 2. Identify Models to Compare
    # Extract available joke columns (ending in _joke)
    available_models = [k.replace("_joke", "") for k in data[0].keys() if k.endswith("_joke")]
    
    if specific_models:
        # Filter to only requested models
        models_to_compare = [m for m in available_models if m in specific_models]
    else:
        models_to_compare = available_models
        
    print(f"Comparing models: {models_to_compare}")
    print(f"Total entries: {len(data)}")
    
    # Generate all pairs
    all_pairs = list(combinations(models_to_compare, 2))
    print(f"Pairs per prompt: {len(all_pairs)} (Total evaluations: {len(data) * len(all_pairs) * 2})") 
    
    # 3. Setup Judge
    print(f"Initializing Judge: {judge_name}...")
    try:
        judge = get_judge_model(judge_name)
    except Exception as e:
        print(f"Error initializing judge: {e}")
        return
    
    # 4. Run Loop
    existing_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    # Unique key logic
                    key = f"{r['prompt_id']}_{r['judge_model']}_{r['model_a']}_{r['model_b']}_{r['swapped']}"
                    existing_keys.add(key)
                except: pass
        print(f"Resuming... {len(existing_keys)} evaluations already done.")

    # Error Log File (Robust naming)
    base_name = os.path.splitext(output_file)[0]
    error_log_file = f"{base_name}_errors.jsonl"
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    progress_bar = tqdm(data, desc="Evaluating")

    with open(output_file, 'a') as f_out, open(error_log_file, 'a') as f_err:
        for item in progress_bar:
            prompt_id = item['id']
            prompt_text = item['prompt']
            
            for m1, m2 in all_pairs:
                joke_1 = item.get(f"{m1}_joke", "")
                joke_2 = item.get(f"{m2}_joke", "")
                
                if not joke_1 or not joke_2 or "ERROR" in joke_1 or "ERROR" in joke_2:
                    continue
                    
                # Run A/B and B/A
                run_configs = [
                    (m1, joke_1, m2, joke_2, False),  # Normal
                    (m2, joke_2, m1, joke_1, True)    # Swapped
                ]
                
                for mod_a, j_a, mod_b, j_b, is_swapped in run_configs:
                    resumption_key = f"{prompt_id}_{judge_name}_{m1}_{m2}_{is_swapped}"
                    
                    if resumption_key in existing_keys:
                        stats["skipped"] += 1
                        continue
                        
                    eval_res = evaluate_pair(judge, prompt_text, mod_a, j_a, mod_b, j_b)
                    
                    decision = eval_res.get("decision", "ERROR")
                    
                    if decision == "ERROR":
                        stats["failed"] += 1
                        # Log error
                        error_rec = {
                           "prompt_id": prompt_id,
                           "models": (mod_a, mod_b),
                           "error": eval_res.get("reasoning", "Unknown error")
                        }
                        f_err.write(json.dumps(error_rec) + "\n")
                        f_err.flush()
                        continue
                    
                    stats["success"] += 1
                    
                    # --- Resolve Explicit Winner ---
                    winner_model = "TIE"
                    loser_model = "TIE"
                    
                    if decision == "A":
                        winner_model = mod_a
                        loser_model = mod_b
                    elif decision == "B":
                        winner_model = mod_b
                        loser_model = mod_a
                    
                    result_record = {
                        "judge_model": judge_name,
                        "prompt_id": prompt_id,
                        "original_prompt": prompt_text,
                        "model_a": m1, # Canonical Model A
                        "model_b": m2, # Canonical Model B
                        "joke_a": joke_1, 
                        "joke_b": joke_2,
                        "swapped": is_swapped,
                        "run_model_a": mod_a, # Actual input A
                        "run_model_b": mod_b, # Actual input B
                        "decision": decision,
                        "winner_model": winner_model, # EXPLICIT WINNER
                        "loser_model": loser_model,
                        "reasoning": eval_res.get("reasoning", ""),
                        "winner_features": eval_res.get("winner_features", []),
                        "loser_features": eval_res.get("loser_features", []),
                        "timestamp": time.time()
                    }
                    f_out.write(json.dumps(result_record) + "\n")
                    f_out.flush()
            
            progress_bar.set_postfix(stats)

    print(f"Evaluation complete. Saved to {output_file}")
    print(f"Final Stats: Success={stats['success']}, Failed={stats['failed']}, Skipped={stats['skipped']}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    _repo = Path(__file__).resolve().parent.parent
    _default_input = str(_repo / "results" / "benchmark_results_v3_updated.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=str, required=True, choices=MODEL_MAP.keys(), help="Which judge model to use")
    parser.add_argument("--input", type=str, default=_default_input, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, help="Limit number of prompts to evaluate")
    parser.add_argument("--models", type=str, nargs="+", help="Specific list of models to compare (columns without _joke)")
    
    args = parser.parse_args()
    
    run_evaluation(args.input, args.output, args.judge, args.limit, args.models)

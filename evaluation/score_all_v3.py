"""
Scoring Script for HumorGen V3 (Pairwise Win Rate)
Reads: SEMEVAL/outputs/benchmark_results_v3_full.json
Judge: GPT-5-mini (or fallback)
Method: Head-to-Head Comparison (Pairwise)
"""

import sys
import json
import os
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
# Removed langchain usage to avoid schematic errors
from openai import OpenAI

# Config (paths relative to repo root = parent of evaluation/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = _REPO_ROOT / "results" / "benchmark_results_v3_full.json"
OUTPUT_FILE = _REPO_ROOT / "results" / "benchmark_pairwise_scores_v3_OLD.json"
JUDGE_MODEL = "gpt-4o" 

# Pairwise Setup
TARGETS = ["SFT_V3", "DPO_V3", "GRPO_V3"] 
BASELINES = ["Qwen_7B_Instruct", "Qwen_32B_Instruct", "GPT-4o-mini"]

def get_judge_client():
    """Initialize OpenAI Client with env var."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client

def evaluate_pair(client, prompt, joke_a, joke_b):
    """Run one pairwise comparison (Direct - No CoT)."""
    eval_prompt = f"""Compare the following two jokes based on the prompt.
    Which one is FUNNIER and clearer?
    
    Prompt: {prompt}
    
    Model A Joke: "{joke_a}"
    
    Model B Joke: "{joke_b}"
    
    Evaluation Criteria:
    - Surprise & Incongruity
    - Narrative payoff
    - Originality
    
    Task: Decide which joke is better.
    OUTPUT FORMAT:
    First line: "WINNER: A" or "WINNER: B"
    Second line: Short reason.
    """
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert humor judge."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        lines = content.split('\n')
        
        winner_tag = "Error"
        reason = content
        
        # Parse output
        first_line = lines[0].upper()
        if "WINNER: A" in first_line or "WINNER: A" in content:
            winner_tag = "A"
        elif "WINNER: B" in first_line or "WINNER: B" in content:
            winner_tag = "B"
        else:
            # Fallback
            if "Model A" in content and "Model B" not in content: winner_tag = "A"
            elif "Model B" in content and "Model A" not in content: winner_tag = "B"
            else: winner_tag = "Tie" 
            
        return winner_tag, reason

    except Exception as e:
        print(f"Judge Error (Model: {JUDGE_MODEL}): {e}")
        return "Error", str(e)

def main():
    if not INPUT_FILE.exists():
        print("Input file not found.")
        return
        
    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    judge_client = get_judge_client()
    results = []
    
    print(f"Starting Pairwise Evaluation for {TARGETS} vs {BASELINES}...")
    
    for row in tqdm(data):
        prompt = row['prompt']
        row_res = {"id": row['id'], "prompt": prompt}
        
        # For each target model (SFT, DPO, GRPO)
        for target in TARGETS:
            target_joke = row.get(f"{target}_joke", "")
            if not target_joke or target_joke == "ERROR": continue

            # Compare against each baseline
            for base in BASELINES:
                base_joke = row.get(f"{base}_joke", "")
                if not base_joke or base_joke == "ERROR": continue
                
                # Randomize order to prevent position bias
                is_flipped = random.random() > 0.5
                
                if is_flipped:
                    # A=Baseline, B=Target
                    winner_tag, reason = evaluate_pair(judge_client, prompt, base_joke, target_joke)
                    if winner_tag == "A": winner_model = base
                    elif winner_tag == "B": winner_model = target
                    elif winner_tag == "Tie": winner_model = "Tie"
                    else: winner_model = "Error"
                else:
                    # A=Target, B=Baseline
                    winner_tag, reason = evaluate_pair(judge_client, prompt, target_joke, base_joke)
                    if winner_tag == "A": winner_model = target
                    elif winner_tag == "B": winner_model = base
                    elif winner_tag == "Tie": winner_model = "Tie"
                    else: winner_model = "Error"
                    
                row_res[f"{target}_vs_{base}_winner"] = winner_model
                row_res[f"{target}_vs_{base}_reason"] = reason
            
        results.append(row_res)
        
        # Save incrementally
        with open(OUTPUT_FILE, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Win Rate Summary
    print("\n=== Win Rate Summary ===")
    df = pd.DataFrame(results)
    
    summary_data = []
    for target in TARGETS:
        for base in BASELINES:
            col = f"{target}_vs_{base}_winner"
            if col in df.columns:
                wins = df[col].value_counts()
                target_wins = wins.get(target, 0)
                total = len(df)
                wr = (target_wins / total) * 100
                print(f"{target} vs {base}: {wr:.1f}% Win Rate ({target_wins}/{total})")
                summary_data.append({"Model": target, "Opponent": base, "Win Rate": wr})
    
    # Save summary CSV for paper (List format)
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(_REPO_ROOT / "results" / "win_rates_summary.csv", index=False)
    
    # Save Matrix format (Pivot Table) - Ideal for Paper Table
    matrix_df = summary_df.pivot(index="Model", columns="Opponent", values="Win Rate")
    matrix_df.to_csv(_REPO_ROOT / "results" / "win_rates_matrix.csv")
    print("\n=== Saved outputs ===")
    print(f"1. Detailed Logs: {OUTPUT_FILE}")
    print(f"2. Summary CSV:   results/win_rates_summary.csv")
    print(f"3. Matrix CSV:    results/win_rates_matrix.csv")
    print("\nMatrix Preview:")
    print(matrix_df)

if __name__ == "__main__":
    main()

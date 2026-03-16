import json
import pandas as pd
from collections import defaultdict

def load_all_data():
    files = [
        "newdata/results_pilot/pilot_qwen.jsonl",
        "newdata/results_pilot/pilot_gpt4o.jsonl",
        "newdata/results_pilot/pilot_llama.jsonl"
    ]
    all_data = []
    for f in files:
        try:
            with open(f, "r") as file:
                for line in file:
                    if line.strip():
                        all_data.append(json.loads(line))
        except:
            pass
    return all_data

def generate_head_to_head_matrix():
    data = load_all_data()
    if not data:
        print("No data found!")
        return
    
    # Get all models
    models = sorted(set(
        [d.get("winner_model") for d in data] + 
        [d.get("loser_model") for d in data]
    ))
    
    # Remove SFT_V3 if present (only 1 match)
    models = [m for m in models if m and m != "SFT_V3"]
    
    # Build head-to-head counts
    # h2h[model_a][model_b] = wins of model_a against model_b
    h2h_wins = defaultdict(lambda: defaultdict(int))
    h2h_total = defaultdict(lambda: defaultdict(int))
    
    for d in data:
        winner = d.get("winner_model")
        loser = d.get("loser_model")
        if winner and loser and winner in models and loser in models:
            h2h_wins[winner][loser] += 1
            h2h_total[winner][loser] += 1
            h2h_total[loser][winner] += 1
    
    # Create matrix
    print("="*80)
    print("  HEAD-TO-HEAD PAIRWISE COMPARISON MATRIX")
    print("  (Row Model vs Column Model: Win Rate for Row)")
    print("="*80)
    print()
    
    # Build DataFrame
    matrix = []
    for row_model in models:
        row = {"Model": row_model}
        for col_model in models:
            if row_model == col_model:
                row[col_model] = "—"
            else:
                wins = h2h_wins[row_model][col_model]
                total = h2h_total[row_model][col_model]
                if total > 0:
                    win_rate = wins / total * 100
                    row[col_model] = f"{win_rate:.0f}% ({wins}/{total})"
                else:
                    row[col_model] = "N/A"
        matrix.append(row)
    
    df = pd.DataFrame(matrix)
    df = df.set_index("Model")
    print(df.to_string())
    
    print("\n" + "="*80)
    print("  INTERPRETATION GUIDE")
    print("="*80)
    print("- Read as: Row beats Column X% of the time")
    print("- Example: 'DPO_V3 vs kimi_k2: 2% (1/45)' means DPO only beats Kimi 2% of the time")
    print()
    
    # Export to markdown for judge.md
    print("\n" + "="*80)
    print("  MARKDOWN TABLE (Copy to judge.md)")
    print("="*80)
    print()
    
    # Simplified version for markdown
    print("| Model | vs DPO_V3 | vs GRPO_V3 | vs Qwen_32B | vs Qwen_7B | vs GPT-4o | vs llama_3_3 | vs kimi_k2 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    model_map = {
        "DPO_V3": "DPO_V3",
        "GRPO_V3": "GRPO_V3", 
        "Qwen_32B_Instruct": "Qwen_32B",
        "Qwen_7B_Instruct": "Qwen_7B",
        "GPT-4o-mini": "GPT-4o",
        "llama_3_3": "llama_3_3",
        "kimi_k2": "kimi_k2"
    }
    
    col_order = ["DPO_V3", "GRPO_V3", "Qwen_32B_Instruct", "Qwen_7B_Instruct", "GPT-4o-mini", "llama_3_3", "kimi_k2"]
    
    for row_model in col_order:
        if row_model not in models:
            continue
        row_str = f"| **{model_map.get(row_model, row_model)}** |"
        for col_model in col_order:
            if col_model not in models:
                continue
            if row_model == col_model:
                row_str += " — |"
            else:
                wins = h2h_wins[row_model][col_model]
                total = h2h_total[row_model][col_model]
                if total > 0:
                    win_rate = wins / total * 100
                    row_str += f" {win_rate:.0f}% |"
                else:
                    row_str += " N/A |"
        print(row_str)

if __name__ == "__main__":
    generate_head_to_head_matrix()

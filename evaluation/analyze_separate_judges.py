import json
import pandas as pd
from collections import Counter
import glob
import os

def analyze_judge(file_path):
    judge_name = os.path.basename(file_path).replace("pilot_", "").replace(".jsonl", "")
    print(f"\n{'='*20} {judge_name.upper()} JUDGE {'='*20}")
    
    data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if not data:
        print("No data found.")
        return

    df = pd.DataFrame(data)
    total_comparisons = len(df)
    print(f"Total Comparisons: {total_comparisons}")

    # Win Rates
    win_counts = df["winner_model"].value_counts()
    match_counts = df["model_a"].value_counts() + df["model_b"].value_counts()
    
    # Ensure all models are in match_counts even if they never played (unlikely here)
    models = set(df["model_a"].unique()) | set(df["model_b"].unique())
    
    leaderboard = []
    for model in models:
        wins = win_counts.get(model, 0)
        matches = len(df[ (df["model_a"]==model) | (df["model_b"]==model) ])
        win_rate = (wins / matches * 100) if matches > 0 else 0
        leaderboard.append({
            "Model": model,
            "Win Rate": f"{win_rate:.1f}%",
            "Wins": wins,
            "Losses": matches - wins,
            "Matches": matches
        })
    
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("Wins", ascending=False)
    print("\n--- Leaderboard ---")
    print(leaderboard_df.to_string(index=False))

    # Features
    winner_features = []
    for feats in df["winner_features"]:
        if isinstance(feats, list):
            winner_features.extend([f.lower() for f in feats])
            
    loser_features = []
    for feats in df["loser_features"]:
        if isinstance(feats, list):
            loser_features.extend([f.lower() for f in feats])

    print("\n--- Features Associated with WINNING ---")
    for f, c in Counter(winner_features).most_common(5):
        print(f"- {f}: {c} ({c/total_comparisons*100:.1f}%)")

    print("\n--- Features Associated with LOSING ---")
    for f, c in Counter(loser_features).most_common(5):
        print(f"- {f}: {c} ({c/total_comparisons*100:.1f}%)")

# Run Analysis
files = [
    "newdata/results_pilot/pilot_qwen.jsonl",
    "newdata/results_pilot/pilot_gpt4o.jsonl",
    "newdata/results_pilot/pilot_llama.jsonl"
]

for f in files:
    analyze_judge(f)

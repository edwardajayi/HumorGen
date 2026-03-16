import json
import pandas as pd
from collections import Counter, defaultdict
import glob

def load_judge_data(file_path):
    """Load JSONL data from a judge file."""
    data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def generate_report():
    files = {
        "Qwen-32B": "newdata/results_pilot/pilot_qwen.jsonl",
        "GPT-4o-mini": "newdata/results_pilot/pilot_gpt4o.jsonl",
        "Llama-3.3": "newdata/results_pilot/pilot_llama.jsonl"
    }
    
    # Aggregate data across all judges
    all_data = []
    for judge_name, file_path in files.items():
        data = load_judge_data(file_path)
        for d in data:
            d["judge_name"] = judge_name
        all_data.extend(data)
    
    if not all_data:
        print("No data found!")
        return
    
    df = pd.DataFrame(all_data)
    total_comparisons = len(df)
    
    print("="*70)
    print("   ROBUST HUMOR EVALUATION REPORT")
    print("="*70)
    print(f"\nTotal Comparisons Analyzed: {total_comparisons}")
    print(f"Judges: {list(files.keys())}")
    
    # ========================================
    # SECTION 1: OVERALL LEADERBOARD
    # ========================================
    print("\n" + "="*70)
    print("  SECTION 1: OVERALL LEADERBOARD (All Judges Combined)")
    print("="*70)
    
    models = set(df["winner_model"].unique()) | set(df["loser_model"].unique())
    leaderboard = []
    for model in models:
        wins = len(df[df["winner_model"] == model])
        losses = len(df[df["loser_model"] == model])
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        leaderboard.append({
            "Model": model,
            "Win Rate": f"{win_rate:.1f}%",
            "Wins": wins,
            "Losses": losses,
            "Total Matches": total
        })
    
    lb_df = pd.DataFrame(leaderboard).sort_values("Wins", ascending=False)
    print(lb_df.to_string(index=False))
    
    # ========================================
    # SECTION 2: FEATURES THAT MAKE EACH MODEL WIN
    # ========================================
    print("\n" + "="*70)
    print("  SECTION 2: TOP WINNING FEATURES PER MODEL")
    print("  (What features are cited when THIS model wins)")
    print("="*70)
    
    model_win_features = defaultdict(list)
    for _, row in df.iterrows():
        winner = row.get("winner_model", "")
        feats = row.get("winner_features", [])
        if isinstance(feats, list):
            model_win_features[winner].extend([f.lower() for f in feats])
    
    for model in sorted(model_win_features.keys()):
        feats = model_win_features[model]
        total_wins = len(df[df["winner_model"] == model])
        print(f"\n--- {model} (Total Wins: {total_wins}) ---")
        for feat, count in Counter(feats).most_common(5):
            pct = count / total_wins * 100 if total_wins > 0 else 0
            print(f"  {feat}: {count} ({pct:.1f}%)")
    
    # ========================================
    # SECTION 3: FEATURES THAT MAKE EACH MODEL LOSE
    # ========================================
    print("\n" + "="*70)
    print("  SECTION 3: TOP LOSING FEATURES PER MODEL")
    print("  (What flaws are cited when THIS model loses)")
    print("="*70)
    
    model_lose_features = defaultdict(list)
    for _, row in df.iterrows():
        loser = row.get("loser_model", "")
        feats = row.get("loser_features", [])
        if isinstance(feats, list):
            model_lose_features[loser].extend([f.lower() for f in feats])
    
    for model in sorted(model_lose_features.keys()):
        feats = model_lose_features[model]
        total_losses = len(df[df["loser_model"] == model])
        print(f"\n--- {model} (Total Losses: {total_losses}) ---")
        for feat, count in Counter(feats).most_common(5):
            pct = count / total_losses * 100 if total_losses > 0 else 0
            print(f"  {feat}: {count} ({pct:.1f}%)")
    
    # ========================================
    # SECTION 4: JUDGE AGREEMENT ANALYSIS
    # ========================================
    print("\n" + "="*70)
    print("  SECTION 4: JUDGE-SPECIFIC PREFERENCES")
    print("="*70)
    
    for judge_name in files.keys():
        judge_df = df[df["judge_name"] == judge_name]
        if judge_df.empty:
            continue
        
        print(f"\n--- {judge_name} Judge ---")
        print(f"    Total Comparisons: {len(judge_df)}")
        
        # Top features this judge likes
        win_feats = []
        for feats in judge_df["winner_features"]:
            if isinstance(feats, list):
                win_feats.extend([f.lower() for f in feats])
        
        print("    Top Features Associated with Winners:")
        for feat, count in Counter(win_feats).most_common(3):
            pct = count / len(judge_df) * 100
            print(f"      {feat}: {count} ({pct:.1f}%)")
        
        # This judge's favorite model
        fav_model = judge_df["winner_model"].value_counts().idxmax()
        fav_wins = judge_df["winner_model"].value_counts().max()
        print(f"    Favorite Model: {fav_model} ({fav_wins} wins)")
    
    print("\n" + "="*70)
    print("   END OF REPORT")
    print("="*70)

if __name__ == "__main__":
    generate_report()

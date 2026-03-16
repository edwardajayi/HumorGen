"""
HumorRank Validator
-------------------
Provides spot-check visualization and statistical validation
of ELO rankings produced by the HumorRank system.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any, Tuple
import numpy as np

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    return data

def spot_check(rankings_file: str, num_prompts: int = 10, seed: int = 42):
    """
    Randomly selects prompt results and displays Top/Bottom jokes for manual inspection.
    """
    print(f"\n--- SPOT CHECK (Seed: {seed}) ---")
    data = load_jsonl(rankings_file)
    if not data: return

    rng = random.Random(seed)
    
    # Selecting distinct prompts
    if len(data) > num_prompts:
        selection = rng.sample(data, num_prompts)
    else:
        selection = data
        
    for item in selection:
        headline = item.get("headline", "Unknown Headline")
        headline_id = item.get("headline_id", "Unknown ID")
        joke_ids = item.get("ranking", [])
        normalized = item.get("normalized_scores", {})
        elo = item.get("elo_scores", {})
        original_elo = item.get("raw_elo_scores", {})
        wins = item.get("win_counts", {})
        ties = item.get("tie_counts", {})
        games = item.get("games_played", {})
        
        # Sort by normalized score desc
        sorted_jokes = sorted(joke_ids, key=lambda j: normalized.get(j, 0), reverse=True)
        
        print(f"\nPrompt [{headline_id}]: \"{headline}\"")
        print(f"{'Rank':<4} {'Score':<6} {'ELO':<6} {'W/L/T':<10} {'ID':<15} {'Text (Truncated)'}")
        print("-" * 80)
        
        def print_row(rank, jid, icon=""):
            n_score = normalized.get(jid, 0)
            e_score = elo.get(jid, 0)
            w = wins.get(jid, 0)
            t = ties.get(jid, 0)
            g = games.get(jid, 0)
            l = g - w - t
            # Need joke text lookup? Ideally pass content dict or assume ID meaningful
            # Without content file, just showing ID is safest, or if ID contains model name
            print(f"{icon} {rank:<2} {n_score:<6.1f} {e_score:<6.0f} {w}/{l}/{t:<4} {jid:<15}")

        # Top 3
        for i in range(min(3, len(sorted_jokes))):
            print_row(i+1, sorted_jokes[i], "1st" if i==0 else "2nd" if i==1 else "3rd")
            
        print("...")
        
        # Bottom 3
        start_idx = max(3, len(sorted_jokes) - 3)
        for i in range(start_idx, len(sorted_jokes)):
            print_row(i+1, sorted_jokes[i], "...")
            
        # Summary
        max_elo = max(elo.values()) if elo else 0
        min_elo = min(elo.values()) if elo else 0
        range_elo = max_elo - min_elo
        converged = item.get("converged", False)
        print(f"Total Comparisons: {item.get('comparisons_made', 0)} | Converged: {converged} | ELO Range: {range_elo:.1f}")


def compute_metrics(rankings_file: str) -> Dict[str, Any]:
    """
    Computes aggregate statistics for the entire ranking run.
    """
    data = load_jsonl(rankings_file)
    if not data: return {}
    
    total_prompts = len(data)
    total_jokes = 0
    total_comparisons = 0
    total_elo_range = 0
    converged_count = 0
    total_games = 0
    total_ties_global = 0
    
    for item in data:
        joke_ids = item.get("ranking", [])
        total_jokes += len(joke_ids)
        total_comparisons += item.get("comparisons_made", 0)
        
        elo = item.get("elo_scores", {})
        if elo:
            total_elo_range += (max(elo.values()) - min(elo.values()))
            
        if item.get("converged"):
            converged_count += 1
            
        games = item.get("games_played", {})
        ties = item.get("tie_counts", {})
        total_games += sum(games.values()) # Note: games sum double counts matches (A vs B is 1 game for A, 1 for B)
        # Matches = total_games / 2
        total_ties_global += sum(ties.values()) # Same for ties
        
    avg_jokes = total_jokes / total_prompts if total_prompts else 0
    avg_comps = total_comparisons / total_prompts if total_prompts else 0
    avg_range = total_elo_range / total_prompts if total_prompts else 0
    convergence_rate = (converged_count / total_prompts) * 100 if total_prompts else 0
    
    # Real matches = games sum / 2
    real_total_matches = total_games / 2
    tie_rate = (total_ties_global / 2) / real_total_matches if real_total_matches else 0
    
    return {
        "total_prompts": total_prompts,
        "total_jokes_ranked": total_jokes,
        "avg_jokes_per_prompt": round(avg_jokes, 1),
        "avg_comparisons_per_prompt": round(avg_comps, 1),
        "avg_elo_range": round(avg_range, 1),
        "convergence_rate": f"{convergence_rate:.1f}%",
        "tie_rate": f"{tie_rate*100:.1f}%"
    }

def generate_report(rankings_file: str, output_file: str):
    metrics = compute_metrics(rankings_file)
    with open(output_file, "w") as f:
        f.write("# HumorRank Validation Report\n\n")
        f.write("## Aggregate Metrics\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumorRank Validator")
    parser.add_argument("--rankings", required=True, help="Path to humorrank_rankings.jsonl")
    parser.add_argument("--spot-check", type=int, default=10, help="Number of prompts to spot check")
    parser.add_argument("--report", type=str, help="Path to save markdown report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    spot_check(args.rankings, args.spot_check, args.seed)
    
    metrics = compute_metrics(args.rankings)
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    if args.report:
        generate_report(args.rankings, args.report)
        print(f"\nReport saved to {args.report}")

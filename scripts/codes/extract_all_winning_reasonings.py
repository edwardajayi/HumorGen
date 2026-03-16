#!/usr/bin/env python3
import json
import os
import glob
from tqdm import tqdm

def main():
    # Repo root = two levels up from scripts/codes/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, "results", "all_winning_reasonings.jsonl")
    
    files = glob.glob(os.path.join(project_root, "results/humor_rank_h100_final/*/humorrank_match_history.jsonl")) + \
            glob.glob(os.path.join(project_root, "results/humor_rank_mini/full_100/humorrank_match_history.jsonl"))

    print(f"Scanning {len(files)} files for winning reasonings...")
    
    winning_data = []
    
    for fpath in tqdm(files, desc="Processing Shards"):
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    entry = json.loads(line)
                    headline_id = entry.get('headline_id')
                    
                    for match in entry.get('match_history', []):
                        joke_a_id = match.get('joke_a_id')
                        joke_b_id = match.get('joke_b_id')
                        winner_id = match.get('winner_id')
                        reasoning = match.get('reasoning')
                        features = match.get('features', [])
                        
                        # Derive loser_id
                        if winner_id == joke_a_id:
                            loser_id = joke_b_id
                        elif winner_id == joke_b_id:
                            loser_id = joke_a_id
                        else:
                            loser_id = None
                        
                        if reasoning and winner_id and loser_id:
                            winning_data.append({
                                "headline_id": headline_id,
                                "winner_joke_id": f"{headline_id}_{winner_id}",
                                "loser_joke_id": f"{headline_id}_{loser_id}",
                                "features": features,
                                "reasoning": reasoning
                            })
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print(f"\nExtracted {len(winning_data)} winning reasonings.")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for item in winning_data:
            out_f.write(json.dumps(item) + '\n')
            
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()

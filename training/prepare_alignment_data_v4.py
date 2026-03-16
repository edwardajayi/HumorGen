#!/usr/bin/env python3
"""
Prepare Alignment Training Data V4
Creates SFT, DPO, and GRPO datasets from HumorRank Elo-rated jokes.

Input:  humorrank_grpo_rewards.jsonl (12 shards)
Output: results/alignment_data/sft_train_v4.jsonl   (~12,000 examples)
        results/alignment_data/dpo_train_v4.jsonl   (~6,000 pairs)
        results/alignment_data/grpo_train_v4.jsonl  (~1,200 groups)
"""

import json
import glob
import os
import random
from pathlib import Path
from collections import defaultdict
from itertools import product

random.seed(42)

# === PATHS === (repo root = parent of training/)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
INPUT_SHARDS = (
    glob.glob(os.path.join(PROJECT_ROOT, "results/humor_rank_h100_final/*/humorrank_grpo_rewards.jsonl")) +
    glob.glob(os.path.join(PROJECT_ROOT, "results/humor_rank_mini/full_100/humorrank_grpo_rewards.jsonl"))
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/alignment_data")
SFT_OUTPUT  = os.path.join(OUTPUT_DIR, "sft_train_v4.jsonl")
DPO_OUTPUT  = os.path.join(OUTPUT_DIR, "dpo_train_v4.jsonl")
GRPO_OUTPUT = os.path.join(OUTPUT_DIR, "grpo_train_v4.jsonl")


def clean_prompt(prompt):
    """
    'Write a funny joke about: Headline: "xxx"' → 'Write a funny joke about "xxx"'
    'Write a funny joke about: Words: "xxx"'    → 'Write a funny joke about "xxx"'
    """
    p = prompt.strip()
    # Remove the "Headline: " or "Words: " prefix after the colon
    if ": Headline: " in p:
        p = p.replace(": Headline: ", " ")
    elif ": Words: " in p:
        p = p.replace(": Words: ", " ")
    return p


def load_all_data():
    """Load all shards and group by headline_id."""
    headlines = defaultdict(list)
    
    print(f"Loading {len(INPUT_SHARDS)} shard files...")
    for fpath in INPUT_SHARDS:
        with open(fpath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                headlines[entry['headline_id']].append(entry)
    
    # Sort each group by Elo descending
    for hid in headlines:
        headlines[hid].sort(key=lambda x: x['elo'], reverse=True)
    
    print(f"Loaded {sum(len(v) for v in headlines.values())} jokes across {len(headlines)} headlines.")
    return headlines


def create_sft_data(headlines):
    """Top 10 jokes per headline (by Elo rank). Trust the rankings."""
    examples = []
    
    for hid, jokes in headlines.items():
        prompt = clean_prompt(jokes[0]['prompt'])
        top_10 = jokes[:10]
        
        for joke in top_10:
            examples.append({
                "instruction": prompt,
                "output": joke['joke'],
            })
    
    return examples


def create_dpo_data(headlines):
    """Top 5 vs Bottom 5, shuffled cross-pairing, sample 5 pairs per headline."""
    pairs = []
    
    for hid, jokes in headlines.items():
        if len(jokes) < 10:
            continue
            
        prompt = clean_prompt(jokes[0]['prompt'])
        
        # Top 5 and Bottom 5
        top5 = list(jokes[:5])
        bottom5 = list(jokes[-5:])
        
        # Shuffle both pools
        random.shuffle(top5)
        random.shuffle(bottom5)
        
        # All 25 cross-pairs
        all_cross_pairs = list(product(top5, bottom5))
        
        # Sample 5 random pairs
        sampled = random.sample(all_cross_pairs, min(5, len(all_cross_pairs)))
        
        for chosen, rejected in sampled:
            elo_gap = chosen['elo'] - rejected['elo']
                
            pairs.append({
                "prompt": prompt,
                "chosen": chosen['joke'],
                "rejected": rejected['joke'],
                "elo_gap": round(elo_gap, 2),
            })
    
    return pairs


def create_grpo_data(headlines):
    """All 24 jokes per headline with computed advantages."""
    groups = []
    
    for hid, jokes in headlines.items():
        prompt = clean_prompt(jokes[0]['prompt'])
        
        # Compute group stats
        elos = [j['elo'] for j in jokes]
        mean_elo = sum(elos) / len(elos)
        variance = sum((e - mean_elo) ** 2 for e in elos) / len(elos)
        std_elo = variance ** 0.5
        
        candidates = []
        for joke in jokes:
            advantage = (joke['elo'] - mean_elo) / (std_elo + 1e-6)
            candidates.append({
                "response": joke['joke'],
                "reward_winrate": round(joke['reward'], 4),
                "reward_elo": round(joke['elo'], 2),
                "advantage": round(advantage, 4),
            })
        
        groups.append({
            "prompt": prompt,
            "candidates": candidates,
            "mean_reward": round(mean_elo, 2),
            "std_reward": round(std_elo, 2),
        })
    
    return groups


def write_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("ALIGNMENT DATA CREATION V4")
    print("=" * 60)
    
    # Load
    headlines = load_all_data()
    
    # Quality assertions
    assert len(headlines) >= 1100, f"Expected ~1200 headlines, got {len(headlines)}"
    for hid, jokes in headlines.items():
        assert len(jokes) >= 20, f"Headline {hid} has only {len(jokes)} jokes (expected ~24)"
    print("OK: Data integrity checks passed.\n")
    
    # === SFT ===
    print("-" * 40)
    print("Creating SFT data (Top 10 per headline, Elo > 1000)...")
    sft_data = create_sft_data(headlines)
    write_jsonl(sft_data, SFT_OUTPUT)
    print(f"OK: SFT: {len(sft_data)} examples -> {SFT_OUTPUT}")
    
    # Sample
    sample = random.choice(sft_data)
    print(f"  Sample: instruction={sample['instruction'][:60]}...")
    print(f"          output={sample['output'][:80]}...\n")
    
    # === DPO ===
    print("-" * 40)
    print("Creating DPO data (Top 5 vs Bottom 5, shuffled, 5 pairs/headline)...")
    dpo_data = create_dpo_data(headlines)
    write_jsonl(dpo_data, DPO_OUTPUT)
    print(f"OK: DPO: {len(dpo_data)} pairs -> {DPO_OUTPUT}")
    
    # Stats
    gaps = [d['elo_gap'] for d in dpo_data]
    print(f"  Elo gap: min={min(gaps):.1f}, avg={sum(gaps)/len(gaps):.1f}, max={max(gaps):.1f}")
    
    sample = random.choice(dpo_data)
    print(f"  Sample: prompt={sample['prompt'][:60]}...")
    print(f"          chosen={sample['chosen'][:60]}...")
    print(f"          rejected={sample['rejected'][:60]}...")
    print(f"          elo_gap={sample['elo_gap']}\n")
    
    # === GRPO ===
    print("-" * 40)
    print("Creating GRPO data (All 24 candidates per headline)...")
    grpo_data = create_grpo_data(headlines)
    write_jsonl(grpo_data, GRPO_OUTPUT)
    total_candidates = sum(len(g['candidates']) for g in grpo_data)
    print(f"OK: GRPO: {len(grpo_data)} groups ({total_candidates} total candidates) -> {GRPO_OUTPUT}")
    
    sample = random.choice(grpo_data)
    print(f"  Sample: prompt={sample['prompt'][:60]}...")
    print(f"          candidates={len(sample['candidates'])}, mean_reward={sample['mean_reward']}, std={sample['std_reward']}")
    top = max(sample['candidates'], key=lambda x: x['advantage'])
    bot = min(sample['candidates'], key=lambda x: x['advantage'])
    print(f"          best advantage={top['advantage']}, worst={bot['advantage']}\n")
    
    # === SUMMARY ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SFT:  {len(sft_data):,} examples")
    print(f"  DPO:  {len(dpo_data):,} pairs")
    print(f"  GRPO: {len(grpo_data):,} groups ({total_candidates:,} candidates)")
    print(f"\n  Output files:")
    print(f"    {SFT_OUTPUT}")
    print(f"    {DPO_OUTPUT}")
    print(f"    {GRPO_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()

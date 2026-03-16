#!/usr/bin/env python3
"""
Prepare Alignment Training Data — THINKING VERSION
Based on prepare_alignment_data_v4.py (the working version used for trained models).

Identical to V4 except:
  - SFT outputs include <think>[THOUGHT block]</think> before the joke
  - DPO chosen includes <think>[THOUGHT block]</think> before the joke
  - DPO rejected ALSO includes <think>[THOUGHT block]</think> before the joke — symmetric
  - GRPO responses include <think>[THOUGHT block]</think> before the joke

The THOUGHT blocks come from the original teacher model generation (Kimi-K2 / Qwen-32B),
stored in merged_jokes.jsonl under each candidate's "reasoning" field.

Input:  humorrank_grpo_rewards.jsonl (12 shards) + merged_jokes.jsonl (THOUGHT blocks)
Output: training/think/data/sft_think.jsonl   (~12,000 examples)
        training/think/data/dpo_think.jsonl   (~6,000 pairs)
        training/think/data/grpo_think.jsonl  (~1,200 groups)
"""

import json
import glob
import os
import random
from pathlib import Path
from collections import defaultdict
from itertools import product

random.seed(42)

# Repo root = two levels up from training/think/
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
INPUT_SHARDS = (
    glob.glob(os.path.join(PROJECT_ROOT, "results/humor_rank_h100_final/*/humorrank_grpo_rewards.jsonl")) +
    glob.glob(os.path.join(PROJECT_ROOT, "results/humor_rank_mini/full_100/humorrank_grpo_rewards.jsonl"))
)
MERGED_JOKES = os.path.join(PROJECT_ROOT, "results/merged_jokes.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training/think/data")
SFT_OUTPUT  = os.path.join(OUTPUT_DIR, "sft_think.jsonl")
DPO_OUTPUT  = os.path.join(OUTPUT_DIR, "dpo_think.jsonl")
GRPO_OUTPUT = os.path.join(OUTPUT_DIR, "grpo_think.jsonl")


def clean_prompt(prompt):
    p = prompt.strip()
    if ": Headline: " in p:
        p = p.replace(": Headline: ", " ")
    elif ": Words: " in p:
        p = p.replace(": Words: ", " ")
    return p


def build_thought_lookup():
    """Build lookup: (headline_id, joke_first_80_chars) -> THOUGHT block text."""
    lookup = {}
    with open(MERGED_JOKES, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            hid = d['id']
            for c in d['candidates']:
                key = (hid, c['joke'][:80])
                reasoning = c.get('reasoning', '').strip()
                if reasoning:
                    lookup[key] = reasoning
    return lookup


def wrap_with_think(joke_text, thought_text):
    """Wrap joke with <think> block. Returns bare joke if no thought available."""
    if thought_text:
        return f"<think>\n{thought_text}\n</think>\n{joke_text}"
    return joke_text


MAX_RESPONSE_CHARS = 3000  # ~750 tokens; prevents truncation at MAX_SEQ_LENGTH=1024

def load_all_data(thought_lookup):
    """Load all shards and group by headline_id. Attach THOUGHT blocks.

    Fixes applied:
      - Deduplicates by (headline_id, joke[:80]) across shards
      - Skips candidates with no THOUGHT block (this is the think dataset)
      - Skips candidates whose wrapped response exceeds MAX_RESPONSE_CHARS
    """
    headlines = defaultdict(list)
    seen_keys = set()

    print(f"Loading {len(INPUT_SHARDS)} shard files...")
    total = 0
    skipped_dup = 0
    skipped_no_thought = 0
    skipped_too_long = 0
    kept = 0

    for fpath in INPUT_SHARDS:
        with open(fpath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                total += 1
                key = (entry['headline_id'], entry['joke'][:80])

                if key in seen_keys:
                    skipped_dup += 1
                    continue
                seen_keys.add(key)

                thought = thought_lookup.get(key, '')
                if not thought:
                    skipped_no_thought += 1
                    continue

                wrapped_len = len(wrap_with_think(entry['joke'], thought))
                if wrapped_len > MAX_RESPONSE_CHARS:
                    skipped_too_long += 1
                    continue

                entry['thought'] = thought
                headlines[entry['headline_id']].append(entry)
                kept += 1

    for hid in headlines:
        headlines[hid].sort(key=lambda x: x['elo'], reverse=True)

    print(f"Loaded {total} raw entries across {len(INPUT_SHARDS)} shards.")
    print(f"  Kept:               {kept}")
    print(f"  Skipped (duplicate): {skipped_dup}")
    print(f"  Skipped (no THOUGHT): {skipped_no_thought}")
    print(f"  Skipped (too long >={MAX_RESPONSE_CHARS} chars): {skipped_too_long}")
    print(f"  Headlines: {len(headlines)}")
    return headlines


def create_sft_data(headlines):
    """Top 10 jokes per headline. Output = <think>THOUGHT</think>\\njoke"""
    examples = []
    with_think = 0

    for hid, jokes in headlines.items():
        prompt = clean_prompt(jokes[0]['prompt'])
        top_10 = jokes[:10]

        for joke in top_10:
            output = wrap_with_think(joke['joke'], joke['thought'])
            examples.append({
                "instruction": prompt,
                "output": output,
            })
            if joke['thought']:
                with_think += 1

    print(f"  SFT records with <think> block: {with_think}/{len(examples)} ({100*with_think/len(examples):.1f}%)")
    return examples


def create_dpo_data(headlines):
    """
    Top 5 vs Bottom 5, shuffled cross-pairing, sample 5 pairs per headline.
    SYMMETRIC: both chosen AND rejected include <think> blocks.
    The model learns which REASONING leads to better jokes, not just that
    the presence of <think> tags correlates with winning.
    """
    pairs = []

    for hid, jokes in headlines.items():
        if len(jokes) < 10:
            continue

        prompt = clean_prompt(jokes[0]['prompt'])
        top5 = list(jokes[:5])
        bottom5 = list(jokes[-5:])

        random.shuffle(top5)
        random.shuffle(bottom5)

        all_cross_pairs = list(product(top5, bottom5))
        sampled = random.sample(all_cross_pairs, min(5, len(all_cross_pairs)))

        for chosen, rejected in sampled:
            elo_gap = chosen['elo'] - rejected['elo']
            pairs.append({
                "prompt": prompt,
                "chosen": wrap_with_think(chosen['joke'], chosen['thought']),
                "rejected": wrap_with_think(rejected['joke'], rejected['thought']),
                "elo_gap": round(elo_gap, 2),
            })

    return pairs


def create_grpo_data(headlines):
    """All 24 jokes per headline with computed advantages. Responses include <think> blocks."""
    groups = []

    for hid, jokes in headlines.items():
        prompt = clean_prompt(jokes[0]['prompt'])

        elos = [j['elo'] for j in jokes]
        mean_elo = sum(elos) / len(elos)
        variance = sum((e - mean_elo) ** 2 for e in elos) / len(elos)
        std_elo = variance ** 0.5

        candidates = []
        for joke in jokes:
            advantage = (joke['elo'] - mean_elo) / (std_elo + 1e-6)
            candidates.append({
                "response": wrap_with_think(joke['joke'], joke['thought']),
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
    print("ALIGNMENT DATA CREATION — THINKING VERSION")
    print("Based on V4 template (used for existing trained models)")
    print("=" * 60)

    # Build THOUGHT lookup from merged_jokes.jsonl
    print("\nBuilding THOUGHT block lookup from merged_jokes.jsonl...")
    thought_lookup = build_thought_lookup()
    print(f"  {len(thought_lookup)} THOUGHT blocks indexed.\n")

    # Load rewards data with THOUGHT blocks attached
    headlines = load_all_data(thought_lookup)

    # Quality assertions
    assert len(headlines) >= 1100, f"Expected ~1200 headlines, got {len(headlines)}"
    short_headlines = {hid: len(j) for hid, j in headlines.items() if len(j) < 20}
    if short_headlines:
        print(f"WARNING: {len(short_headlines)} headlines have <20 candidates (lost some to no-THOUGHT/too-long):")
        for hid, cnt in sorted(short_headlines.items(), key=lambda x: x[1])[:5]:
            print(f"    {hid}: {cnt} candidates")
    else:
        print("OK: All headlines have >=20 candidates.")

    min_count = min(len(j) for j in headlines.values())
    print(f"  Min candidates per headline: {min_count}")
    print(f"  Total candidates kept: {sum(len(j) for j in headlines.values())}")
    print("OK: Data integrity checks passed.\n")

    # === SFT ===
    print("-" * 40)
    print("Creating SFT-Think data (Top 10 per headline, with <think> blocks)...")
    sft_data = create_sft_data(headlines)
    write_jsonl(sft_data, SFT_OUTPUT)
    print(f"OK: SFT: {len(sft_data)} examples -> {SFT_OUTPUT}")

    sample = random.choice(sft_data)
    print(f"  Sample: instruction={sample['instruction'][:60]}...")
    print(f"          output={sample['output'][:120]}...\n")

    # === DPO ===
    print("-" * 40)
    print("Creating DPO-Think data (Top 5 vs Bottom 5, SYMMETRIC: both chosen+rejected have <think> blocks)...")
    dpo_data = create_dpo_data(headlines)
    write_jsonl(dpo_data, DPO_OUTPUT)
    print(f"OK: DPO: {len(dpo_data)} pairs -> {DPO_OUTPUT}")

    gaps = [d['elo_gap'] for d in dpo_data]
    print(f"  Elo gap: min={min(gaps):.1f}, avg={sum(gaps)/len(gaps):.1f}, max={max(gaps):.1f}")

    sample = random.choice(dpo_data)
    print(f"  Sample chosen:   {sample['chosen'][:100]}...")
    print(f"  Sample rejected: {sample['rejected'][:80]}...")
    print(f"  elo_gap={sample['elo_gap']}\n")

    # === GRPO ===
    print("-" * 40)
    print("Creating GRPO-Think data (All candidates with <think> blocks)...")
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

    # === QUALITY VERIFICATION ===
    print("-" * 40)
    print("Running final quality verification...")
    sft_no_think = sum(1 for s in sft_data if '<think>' not in s['output'])
    dpo_chosen_no_think = sum(1 for d in dpo_data if '<think>' not in d['chosen'])
    dpo_rejected_no_think = sum(1 for d in dpo_data if '<think>' not in d['rejected'])
    grpo_no_think = sum(
        1 for g in grpo_data for c in g['candidates'] if '<think>' not in c['response']
    )
    grpo_group_sizes = [len(g['candidates']) for g in grpo_data]
    max_grpo_size = max(grpo_group_sizes)
    oversized_groups = sum(1 for s in grpo_group_sizes if s > 24)

    max_sft_len = max(len(s['output']) for s in sft_data)
    max_dpo_len = max(max(len(d['chosen']), len(d['rejected'])) for d in dpo_data)
    max_grpo_len = max(len(c['response']) for g in grpo_data for c in g['candidates'])

    all_clean = True
    if sft_no_think > 0:
        print(f"  FAIL: SFT: {sft_no_think} examples missing <think>")
        all_clean = False
    if dpo_chosen_no_think > 0 or dpo_rejected_no_think > 0:
        print(f"  FAIL: DPO: {dpo_chosen_no_think} chosen, {dpo_rejected_no_think} rejected missing <think>")
        all_clean = False
    if grpo_no_think > 0:
        print(f"  FAIL: GRPO: {grpo_no_think} candidates missing <think>")
        all_clean = False
    if oversized_groups > 0:
        print(f"  FAIL: GRPO: {oversized_groups} groups have >{24} candidates (max={max_grpo_size})")
        all_clean = False
    if all_clean:
        print("  OK: All entries have <think> blocks, no duplicates, no oversized groups.")

    print(f"  Max response length - SFT: {max_sft_len}, DPO: {max_dpo_len}, GRPO: {max_grpo_len} chars")

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY — THINKING VERSION (CLEANED)")
    print("=" * 60)
    print(f"  SFT:  {len(sft_data):,} examples  (100% with <think> blocks)")
    print(f"  DPO:  {len(dpo_data):,} pairs     (symmetric: both chosen+rejected have <think>)")
    print(f"  GRPO: {len(grpo_data):,} groups    ({total_candidates:,} candidates, 100% with <think>)")
    print(f"\n  Output dir: {OUTPUT_DIR}")
    print(f"    {SFT_OUTPUT}")
    print(f"    {DPO_OUTPUT}")
    print(f"    {GRPO_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()

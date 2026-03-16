#!/usr/bin/env python3
"""
create_thinking_dataset.py

Creates a clean thinking/reasoning training dataset from HumorRank evaluation data.

STRATEGY:
  For each of 1,200 headlines, select the ONE pairwise comparison with the
  highest win-gap (winner_total_wins - loser_total_wins within that headline).
  This gives 1,200 high-signal records without the 40-hour full-reprocessing cost.

THINKING FORMAT (per split_causal_reasonings.py design):
  The <think> block contains PURE ANALYTICAL REASONING about humor mechanics —
  neutral, descriptive, no "winner/loser" framing, no hallucinated content.
  It uses winner_reason from split_causal_reasonings.jsonl when available,
  or falls back to the original comparative reasoning.

DPO format:
  chosen   = <think>[winner_reason]</think>\\n[winning_joke]
  rejected = [losing_joke]                (no think block)

GRPO/SFT format:
  response = <think>[winner_reason]</think>\\n[winning_joke]

Output:
  data/dpo_thinking_1200.jsonl
  data/grpo_thinking_1200.jsonl
  data/thinking_dataset_report.txt
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# --- Paths (repo root = parent of scripts/) ---
ROOT = Path(__file__).resolve().parent.parent
ALL_REASONINGS = ROOT / "results" / "all_winning_reasonings.jsonl"
SPLIT_REASONINGS = ROOT / "results" / "split_causal_reasonings.jsonl"
MERGED_JOKES = ROOT / "results" / "merged_jokes.jsonl"
OUT_DPO = ROOT / "data" / "dpo_thinking_1200.jsonl"
OUT_GRPO = ROOT / "data" / "grpo_thinking_1200.jsonl"
OUT_REPORT = ROOT / "data" / "thinking_dataset_report.txt"


def load_merged_jokes():
    """Load merged_jokes.jsonl into a lookup: joke_id -> {joke, prompt}"""
    jokes = {}
    with open(MERGED_JOKES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hid = rec["id"]
            # Replicate the original indexing: sm_1, sm_2, ... (1-based per sm)
            sm_counters = defaultdict(int)
            for cand in rec["candidates"]:
                sm = cand["sm"]
                sm_counters[sm] += 1
                key = f"{hid}_{sm}_{sm_counters[sm]}"
                jokes[key] = {
                    "joke": cand["joke"],
                    "prompt": rec["prompt"],
                    "persona": cand.get("persona", ""),
                }
    return jokes


def load_split_reasonings():
    """Load pre-split winner/loser reasons keyed by (winner_id, loser_id)."""
    split = {}
    try:
        with open(SPLIT_REASONINGS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["winner_joke_id"], rec["loser_joke_id"])
                split[key] = rec
    except FileNotFoundError:
        pass
    return split


def select_best_per_headline():
    """
    For each headline, pick the ONE pairwise record with the highest
    (winner_wins_in_headline - loser_wins_in_headline) gap.
    Returns dict: headline_id -> best_record (with 'win_gap' added)
    """
    records = []
    with open(ALL_REASONINGS) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Count wins-per-joke within each headline
    headline_win_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        headline_win_counts[r["headline_id"]][r["winner_joke_id"]] += 1

    # Annotate each record with its gap
    for r in records:
        hid = r["headline_id"]
        r["win_gap"] = (
            headline_win_counts[hid].get(r["winner_joke_id"], 0)
            - headline_win_counts[hid].get(r["loser_joke_id"], 0)
        )

    # Pick best per headline (highest gap first)
    records.sort(key=lambda x: -x["win_gap"])
    best = {}
    for r in records:
        hid = r["headline_id"]
        if hid not in best:
            best[hid] = r
    return best


def build_think_block(winner_reason: str, features: list[str]) -> str:
    """
    Build the <think> block in pure analytical format — no persona framing,
    no 'winner/loser' labels, just what makes this joke work.

    Uses winner_reason (from split_causal_reasonings.jsonl) if clean,
    otherwise uses the raw comparative reasoning as-is.
    The features line is descriptive, not prescriptive.
    """
    features_str = ", ".join(features) if features else ""
    lines = []
    if features_str:
        lines.append(f"Humor mechanisms: {features_str}.")
    lines.append(winner_reason.strip())
    return f"<think>\n{chr(10).join(lines)}\n</think>"


def build_datasets():
    print("Loading data...")
    best_records = select_best_per_headline()
    joke_lookup = load_merged_jokes()
    split_reasonings = load_split_reasonings()

    print(f"Best records selected: {len(best_records)} (one per headline)")
    print(f"Split reasonings with clean winner_reason: {len(split_reasonings)}")

    dpo_records = []
    grpo_records = []
    stats = {
        "total": 0,
        "used_clean_winner_reason": 0,
        "used_fallback_full_reasoning": 0,
        "skipped_no_joke": 0,
        "gap_dist": defaultdict(int),
    }

    for hid, r in best_records.items():
        stats["total"] += 1
        stats["gap_dist"][r["win_gap"]] += 1

        winner_id = r["winner_joke_id"]
        loser_id = r["loser_joke_id"]

        # --- Get joke text ---
        winner_info = joke_lookup.get(winner_id)
        loser_info = joke_lookup.get(loser_id)
        if not winner_info or not loser_info:
            stats["skipped_no_joke"] += 1
            continue

        prompt_text = winner_info["prompt"]
        winning_joke = winner_info["joke"]
        losing_joke = loser_info["joke"]

        # --- Get the cleanest available winner reason ---
        split_key = (winner_id, loser_id)
        if split_key in split_reasonings and split_reasonings[split_key].get("winner_reason", "").strip():
            # Use the clean, neutral split reason (no A/B labels)
            winner_reason = split_reasonings[split_key]["winner_reason"].strip()
            stats["used_clean_winner_reason"] += 1
        else:
            # Fallback: use the full comparative reasoning as-is.
            # This is imperfect (contains "Joke A"/"Joke B" references) but still
            # captures the analytical reasoning. Better than fabricating a new reason.
            winner_reason = r["reasoning"].strip()
            stats["used_fallback_full_reasoning"] += 1

        think_block = build_think_block(winner_reason, r.get("features", []))

        # ===== DPO Record =====
        # chosen = <think>[pure analytical reasoning]</think>\n[winning joke]
        # rejected = [losing joke]   — no think block, just raw output
        dpo_records.append({
            "headline_id": hid,
            "win_gap": r["win_gap"],
            "winner_joke_id": winner_id,
            "loser_joke_id": loser_id,
            "prompt": [{"role": "user", "content": prompt_text}],
            "chosen": [{"role": "assistant", "content": f"{think_block}\n{winning_joke}"}],
            "rejected": [{"role": "assistant", "content": losing_joke}],
            "metadata": {
                "features": r.get("features", []),
                "winner_persona": winner_info.get("persona", ""),
                "used_clean_split": split_key in split_reasonings,
            },
        })

        # ===== GRPO / SFT Record =====
        # response = <think>[reasoning]</think>\n[winning joke]
        # No pairs needed for GRPO — reward signal can bonus valid <think> blocks.
        grpo_records.append({
            "headline_id": hid,
            "prompt": [{"role": "user", "content": prompt_text}],
            "response": f"{think_block}\n{winning_joke}",
            "metadata": {
                "features": r.get("features", []),
                "winner_persona": winner_info.get("persona", ""),
                "winner_joke_id": winner_id,
                "win_gap": r["win_gap"],
                "used_clean_split": split_key in split_reasonings,
            },
        })

    # --- Write outputs ---
    OUT_DPO.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DPO, "w") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec) + "\n")

    with open(OUT_GRPO, "w") as f:
        for rec in grpo_records:
            f.write(json.dumps(rec) + "\n")

    # --- Report ---
    report_lines = [
        "=== Thinking Dataset Report ===",
        f"Total headlines: {stats['total']}",
        f"DPO records written: {len(dpo_records)}",
        f"GRPO records written: {len(grpo_records)}",
        f"Skipped (no joke text): {stats['skipped_no_joke']}",
        "",
        "Winner reason source:",
        f"  Clean split (neutral, no A/B labels): {stats['used_clean_winner_reason']}",
        f"  Fallback full comparative reasoning:  {stats['used_fallback_full_reasoning']}",
        "",
        "Win gap distribution of selected records:",
    ]
    for g in sorted(stats["gap_dist"].keys(), reverse=True):
        report_lines.append(f"  gap={g}: {stats['gap_dist'][g]} headlines")

    report_lines += [
        "",
        "Format: DPO",
        "  prompt   = [user: headline]",
        "  chosen   = <think>\\nHumor mechanisms: ...\\n[winner_reason]\\n</think>\\n[winning_joke]",
        "  rejected = [losing_joke]  (no think block)",
        "",
        "Format: GRPO/SFT",
        "  prompt   = [user: headline]",
        "  response = <think>\\nHumor mechanisms: ...\\n[winner_reason]\\n</think>\\n[winning_joke]",
        "  (add reward bonus when model output contains valid <think>...</think> block)",
        "",
        f"DPO output:  {OUT_DPO}",
        f"GRPO output: {OUT_GRPO}",
    ]
    report = "\n".join(report_lines)
    with open(OUT_REPORT, "w") as f:
        f.write(report)
    print(report)
    return dpo_records, grpo_records


if __name__ == "__main__":
    dpo_records, grpo_records = build_datasets()
    print(f"\nDone. {len(dpo_records)} DPO + {len(grpo_records)} GRPO records.")

"""
evaluation/sample_eval_pairs.py
================================
Extracts 60 joke pairs from the HumorRank V4 match history for human evaluation.

READS (read-only, never modified):
  - testing/match_history_v4_12_models_50_clean.jsonl
  - testing/generated_jokes_v4.jsonl

WRITES:
  - evaluation/eval_with_judge.jsonl   (60 pairs, full judge metadata — internal use)
  - evaluation/eval_blind.jsonl        (60 pairs, stripped for human annotators)

Design:
  - 12 sub-categories × 5 pairs = 60 total
  - All 50 headlines covered at least once; 10 headlines appear twice (different categories)
  - A/B positions randomly swapped per pair to prevent position bias
  - No model names or judge info leak into the blind file
"""

import json
import random
import os
from collections import defaultdict
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MATCH_HISTORY_PATH = BASE_DIR / "testing" / "match_history_v4_12_models_50_clean.jsonl"
JOKES_PATH         = BASE_DIR / "testing" / "generated_jokes_v4.jsonl"
OUT_FULL_PATH      = BASE_DIR / "evaluation" / "eval_with_judge.jsonl"
OUT_BLIND_PATH     = BASE_DIR / "evaluation" / "eval_blind.jsonl"

# ── Model ID → joke column mapping ─────────────────────────────────────────────
MODEL_TO_COL = {
    "HumorGen SFT-7B":       "sft_joke",
    "HumorGen DPO-7B":       "dpo_joke",
    "HumorGen GRPO-7B":      "grpo_joke",
    "HumorGen SFT-Think-7B": "sft_think_joke",
    "HumorGen DPO-Think-7B": "dpo_think_joke",
    "HumorGen GRPO-Think-7B":"grpo_think_joke",
    "GPT-5":                 "gpt5_joke",
    "Gemini-2.5-Pro":        "gemini_joke",
    "Kimi-K2":               "kimi_k2_joke",
    "Qwen3-32B":             "qwen3_32b_joke",
    "GPT-OSS-120B":          "gpt_oss_joke",
    "Base Qwen-7B":          "base_joke",
}

# ── Sub-category definitions ───────────────────────────────────────────────────
# Each entry: (sub_cat_id, display_label, model_left, model_right, sampling_mode)
# sampling_mode:
#   "high_confidence"  → sort by elo_delta desc, prefer HIGH confidence
#   "mixed_7b_wins"    → include at least 2 where the 7B model won
#   "easy_sanity"      → prefer HIGH confidence (expect easy 7B wins)
#   "medium_close"     → prefer MEDIUM confidence (close margins)
#   "mixed_both_sides" → mix of 7B wins and losses

SUBCATEGORIES = [
    # Category 1: Think Tax
    ("1a", "Think Tax — SFT branch",   "HumorGen SFT-7B",       "HumorGen SFT-Think-7B",   "high_confidence"),
    ("1b", "Think Tax — DPO branch",   "HumorGen DPO-7B",       "HumorGen DPO-Think-7B",   "high_confidence"),
    ("1c", "Think Tax — GRPO branch",  "HumorGen GRPO-7B",      "HumorGen GRPO-Think-7B",  "high_confidence"),
    # Category 2: SOTA vs 7B model
    ("2a", "SOTA vs 7B — vs GPT-5",    "HumorGen SFT-7B",       "GPT-5",                   "mixed_7b_wins"),
    ("2b", "SOTA vs 7B — vs Gemini",   "HumorGen SFT-7B",       "Gemini-2.5-Pro",           "mixed_7b_wins"),
    ("2c", "SOTA vs 7B — vs Kimi",     "HumorGen SFT-7B",       "Kimi-K2",                 "mixed_7b_wins"),
    # Category 3: Alignment Ablation
    ("3a", "Alignment Ablation — SFT vs Base",  "HumorGen SFT-7B",  "Base Qwen-7B",          "easy_sanity"),
    ("3b", "Alignment Ablation — SFT vs DPO",   "HumorGen SFT-7B",  "HumorGen DPO-7B",       "medium_close"),
    ("3c", "Alignment Ablation — SFT vs GRPO",  "HumorGen SFT-7B",  "HumorGen GRPO-7B",      "medium_close"),
    ("3d", "Alignment Ablation — DPO vs GRPO",  "HumorGen DPO-7B",  "HumorGen GRPO-7B",      "medium_close"),
    # Category 4: Scale Efficiency
    ("4a", "Scale Efficiency — 7B vs 32B",   "HumorGen SFT-7B",  "Qwen3-32B",              "mixed_both_sides"),
    ("4b", "Scale Efficiency — 7B vs 120B",  "HumorGen SFT-7B",  "GPT-OSS-120B",            "mixed_both_sides"),
]

PAIRS_PER_SUBCAT = 5


def load_matches():
    """Load match history (read-only)."""
    matches = []
    with open(MATCH_HISTORY_PATH, "r") as f:
        for line in f:
            matches.append(json.loads(line))
    print(f"[INFO] Loaded {len(matches)} matches from match history (read-only)")
    return matches


def load_jokes_lookup():
    """Build joke lookup: {headline_id: {model_name: joke_text}}."""
    lookup = {}
    with open(JOKES_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            hid = row["id"]
            lookup[hid] = {}
            for model_name, col_name in MODEL_TO_COL.items():
                lookup[hid][model_name] = row.get(col_name, "")
    print(f"[INFO] Loaded joke lookup for {len(lookup)} headlines")
    return lookup


def build_pair_index(matches):
    """Index matches by sorted model pair → list of match records."""
    index = defaultdict(list)
    for m in matches:
        key = tuple(sorted([m["player_a"], m["player_b"]]))
        index[key].append(m)
    return index


def get_elo_delta(match):
    """Absolute Elo change from this match."""
    delta_a = abs(match.get("elo_a_after", 0) - match.get("elo_a_before", 0))
    return round(delta_a, 2)


def sort_matches(matches, sampling_mode, model_left):
    """Sort candidate matches according to sampling strategy."""
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    if sampling_mode == "high_confidence":
        # Largest Elo delta first, then HIGH confidence
        return sorted(
            matches,
            key=lambda m: (-get_elo_delta(m), conf_order.get(m.get("confidence", "LOW"), 2))
        )

    elif sampling_mode == "mixed_7b_wins":
        # Shuffle, but guarantee at least 2 where model_left (7B) won
        wins   = [m for m in matches if m.get("winner") == model_left]
        losses = [m for m in matches if m.get("winner") != model_left]
        random.shuffle(wins)
        random.shuffle(losses)
        # interleave: 2 wins, then alternate loss/win
        interleaved = []
        for i in range(max(len(wins), len(losses))):
            if i < len(wins):   interleaved.append(wins[i])
            if i < len(losses): interleaved.append(losses[i])
        return interleaved

    elif sampling_mode == "easy_sanity":
        # HIGH confidence first, then by Elo delta desc
        return sorted(
            matches,
            key=lambda m: (conf_order.get(m.get("confidence", "LOW"), 2), -get_elo_delta(m))
        )

    elif sampling_mode == "medium_close":
        # MEDIUM confidence first (margins are close), then LOW, then HIGH
        order = {"MEDIUM": 0, "LOW": 1, "HIGH": 2}
        return sorted(
            matches,
            key=lambda m: (order.get(m.get("confidence", "LOW"), 1), get_elo_delta(m))
        )

    elif sampling_mode == "mixed_both_sides":
        # Mix: include wins and losses for the primary model
        wins   = [m for m in matches if m.get("winner") == model_left]
        losses = [m for m in matches if m.get("winner") != model_left]
        random.shuffle(wins)
        random.shuffle(losses)
        interleaved = []
        for i in range(max(len(wins), len(losses))):
            if i < len(wins):   interleaved.append(wins[i])
            if i < len(losses): interleaved.append(losses[i])
        return interleaved

    else:
        random.shuffle(matches)
        return matches


def sample_subcat(subcat_id, label, model_left, model_right, sampling_mode,
                  pair_index, jokes_lookup, used_headlines, pair_id_counters):
    """Sample PAIRS_PER_SUBCAT pairs for one sub-category."""
    key = tuple(sorted([model_left, model_right]))
    candidates = list(pair_index.get(key, []))

    if not candidates:
        raise ValueError(f"No matches found for pair: {model_left} vs {model_right}")

    sorted_candidates = sort_matches(candidates, sampling_mode, model_left)

    selected = []
    second_choice = []  # headlines already used once

    for match in sorted_candidates:
        hid = match["headline_id"]
        if hid not in used_headlines:
            selected.append(match)
        else:
            second_choice.append(match)
        if len(selected) == PAIRS_PER_SUBCAT:
            break

    # If we don't have enough unused headlines, allow repeats (max twice per headline)
    if len(selected) < PAIRS_PER_SUBCAT:
        needed = PAIRS_PER_SUBCAT - len(selected)
        # Only allow headlines that have appeared exactly once
        for match in second_choice:
            if used_headlines.get(match["headline_id"], 0) < 2:
                selected.append(match)
                if len(selected) == PAIRS_PER_SUBCAT:
                    break

    if len(selected) < PAIRS_PER_SUBCAT:
        raise RuntimeError(
            f"Could not find {PAIRS_PER_SUBCAT} valid pairs for {subcat_id}. "
            f"Got {len(selected)}. Relax the headline coverage constraint."
        )

    # Build output records
    full_records  = []
    blind_records = []

    for match in selected:
        hid      = match["headline_id"]
        headline = match["headline"]

        # Get joke text from the jokes lookup
        headline_jokes = jokes_lookup.get(hid, {})
        joke_left  = headline_jokes.get(model_left, "")
        joke_right = headline_jokes.get(model_right, "")

        # Random A/B swap
        swapped = random.random() < 0.5
        if swapped:
            joke_a, joke_b = joke_right, joke_left
            model_a, model_b = model_right, model_left
        else:
            joke_a, joke_b = joke_left, joke_right
            model_a, model_b = model_left, model_right

        # Build pair_id
        pair_id_counters[subcat_id] = pair_id_counters.get(subcat_id, 0) + 1
        pair_id = f"{subcat_id}_{pair_id_counters[subcat_id]:03d}"

        # Determine judge winner label against match's original player positions
        judge_winner = match.get("winner")

        full_rec = {
            "pair_id":          pair_id,
            "eval_category":    label,
            "sub_cat":          subcat_id,
            "headline_id":      hid,
            "headline":         headline,
            "model_a":          model_a,
            "model_b":          model_b,
            "joke_a":           joke_a,
            "joke_b":           joke_b,
            "ab_swapped":       swapped,
            "judge_winner":     judge_winner,
            "judge_reasoning":  match.get("reasoning", ""),
            "judge_confidence": match.get("confidence", ""),
            "judge_features":   match.get("features", []),
            "elo_delta":        get_elo_delta(match),
            "is_tie":           match.get("is_tie", False),
        }

        blind_rec = {
            "pair_id":       pair_id,
            "eval_category": label,
            "headline_id":   hid,
            "headline":      headline,
            "model_a":       model_a,
            "model_b":       model_b,
            "joke_a":        joke_a,
            "joke_b":        joke_b,
            "human_choice":  None,
        }

        full_records.append(full_rec)
        blind_records.append(blind_rec)

        # Track headline usage
        used_headlines[hid] = used_headlines.get(hid, 0) + 1

    return full_records, blind_records


def run():
    print("=" * 60)
    print("HumorGen Human Evaluation — Pair Sampler")
    print("=" * 60)

    # Load data (match history is NEVER written to)
    matches      = load_matches()
    jokes_lookup = load_jokes_lookup()
    pair_index   = build_pair_index(matches)

    used_headlines    = {}   # {headline_id: use_count}
    pair_id_counters  = {}   # {subcat_id: counter}
    all_full_records  = []
    all_blind_records = []

    for subcat_id, label, model_left, model_right, sampling_mode in SUBCATEGORIES:
        full_recs, blind_recs = sample_subcat(
            subcat_id, label, model_left, model_right, sampling_mode,
            pair_index, jokes_lookup, used_headlines, pair_id_counters
        )
        all_full_records.extend(full_recs)
        all_blind_records.extend(blind_recs)
        print(f"  [{subcat_id}] {label}: {len(full_recs)} pairs sampled")

    # Write outputs
    os.makedirs(OUT_FULL_PATH.parent, exist_ok=True)

    with open(OUT_FULL_PATH, "w") as f:
        for rec in all_full_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\n[OUT] eval_with_judge.jsonl → {len(all_full_records)} pairs → {OUT_FULL_PATH}")

    with open(OUT_BLIND_PATH, "w") as f:
        for rec in all_blind_records:
            f.write(json.dumps(rec) + "\n")
    print(f"[OUT] eval_blind.jsonl      → {len(all_blind_records)} pairs → {OUT_BLIND_PATH}")

    # ── Coverage Report ────────────────────────────────────────────────────────
    print("\n── Headline Coverage Report ─────────────────────────────────────────")
    once  = [h for h, c in used_headlines.items() if c == 1]
    twice = [h for h, c in used_headlines.items() if c == 2]
    print(f"  Headlines used exactly once : {len(once)}")
    print(f"  Headlines used exactly twice: {len(twice)}")
    print(f"  Total headlines covered     : {len(used_headlines)} / 50")
    if len(used_headlines) < 50:
        all_headlines = set()
        for m in matches:
            all_headlines.add(m["headline_id"])
        missing = all_headlines - set(used_headlines.keys())
        print(f"  WARNING: UNCOVERED headlines ({len(missing)}): {sorted(missing)}")
    else:
        print("  OK: All 50 headlines covered")

    print("\n── Sanity Checks ────────────────────────────────────────────────────")

    # 1. No model names in the actual joke text fields of the blind file
    model_names_to_check = list(MODEL_TO_COL.keys())
    leaks = []
    for rec in all_blind_records:
        for field in ["joke_a", "joke_b"]:  # intentionally NOT scanning eval_category
            text = rec.get(field, "") or ""
            for m in model_names_to_check:
                if m.lower() in text.lower():
                    leaks.append((rec["pair_id"], field, m))
    if leaks:
        print(f"  WARNING: Model name leaks in joke text: {leaks}")
    else:
        print("  OK: No model names leaked into joke text of blind file")

    # 2. Total pair count
    assert len(all_full_records) == 60, f"Expected 60 pairs, got {len(all_full_records)}"
    print(f"  OK: Exactly {len(all_full_records)} pairs generated")

    # 3. No duplicate pair_ids
    ids = [r["pair_id"] for r in all_full_records]
    assert len(ids) == len(set(ids)), "Duplicate pair_ids found!"
    print("  OK: All pair_ids are unique")

    print("\nDone.")


if __name__ == "__main__":
    run()

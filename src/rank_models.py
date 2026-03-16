#!/usr/bin/env python3
"""
Full HumorRank pipeline test on generated_jokes_v5.jsonl data (50 headlines, 13 models).
Includes Think models and the new Comedian SFT ablation.
Outputs results to the testing/ directory as v5.

Usage:
    python -m src.rank_models   # or: python humorrank/test_full_v5.py from repo root
"""

import os
import sys
import json
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "testing")
LOG_FILE = os.path.join(OUT_DIR, "humorrank_v5_13_models_50.log")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# Groq keys from env: GROQ_API_KEY or comma-separated GROQ_API_KEYS
_raw = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
GROQ_API_KEYS = [k.strip() for k in _raw.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise ValueError("Set GROQ_API_KEY or GROQ_API_KEYS in environment for the judge.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEYS[0]
logger.info(f"Loaded {len(GROQ_API_KEYS)} Groq API key(s)")

from humorrank.judge.api_judge import APIJudge
from humorrank.tournament.runner import Tournament
from humorrank.tournament.bradley_terry import BradleyTerry
from humorrank.tournament.elo import EloRating
from humorrank.config import TournamentConfig
from humorrank.analysis.bias import BiasAnalyzer
from humorrank.analysis.features import FeatureAnalyzer

# ── All 10 models ────────────────────────────────────────────────────────
ALL_MODELS = {
    "kimi_k2_joke": "Kimi-K2",
    "dpo_joke": "HumorGen DPO-7B",
    "grpo_joke": "HumorGen GRPO-7B",
    "dpo_think_joke": "HumorGen DPO-Think-7B",
    "grpo_think_joke": "HumorGen GRPO-Think-7B",
    "sft_joke": "HumorGen SFT-7B",
    "sft_think_joke": "HumorGen SFT-Think-7B",
    "gpt_oss_joke": "GPT-OSS-120B",
    "qwen3_32b_joke": "Qwen3-32B",
    "gpt5_joke": "GPT-5",
    "gemini_joke": "Gemini-2.5-Pro",
    "base_joke": "Base Qwen-7B",
    "comedian_joke": "HumorGen-Com-7B",
}

# ── Load ALL headlines ──────────────────────────────────────────────────
JOKES_PATH = os.path.join(PROJECT_ROOT, "testing", "generated_jokes_v5.jsonl")
entries = []
with open(JOKES_PATH) as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))
            if len(entries) >= 50:
                break

model_names = list(ALL_MODELS.values())
num_pairs = len(model_names) * (len(model_names) - 1) // 2
logger.info(f"Headlines: {len(entries)} | Models: {len(model_names)} | Pairs/headline: {num_pairs}")
logger.info(f"Total matches expected: {len(entries) * num_pairs}")

# ── Judge ───────────────────────────────────────────────────────────────
judge = APIJudge(
    model_name="llama-3.3-70b-versatile",
    provider="groq",
    api_keys=GROQ_API_KEYS,
)

# ── Load existing progress ───────────────────────────────────────────
match_history_path = os.path.join(OUT_DIR, "match_history_v5_13_models_50.jsonl")
completed_headlines = set()
headline_pairs = {}   # hid -> set of frozenset(player_a, player_b)

if os.path.exists(match_history_path):
    headline_counts = {}
    with open(match_history_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    m = json.loads(line)
                    hid = m.get("headline_id")
                    if hid and m.get("confidence") != "ERROR":
                        pair = frozenset([m.get("player_a", ""), m.get("player_b", "")])
                        if hid not in headline_pairs:
                            headline_pairs[hid] = set()
                        headline_pairs[hid].add(pair)
                        headline_counts[hid] = len(headline_pairs[hid])
                except: continue
    
    for hid, count in headline_counts.items():
        if count >= 78:
            completed_headlines.add(hid)
            
    logger.info(f"Resuming: {len(completed_headlines)} headlines fully done, "
                f"{len(headline_counts) - len(completed_headlines)} partially done.")

# ── Run tournament per headline, stream results to file ─────────────────
config = TournamentConfig(k_factor=16, stable_elo_shuffles=10)
tournament = Tournament(config)

all_matches = []
# Load existing matches into memory for ranking at the end
if os.path.exists(match_history_path):
    with open(match_history_path, 'r') as f:
        for line in f:
            if line.strip():
                try: all_matches.append(json.loads(line))
                except: continue

start_time = time.time()

# Open in append mode 'a'
with open(match_history_path, "a") as match_f:
    for i, entry in enumerate(entries):
        if entry["id"] in completed_headlines:
            continue
            
        headline = entry["headline"]
        t0 = time.time()

        jokes = {}
        for col, name in ALL_MODELS.items():
            jokes[name] = entry.get(col, "[NO JOKE]")

        # Build set of pairs already done for this headline (for partial resume)
        already_done = set()
        if entry["id"] in headline_pairs:
            already_done = {tuple(sorted(p)) for p in headline_pairs[entry["id"]]}

        if already_done:
            logger.info(f"  Partial resume: {len(already_done)} pairs already done, "
                        f"{78 - len(already_done)} remaining.")

        result = tournament.run_round_robin(
            player_ids=model_names,
            contents=jokes,
            headline=headline,
            judge=judge,
            skip_pairs=already_done if already_done else None,
        )

        for m in result["match_history"]:
            m["headline_id"] = entry["id"]
            m["headline"] = headline
            match_f.write(json.dumps(m) + "\n")
            
            # STOP ONLY IF API EXHAUSTED (recorded as ERROR with specific reasoning)
            if m.get("confidence") == "ERROR":
                reason = m.get("reasoning", "").lower()
                if "api keys exhausted" in reason or "global retries" in reason:
                    match_f.flush()
                    logger.error("!!! REAL API EXHAUSTION DETECTED !!!")
                    logger.error(f"Reason: {m.get('reasoning')}")
                    logger.error("Stopping execution to prevent invalid results.")
                    sys.exit(1)
                else:
                    logger.warning(f"Judge Parse Error on headline {entry['id']}: {m.get('reasoning')[:200]}...")

        match_f.flush()

        all_matches.extend(result["match_history"])

        wins_this = {}
        for m in result["match_history"]:
            w = m["winner"] or "TIE"
            wins_this[w] = wins_this.get(w, 0) + 1

        elapsed = time.time() - t0
        top_winner = max(wins_this, key=wins_this.get) if wins_this else "?"
        logger.info(
            f"[{i+1:2d}/{len(entries)}] {entry['id']} | {elapsed:.1f}s | "
            f"{len(result['match_history'])} matches | top: {top_winner}"
        )

total_time = time.time() - start_time
logger.info(f"Finished evaluation loop in {total_time:.1f}s ({total_time/60:.1f}min)")

# ── Deduplicate all_matches for airtight resumption ──────────────────
# If a headline failed part-way, resuming it generates all 78 pairs again.
# We deduplicate by (headline_id, player_a, player_b) to keep only the latest.
deduped_matches = []
seen_matchups = set()
for m in reversed(all_matches):
    if m.get("confidence") == "ERROR": continue
    hid = m.get("headline_id") or m.get("headline", "")
    players = tuple(sorted([m.get("player_a", ""), m.get("player_b", "")]))
    key = (hid, players)
    if key not in seen_matchups:
        seen_matchups.add(key)
        deduped_matches.append(m)

all_matches = list(reversed(deduped_matches))
logger.info(f"Unique valid matches across all runs: {len(all_matches)}")

# ── 1. Match Matrix/History ──────────────────────────────────────────
match_data_path = os.path.join(OUT_DIR, "match_history_v5_13_models_50.jsonl")

# ── 2. Bradley-Terry ─────────────────────────────────────────────────
bt = BradleyTerry()
bt_results = bt.fit_with_ci(all_matches, model_names, num_bootstrap=100)

# ── 3. Elo ───────────────────────────────────────────────────────────
elo = EloRating(k_factor=16)
stable_elo = elo.compute_stable(all_matches, model_names, num_shuffles=config.stable_elo_shuffles)

# ── 4. Position Bias ─────────────────────────────────────────────────
bias_analyzer = BiasAnalyzer(all_matches)
bias_stats = bias_analyzer.position_bias()

# ── 5. Feature Analysis ──────────────────────────────────────────────
feature_analyzer = FeatureAnalyzer(all_matches)
feature_stats = feature_analyzer.feature_win_rates()

# ── Report ───────────────────────────────────────────────────────────
logger.info("\n" + "="*90)
logger.info(f"{'HUMORRANK V5 LEADERBOARD (50 Headlines)':^90}")
logger.info("="*90)
logger.info(f"{'Rank':<6} {'Model':<20} {'BT Rating':<12} {'95% CI':<24} {'Stable ELO':<12} {'W':<6} {'L':<6} {'T':<6} {'Win%'}")
logger.info("-" * 90)

stats = {m: {"W": 0, "L": 0, "T": 0, "G": 0} for m in model_names}
for m in all_matches:
    a, b = m["player_a"], m["player_b"]
    stats[a]["G"] += 1
    stats[b]["G"] += 1
    if m.get("is_tie"):
        stats[a]["T"] += 1
        stats[b]["T"] += 1
    elif m["score_a"] == 1.0:
        stats[a]["W"] += 1
        stats[b]["L"] += 1
    else:
        stats[b]["W"] += 1
        stats[a]["L"] += 1

ranked = sorted(model_names, key=lambda m: bt_results[m]["rating"], reverse=True)
leaderboard = []

for i, name in enumerate(ranked):
    r = bt_results[name]
    s = stats[name]
    wlt = f"{s['W']}-{s['L']}-{s['T']}"
    wr = s['W'] / s['G'] * 100 if s['G'] > 0 else 0
    ci_str = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
    
    logger.info(f"{i+1:<6} {name:<20} {r['rating']:<12.2f} {ci_str:<24} {stable_elo[name]:<12.1f} {s['W']:<6} {s['L']:<6} {s['T']:<6} {wr:.1f}")
    
    leaderboard.append({
        "model": name,
        "bt_rating": r["rating"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "stable_elo": stable_elo[name],
        "wins": s["W"],
        "losses": s["L"],
        "ties": s["T"],
        "total_games": s["G"],
        "win_rate": wr
    })

logger.info("="*90)

# Save results
lb_path = os.path.join(OUT_DIR, "leaderboard_v5_13_models_50.json")
with open(lb_path, "w") as f:
    json.dump(leaderboard, f, indent=2)

logger.info("\n=== LLM POSITION BIAS (raw A vs B as LLM sees them) ===")
ba = BiasAnalyzer(all_matches)
pb = ba.position_bias()
logger.info(f"  LLM picked first: {pb['llm_picked_first']} | second: {pb['llm_picked_second']} | ties: {pb['ties']}")
logger.info(f"  First rate: {pb['first_rate']:.1%} | Second rate: {pb['second_rate']:.1%} | Tie rate: {pb['tie_rate']:.1%}")

# ── Save leaderboard JSON ──────────────────────────────────────────────
leaderboard_data = []
for name in ranked:
    r = bt_results[name]
    s = stats[name]
    leaderboard_data.append({
        "model": name,
        "bt_rating": r["rating"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "stable_elo": round(stable_elo[name], 2),
        "wins": s["W"],
        "losses": s["L"],
        "ties": s["T"],
        "total_games": s["G"],
        "win_rate": round(s["W"] / s["G"] * 100, 1) if s["G"] else 0,
    })

with open(os.path.join(OUT_DIR, "leaderboard_v5_13_models_50.json"), "w") as f:
    json.dump(leaderboard_data, f, indent=2)

logger.info(f"\nResults saved to {OUT_DIR}/")
logger.info("Full pipeline test v5 (13 models, 50 headlines) complete.")

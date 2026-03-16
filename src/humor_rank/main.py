import os
import sys
import argparse
import datetime
from typing import List, Dict, Any, Optional

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.humor_rank.judge import HumorJudge, LocalHuggingFaceJudge
from src.humor_rank.tournament import EloTournament
from src.humor_rank.scoring import normalize_scores, generate_dpo_pairs, generate_grpo_rewards
from src.humor_rank.utils import extract_jokes, save_jsonl, save_json, load_jsonl
from src.humor_rank.checkpoint import CheckpointManager

from functools import partial
from concurrent.futures import ProcessPoolExecutor

import logging
import time

# Logging Setup
def setup_logging(output_dir: str):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"humorrank_run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

from concurrent.futures import ThreadPoolExecutor

def process_headline(headline_data: Dict[str, Any], judge: HumorJudge, tournament: EloTournament, config: Dict[str, Any], lock: Any = None) -> Dict[str, Any]:
    headline_id = headline_data["id"]
    headline = headline_data["prompt"]
    
    # Simple console log to show progress
    if lock:
        with lock:
            logger.info(f"Processing Headline ID: {headline_id}")
    else:
        logger.info(f"Processing Headline ID: {headline_id}")
    
    jokes = extract_jokes(headline_data)
    if not jokes:
        return None
        
    joke_ids = [j[0] for j in jokes]
    jokes_content = {j[0]: j[1] for j in jokes}
    
    # Initialize state
    elo = {jid: config['initial_rating'] for jid in joke_ids}
    wins = {jid: 0 for jid in joke_ids}
    ties = {jid: 0 for jid in joke_ids}
    games = {jid: 0 for jid in joke_ids}
    opponents_played = {jid: set() for jid in joke_ids}
    match_history = []
    
    elo_history = []
    comparisons_made = 0
    converged = False
    
    # Run adaptive rounds
    for r in range(config['max_rounds']):
        # logger.info(f"  > Round {r+1} start...")
        pairs = tournament.swiss_pairing(joke_ids, elo, wins, games, opponents_played)
        
        comps = tournament.run_round(pairs, jokes_content, headline, judge, elo, wins, games, opponents_played, ties, match_history)
        comparisons_made += comps
        
        current_elo_state = elo.copy()
        elo_history.append(current_elo_state)
        
        # Check convergence safely
        if r + 1 >= config['min_rounds'] and len(elo_history) >= 2:
            last = elo_history[-1]
            prev = elo_history[-2]
            max_change = max(abs(last[j] - prev[j]) for j in last)
            
            if max_change < config['convergence_threshold']:
                converged = True
                break
                
    # Compute Stable ELO (Averaged over 5 shuffles)
    stable_elo = tournament.compute_stable_elo(match_history, joke_ids, num_shuffles=5)
    
    # Finalize scores using Stable ELO
    normalized = normalize_scores(stable_elo, wins, games)
    
    # Generate subsets
    dpo_pairs = generate_dpo_pairs(joke_ids, normalized, jokes_content, headline, headline_id, config['min_dpo_gap'])
    grpo_rewards = generate_grpo_rewards(joke_ids, stable_elo, normalized, jokes_content, headline, headline_id)
    
    if lock:
        with lock:
            logger.info(f"Finished {headline_id}. Total Comparisons: {comparisons_made}. DPO Pairs: {len(dpo_pairs)}")
    else:
        logger.info(f"Finished {headline_id}. Total Comparisons: {comparisons_made}. DPO Pairs: {len(dpo_pairs)}")
    
    return {
        "ranking": {
            "headline_id": headline_id,
            "headline": headline,
            "ranking": sorted(joke_ids, key=lambda j: normalized[j], reverse=True),
            "elo_scores": stable_elo, # Use stable ELO
            "raw_elo_scores": elo,    # Keep original ELO for reference
            "normalized_scores": normalized,
            "win_counts": wins,
            "tie_counts": ties,
            "games_played": games,
            "comparisons_made": comparisons_made,
            "converged": converged,
            "match_history": match_history
        },
        "dpo_pairs": dpo_pairs,
        "grpo_rewards": grpo_rewards
    }

def main():
    parser = argparse.ArgumentParser(description="HumorRank ELO System")
    parser.add_argument("--input", default="results/merged_jokes.jsonl")
    parser.add_argument("--output_dir", default="results/humor_rank")
    parser.add_argument("--judge_model", default="llama-3.3-70b-versatile")
    parser.add_argument("--limit", type=int, help="Limit number of headlines to process")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N headlines")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads (1=Sequential, >1=Parallel)")
    parser.add_argument("--local_execution", action="store_true", help="Use in-process local model loading via model_utils")
    parser.add_argument("--shard_idx", type=int, default=0, help="Index of current shard (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    # Config
    config = {
        'initial_rating': 1000,
        'k_factor': 36,     # Increased from 32 to compensate for single-call variance
        'min_rounds': 2,    # Reduced from 5 (see experiments.md Issue #2)
        'max_rounds': 3,    # Reduced from 8 (see experiments.md Issue #2)
        'convergence_threshold': 5.0,
        'min_dpo_gap': 15,
        'checkpoint_interval': 1,
        'judge_model': args.judge_model
    }
    
    logger.info(f"Starting HumorRank with Config: {config}")
    
    data = load_jsonl(args.input)
    if not data:
        logger.error(f"No data found in {args.input}")
        return
        
    if args.skip:
        data = data[args.skip:]
        logger.info(f"Skipping first {args.skip} items. Remaining: {len(data)}")

    if args.limit:
        data = data[:args.limit]
        
    checkpoint = CheckpointManager(os.path.join(args.output_dir, "checkpoint.json"))
    state = checkpoint.load()
    processed_ids = set(state["processed_ids"])
    
    # Sort to ensure deterministic sharding
    data.sort(key=lambda x: x["id"]) 
    
    # Sharding Logic
    if args.num_shards > 1:
        # data[i] for i in range(len(data)) if i % num_shards == shard_idx
        # But we must respect processed_ids.
        # Strategy: Shard the *whole* dataset first, then filter processed.
        my_shard_data = [d for i, d in enumerate(data) if i % args.num_shards == args.shard_idx]
        logger.info(f"Shard {args.shard_idx}/{args.num_shards}: Assigned {len(my_shard_data)} of {len(data)} total records.")
        data = my_shard_data

    to_process = [item for item in data if item["id"] not in processed_ids]
    logger.info(f"Loaded {len(data)} headlines (Shard {args.shard_idx}). {len(processed_ids)} already processed (in this shard). Starting processing on {len(to_process)} items with {args.workers} threads.")
    
    rankings_path = os.path.join(args.output_dir, "humorrank_rankings.jsonl")
    dpo_path = os.path.join(args.output_dir, "humorrank_dpo_pairs.jsonl")
    grpo_path = os.path.join(args.output_dir, "humorrank_grpo_rewards.jsonl")
    history_path = os.path.join(args.output_dir, "humorrank_match_history.jsonl")
    
    # Initialize Judge and Tournament ONCE
    if args.local_execution or os.path.exists(config['judge_model']):
        logger.info(f"Instantiating Local In-Process Judge (using model_utils) for model: {config['judge_model']}")
        judge = LocalHuggingFaceJudge(model_name=config['judge_model'])
    else:
        logger.info(f"Instantiating API/Server Judge for model: {config['judge_model']}")
        judge = HumorJudge(model_name=config['judge_model'])
        
    tournament = EloTournament(k_factor=config['k_factor'], initial_rating=config['initial_rating'])
    
    import threading
    log_lock = threading.Lock()
    
    # Process Function
    def worker_func(item):
        try:
            return process_headline(item, judge, tournament, config, log_lock)
        except Exception as e:
            logger.error(f"Error processing {item.get('id', 'unknown')}: {e}")
            return None

    if args.workers == 1:
        # Sequential Processing with TQDM
        from tqdm import tqdm
        for i, item in enumerate(tqdm(to_process, desc="Headlines", unit="prompt")):
            result = worker_func(item)
            if result:
                # Separate Match History
                ranking_data = result["ranking"].copy()
                match_history_data = {
                    "headline_id": ranking_data["headline_id"],
                    "match_history": ranking_data.pop("match_history") 
                }
                
                # Save Separately
                save_jsonl([ranking_data], rankings_path, append=True)
                save_jsonl([match_history_data], history_path, append=True)
                
                # Save derived datasets
                save_jsonl(result["dpo_pairs"], dpo_path, append=True)
                save_jsonl(result["grpo_rewards"], grpo_path, append=True)
                
                processed_ids.add(ranking_data["headline_id"])
            
            if (i + 1) % config['checkpoint_interval'] == 0:
                logger.info(f"Checkpointed at {len(processed_ids)} processed headlines.")
                checkpoint.save(list(processed_ids), {})
    else:
        # Threaded Processing with TQDM
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {executor.submit(worker_func, item): item["id"] for item in to_process}
            
            from concurrent.futures import as_completed
            for i, future in enumerate(tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Headlines", unit="prompt")):
                result = future.result()
                if result:
                    # Separate Match History
                    ranking_data = result["ranking"].copy()
                    match_history_data = {
                        "headline_id": ranking_data["headline_id"],
                        "match_history": ranking_data.pop("match_history") 
                    }
                    
                    # Save Separately
                    save_jsonl([ranking_data], rankings_path, append=True)
                    save_jsonl([match_history_data], history_path, append=True)
                    
                    save_jsonl(result["dpo_pairs"], dpo_path, append=True)
                    save_jsonl(result["grpo_rewards"], grpo_path, append=True)
                    processed_ids.add(ranking_data["headline_id"])
                
                if (i + 1) % config['checkpoint_interval'] == 0:
                    logger.info(f"Checkpointed at {len(processed_ids)} processed headlines.")
                    checkpoint.save(list(processed_ids), {})

    checkpoint.save(list(processed_ids), {})
    logger.info("Ranking complete.")
    
if __name__ == "__main__":
    main()

from typing import List, Dict, Any

def normalize_scores(elo: Dict[str, float], wins: Dict[str, int], games: Dict[str, int]) -> Dict[str, float]:
    """
    Standardize ELO and Win dominance into a 0-100 scale for GRPO.
    """
    if not elo:
        return {}
        
    min_elo = min(elo.values())
    max_elo = max(elo.values())
    max_wins = max(wins.values()) if wins else 1
    
    normalized = {}
    for joke_id in elo:
        # Position Weight
        if max_elo == min_elo:
            pos_weight = 0.5
        else:
            pos_weight = (elo[joke_id] - min_elo) / (max_elo - min_elo)
            
        # Win Dominance
        win_dom = wins[joke_id] / max_wins if max_wins > 0 else 0
        
        # Confidence Scaling (at least 5 games for full score stability)
        conf = min(games[joke_id] / 5, 1.0)
        
        # Combined score (60% ELO position, 40% Win rate) * confidence
        score = (pos_weight * 0.6 + win_dom * 0.4) * conf * 100
        normalized[joke_id] = round(score, 1)
        
    return normalized

def generate_dpo_pairs(joke_ids: List[str], normalized_scores: Dict[str, float], jokes_content: Dict[str, str], headline: str, headline_id: str, min_gap: float = 15.0) -> List[Dict[str, Any]]:
    """
    Generates strong (Top/Bottom) and subtle (Adjacent) DPO pairs.
    """
    pairs = []
    sorted_ids = sorted(joke_ids, key=lambda j: normalized_scores[j], reverse=True)
    
    # Strong Pairs: Top 3 vs Bottom 3
    for chosen_id in sorted_ids[:3]:
        for rejected_id in sorted_ids[-3:]:
            gap = normalized_scores[chosen_id] - normalized_scores[rejected_id]
            if gap >= min_gap:
                pairs.append({
                    "prompt": f"Write a funny joke about: {headline}",
                    "chosen": jokes_content[chosen_id],
                    "rejected": jokes_content[rejected_id],
                    "elo_gap": gap,
                    "headline_id": headline_id
                })
                
    # Adjacent Pairs: Nuanced signal iff gap >= 10
    for i in range(len(sorted_ids) - 1):
        chosen_id = sorted_ids[i]
        rejected_id = sorted_ids[i+1]
        gap = normalized_scores[chosen_id] - normalized_scores[rejected_id]
        if gap >= 10.0:
            pairs.append({
                "prompt": f"Write a funny joke about: {headline}",
                "chosen": jokes_content[chosen_id],
                "rejected": jokes_content[rejected_id],
                "elo_gap": gap,
                "headline_id": headline_id
            })
            
    return pairs

def generate_grpo_rewards(joke_ids: List[str], elo_scores: Dict[str, float], normalized_scores: Dict[str, float], jokes_content: Dict[str, str], headline: str, headline_id: str) -> List[Dict[str, Any]]:
    """
    Generates training data for GRPO with normalized rewards.
    """
    sorted_ids = sorted(joke_ids, key=lambda j: normalized_scores[j], reverse=True)
    ranks = {jid: i+1 for i, jid in enumerate(sorted_ids)}
    
    rewards = []
    for jid in joke_ids:
        # Scale score to [0, 1] for reward
        reward = normalized_scores[jid] / 100.0
        rewards.append({
            "prompt": f"Write a funny joke about: {headline}",
            "joke": jokes_content[jid],
            "reward": reward,
            "elo": elo_scores[jid],
            "rank": ranks[jid],
            "headline_id": headline_id
        })
    return rewards

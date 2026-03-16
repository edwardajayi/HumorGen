import random
from typing import List, Dict, Any, Set, Tuple
from tqdm import tqdm

class EloTournament:
    def __init__(self, k_factor: int = 32, initial_rating: float = 1000.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def update_elo(self, rating_a: float, rating_b: float, score_a: float) -> Tuple[float, float]:
        """
        score_a should be 1.0 for win, 0.5 for tie, 0.0 for loss.
        """
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a
        
        score_b = 1.0 - score_a
        
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)
        
        return new_rating_a, new_rating_b

    def swiss_pairing(self, joke_ids: List[str], elo: Dict[str, float], wins: Dict[str, int], games: Dict[str, int], opponents_played: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """
        Pairs jokes based on current ELO and wins while avoiding rematches.
        """
        # Sort by primary: wins (desc), secondary: games (asc), tertiary: elo (desc)
        sorted_jokes = sorted(
            joke_ids,
            key=lambda j: (-wins[j], games[j], -elo[j])
        )
        
        pairs = []
        used = set()
        
        for i, joke_a in enumerate(sorted_jokes):
            if joke_a in used:
                continue
            
            # Find best candidate: closest in rank, hasn't played before
            best_match = None
            for j in range(i + 1, len(sorted_jokes)):
                joke_b = sorted_jokes[j]
                if joke_b in used:
                    continue
                
                if joke_b not in opponents_played[joke_a]:
                    best_match = joke_b
                    break
            
            # If no new opponent, take the closest available (rematch)
            if best_match is None:
                for j in range(i + 1, len(sorted_jokes)):
                    joke_b = sorted_jokes[j]
                    if joke_b not in used:
                        best_match = joke_b
                        break
            
            if best_match:
                pairs.append((joke_a, best_match))
                used.add(joke_a)
                used.add(best_match)
        
        return pairs

    def run_round(self, pairs: List[Tuple[str, str]], jokes_content: Dict[str, str], headline: str, judge: Any, 
                  elo: Dict[str, float], wins: Dict[str, int], games: Dict[str, int], opponents_played: Dict[str, Set[str]],
                  ties: Dict[str, int], match_history: List[Dict[str, Any]]) -> int:
        """
        Executes a round of matches and updates ELO. Returns count of comparisons made.
        """
        api_calls = 0
        
        # Use tqdm for progress tracking within the round
        for id_a, id_b in tqdm(pairs, desc="  Matches", unit="pair", leave=False):
            joke_a = jokes_content[id_a]
            joke_b = jokes_content[id_b]
            
            # Record ELO before update for history
            elo_a_before = elo[id_a]
            elo_b_before = elo[id_b]

            # Pass IDs for robust tracking
            result = judge.compare(joke_a, joke_b, headline, id_a, id_b)
            api_calls += 1  # Single call per comparison (see experiments.md Issue #2)
            
            score_a = 0.5
            winner_id = None
            is_tie = False
            
            if result["is_tie"] or result["confidence"] == "ERROR":
                score_a = 0.5
                is_tie = True
                ties[id_a] += 1
                ties[id_b] += 1
            else:
                winner_id = result["winner_id"]
                score_a = 1.0 if winner_id == id_a else 0.0
                if score_a == 1.0:
                    wins[id_a] += 1
                else:
                    wins[id_b] += 1
            
            ra, rb = self.update_elo(elo[id_a], elo[id_b], score_a)
            elo[id_a], elo[id_b] = ra, rb
            
            games[id_a] += 1
            games[id_b] += 1
            opponents_played[id_a].add(id_b)
            opponents_played[id_b].add(id_a)
            
            # Record Match History
            match_history.append({
                "joke_a_id": id_a,
                "joke_b_id": id_b,
                "winner_id": winner_id,
                "is_tie": is_tie,
                "confidence": result["confidence"],
                "score_a": score_a,
                "elo_a_before": elo_a_before,
                "elo_b_before": elo_b_before,
                "elo_a_after": ra,
                "elo_b_after": rb,
                "reasoning": result.get("reasoning", ""),
                "features": result.get("features", [])
            })
            
        return api_calls

    def compute_stable_elo(self, match_results: List[Dict[str, Any]], joke_ids: List[str], 
                           num_shuffles: int = 5, seed: int = 42) -> Dict[str, float]:
        """
        Re-compute ELO by replaying matches in different random orders to average out order effects.
        """
        final_elos = {jid: 0.0 for jid in joke_ids}
        
        for i in range(num_shuffles):
            # Reset temporary ELOs
            temp_elo = {jid: self.initial_rating for jid in joke_ids}
            
            # Shuffle matches deterministically
            rng = random.Random(seed + i)
            shuffled_matches = list(match_results)
            rng.shuffle(shuffled_matches)
            
            for match in shuffled_matches:
                id_a = match['joke_a_id']
                id_b = match['joke_b_id']
                score_a = match['score_a']
                
                ra, rb = self.update_elo(temp_elo[id_a], temp_elo[id_b], score_a)
                temp_elo[id_a] = ra
                temp_elo[id_b] = rb
            
            # Accumulate
            for jid in joke_ids:
                final_elos[jid] += temp_elo[jid]
        
        # Average across shuffles
        for jid in joke_ids:
            final_elos[jid] /= num_shuffles

        return final_elos


class BradleyTerry:
    """
    Bradley-Terry model for ranking, as used by LMSYS Chatbot Arena (Dec 2023+).

    Unlike sequential Elo, BT uses global Maximum Likelihood Estimation (MLE)
    over all match results at once. This means:
      - Order-independent: no match-order bias
      - Ties handled natively: each tie adds 0.5 to both models' win counts
      - Statistically grounded: converges to the most likely strength ranking

    Usage:
        bt = BradleyTerry()
        bt_elo = bt.fit(match_results, model_ids)
        # bt_elo[model_id] = Elo-style rating (centered at 1000)
    """

    def __init__(self, initial_rating: float = 1000.0, scale: float = 400.0):
        self.initial_rating = initial_rating
        self.scale = scale  # Elo scaling: rating = 1000 + scale * log10(strength)

    def fit(self, match_results: List[Dict[str, Any]], model_ids: List[str],
            max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, float]:
        """
        Fit Bradley-Terry model using the MM (minorization-maximization) algorithm.

        match_results: list of dicts with keys:
            joke_a_id, joke_b_id, score_a
            (score_a = 1.0 win, 0.5 tie, 0.0 loss for model_a)
        model_ids: list of all model identifiers
        """
        import math

        n = len(model_ids)
        idx = {m: i for i, m in enumerate(model_ids)}

        # Build win matrix: W[i][j] = wins of i over j (ties count as 0.5)
        W = [[0.0] * n for _ in range(n)]
        N = [[0.0] * n for _ in range(n)]  # total games between i and j

        for m in match_results:
            a, b = m["joke_a_id"], m["joke_b_id"]
            if a not in idx or b not in idx:
                continue
            i, j = idx[a], idx[b]
            score_a = m.get("score_a", 0.5)
            score_b = 1.0 - score_a

            W[i][j] += score_a
            W[j][i] += score_b
            N[i][j] += 1
            N[j][i] += 1

        # MM Algorithm: iterative update until convergence
        # p[i] = strength of model i (initialized equally)
        p = [1.0] * n

        for iteration in range(max_iter):
            p_old = p[:]

            for i in range(n):
                numerator = sum(W[i][j] for j in range(n) if j != i)
                denominator = sum(
                    N[i][j] / (p[i] + p[j])
                    for j in range(n)
                    if j != i and N[i][j] > 0
                )
                if denominator > 0:
                    p[i] = numerator / denominator

            # Normalize so product of strengths = 1 (identifiability)
            geo_mean = math.exp(sum(math.log(max(pi, 1e-10)) for pi in p) / n)
            p = [pi / geo_mean for pi in p]

            # Check convergence
            max_change = max(abs(p[i] - p_old[i]) for i in range(n))
            if max_change < tol:
                break

        # Convert strength → Elo-style rating
        # rating_i = initial_rating + scale * log10(p_i)
        bt_elo = {}
        for m_id, i in idx.items():
            bt_elo[m_id] = round(
                self.initial_rating + self.scale * math.log10(max(p[i], 1e-10)), 2
            )

        return bt_elo

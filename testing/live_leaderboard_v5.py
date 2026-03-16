import json
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humorrank.tournament.bradley_terry import BradleyTerry
from humorrank.tournament.elo import EloRating

# ── Short display names for heatmap columns ──────────────────────────────────
SHORT_NAMES = {
    "GPT-5":                  "GPT-5",
    "Kimi-K2":                "KimiK2",
    "Gemini-2.5-Pro":         "Gem2.5",
    "HumorGen SFT-7B":        "SFT-7B",
    "HumorGen DPO-7B":        "DPO-7B",
    "HumorGen GRPO-7B":       "GRPO-7B",
    "HumorGen SFT-Think-7B":  "SFT-Thk",
    "GPT-OSS-120B":           "GPT-OSS",
    "Qwen3-32B":              "Qw3-32B",
    "HumorGen DPO-Think-7B":  "DPO-Thk",
    "HumorGen GRPO-Think-7B": "GRP-Thk",
    "Base Qwen-7B":           "Base-7B",
    "HumorGen-Com-7B":        "Com-7B",
}

# ── BT leaderboard plot: Y-axis labels (HG = HumorGen, -T = Think/reasoning traces) ──
BT_PLOT_NAMES = {
    "HumorGen SFT-7B":        "HG-SFT",
    "HumorGen DPO-7B":        "HG-DPO",
    "HumorGen GRPO-7B":       "HG-GRPO",
    "HumorGen SFT-Think-7B":  "HG-SFT-T",
    "HumorGen DPO-Think-7B":  "HG-DPO-T",
    "HumorGen GRPO-Think-7B": "HG-GRPO-T",
    "HumorGen-Com-7B":        "HG-Com",
    "Base Qwen-7B":           "Base-7B",
    "GPT-5":                  "GPT-5",
    "Kimi-K2":                "Kimi-K2",
    "Gemini-2.5-Pro":         "Gemini-2.5-Pro",
    "GPT-OSS-120B":           "GPT-OSS-120B",
    "Qwen3-32B":              "Qwen3-32B",
}

# ── ANSI colour helpers ───────────────────────────────────────────────────────
def _rgb(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def _bg(r, g, b, text):
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"

BOLD  = lambda t: f"\033[1m{t}\033[0m"
DIM   = lambda t: f"\033[2m{t}\033[0m"
RESET = "\033[0m"

def win_colour(pct):
    """Return a background-coloured cell for a win-rate value (0-100)."""
    if pct is None:                      # diagonal
        return _bg(40, 40, 40, "  --  ")
    if pct >= 70:
        r, g, b = 0, 180, 80            # strong green
    elif pct >= 55:
        r, g, b = 100, 200, 100         # light green
    elif pct >= 45:
        r, g, b = 200, 200, 60         # yellow (close)
    elif pct >= 30:
        r, g, b = 220, 120, 60         # orange
    else:
        r, g, b = 200, 50, 50          # red
    return _bg(r, g, b, f"{pct:5.1f}%")

def rank_colour(rank, total):
    """Colour for rank badge."""
    if rank == 1:   return _rgb(255, 215,   0, f"#{rank}")   # gold
    if rank == 2:   return _rgb(192, 192, 192, f"#{rank}")   # silver
    if rank == 3:   return _rgb(205, 127,  50, f"#{rank}")   # bronze
    if rank > total - 2:
        return _rgb(180,  60,  60, f"#{rank}")               # red for bottom
    return _rgb(150, 150, 255, f"#{rank}")                   # default blue


def save_plots(ranked, head2head, player_list, output_dir):
    """Save visualization plots: BT Ranking Bar Chart and Win-Rate Heatmap."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Configure matplotlib for paper-ready styling
    plt.rcParams.update({'font.size': 14})
    
    # 1. BT Ranking Bar Chart (x-axis 400–1200; HG = HumorGen, -T = Think)
    plt.figure(figsize=(12, 8))
    raw_names = [r["name"] for r in reversed(ranked)]
    names = [BT_PLOT_NAMES.get(n, n) for n in raw_names]
    ratings = [r["bt"] for r in reversed(ranked)]
    
    # Assign colors based on name mapping logic
    colors = []
    for name in raw_names:
        if name == "HumorGen SFT-7B":
            colors.append("#00DC78")  # Green
        elif name in ("HumorGen DPO-7B", "HumorGen GRPO-7B"):
            colors.append("#64B4FF")  # Blue
        elif "Think" in name:
            colors.append("#A0A0A0")  # Grey
        elif name == "Base Qwen-7B":
            colors.append("#785050")  # Dark Red
        elif name == "HumorGen-Com-7B":
            colors.append("#FF8000")  # Orange
        else:
            colors.append("#FFD700")  # Gold

    bars = plt.barh(names, ratings, color=colors)
    plt.xlim(500, 1200)
    plt.xlabel("Bradley-Terry Rating", fontsize=16, fontweight='bold')
    plt.title("Model Performance (BT Rating)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add value labels inside bars (stay within plot)
    for bar, rt in zip(bars, ratings):
        x_pos = min(rt, 1180) - 30
        plt.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{rt:.1f}',
                 va='center', ha='right', fontsize=11, fontweight='bold', color='#1a1a1a')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bt_leaderboard_v5_full.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Pairwise Win-Rate Heatmap
    matrix = []
    for p1 in player_list:
        row = []
        for p2 in player_list:
            if p1 == p2:
                row.append(np.nan)
            else:
                h = head2head[p1][p2]
                if h["total"] == 0:
                    row.append(np.nan)
                else:
                    row.append(h["wins"] / h["total"] * 100)
        matrix.append(row)
        
    df = pd.DataFrame(matrix, index=[SHORT_NAMES.get(p, p) for p in player_list], 
                      columns=[SHORT_NAMES.get(p, p) for p in player_list])
    
    plt.figure(figsize=(12, 10))
    # Use subdued colormap for readability
    ax = sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", center=50, 
                cbar_kws={'label': 'Win Rate (%)'}, square=True, annot_kws={"size": 11})
    
    ax.figure.axes[-1].yaxis.label.set_size(14)
    ax.figure.axes[-1].yaxis.label.set_weight('bold')
    
    plt.xlabel("Opponent", fontsize=16, fontweight='bold')
    plt.ylabel("Model", fontsize=16, fontweight='bold')
    plt.title("Pairwise Win-Rate Heatmap (Row beats Column %)")
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "winrate_heatmap_v5_full.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Save Leaderboard Data to CSV & MD
    df_ranked = pd.DataFrame(ranked)
    
    # Clean up formatting for output tables
    df_out = df_ranked[['name', 'bt', 'ci_low', 'ci_high', 'elo', 'wins', 'losses', 'ties', 'win_rate']].copy()
    df_out['Rank'] = range(1, len(df_out) + 1)
    df_out['95% CI'] = df_out.apply(lambda x: f"[{x['ci_low']:.1f}, {x['ci_high']:.1f}]", axis=1)
    df_out['bt'] = df_out['bt'].round(2)
    df_out['elo'] = df_out['elo'].round(1)
    df_out['win_rate'] = df_out['win_rate'].round(1).astype(str) + "%"
    
    df_out = df_out[['Rank', 'name', 'bt', '95% CI', 'elo', 'wins', 'losses', 'ties', 'win_rate']]
    df_out.columns = ['Rank', 'Model', 'BT Rating', '95% CI', 'Stable ELO', 'Wins', 'Losses', 'Ties', 'Win%']
    
    df_out.to_csv(os.path.join(output_dir, "leaderboard_v5_full.csv"), index=False)
    
    with open(os.path.join(output_dir, "leaderboard_v5_full.md"), "w") as f:
        f.write("# HumorRank Leaderboard\n\n")
        f.write(df_out.to_markdown(index=False))
        
    # Reset rcParams just in case
    plt.rcParams.update({'font.size': 10})
    print(f"Paper-ready plots and tables saved to {output_dir}")


def compute_live_leaderboard():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    match_file = os.path.join(_root, "testing", "match_history_v5_13_models_50.jsonl")
    plots_dir = os.path.join(_root, "plots")
    if not os.path.exists(match_file):
        print("Match file not found.")
        return

    matches   = []
    player_ids = set()
    stats      = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "games": 0})
    head2head  = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))

    with open(match_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                m = json.loads(line)
                if m.get("confidence") == "ERROR": continue

                a, b  = m["player_a"], m["player_b"]
                player_ids.add(a)
                player_ids.add(b)

                score_a = 0.5 if m.get("is_tie") else (1.0 if m.get("winner") == a else 0.0)
                matches.append({"player_a": a, "player_b": b, "score_a": score_a})

                # head-to-head tally
                head2head[a][b]["total"] += 1
                head2head[b][a]["total"] += 1
                if score_a == 1.0:
                    head2head[a][b]["wins"] += 1
                    stats[a]["wins"]   += 1
                    stats[b]["losses"] += 1
                elif score_a == 0.0:
                    head2head[b][a]["wins"] += 1
                    stats[b]["wins"]   += 1
                    stats[a]["losses"] += 1
                else:
                    stats[a]["ties"] += 1
                    stats[b]["ties"] += 1

                stats[a]["games"] += 1
                stats[b]["games"] += 1
            except:
                continue

    if not matches:
        print("No valid matches found yet.")
        return

    player_list = sorted(list(player_ids))
    total       = len(player_list)

    # ── Bradley-Terry & Stable ELO ───────────────────────────────────────────
    bt  = BradleyTerry()
    bt_results = bt.fit_with_ci(matches, player_list, num_bootstrap=100)

    elo = EloRating(k_factor=16)
    stable_elo = elo.compute_stable(matches, player_list, num_shuffles=5)

    # ── Sort by BT ───────────────────────────────────────────────────────────
    ranked = []
    for name in player_list:
        s  = stats[name]
        wr = s["wins"] / s["games"] * 100 if s["games"] else 0
        ranked.append({
            "name":     name,
            "short":    SHORT_NAMES.get(name, name[:8]),
            "bt":       bt_results[name]["rating"],
            "ci_low":   bt_results[name]["ci_low"],
            "ci_high":  bt_results[name]["ci_high"],
            "elo":      stable_elo[name],
            "wins":     s["wins"],
            "losses":   s["losses"],
            "ties":     s["ties"],
            "games":    s["games"],
            "win_rate": wr,
        })

    ranked.sort(key=lambda x: x["bt"], reverse=True)
    rank_order = [r["name"] for r in ranked]   # ordered list for heatmap

    # ── Progress stats ───────────────────────────────────────────────────────
    n_models       = total
    total_possible = n_models * (n_models - 1) // 2 * 50   # 50 headlines
    completed      = len(matches)
    pct_done       = completed / total_possible * 100 if total_possible else 0
    headlines_done = completed // (n_models * (n_models - 1) // 2)

    # ════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — LEADERBOARD
    # ════════════════════════════════════════════════════════════════════════
    W = 102
    print()
    print(_rgb(255, 200, 0,  "═" * W))
    print(_rgb(255, 200, 0, f"{'  LIVE HUMORRANK V5 LEADERBOARD':^{W}}"))
    print(_rgb(255, 200, 0,  "═" * W))

    hdr = (f"{'Rank':<5} {'Model':<22} {'Scale':<16} "
           f"{'BT Rating':<12} {'95% CI':<20} {'Stable ELO':<12} "
           f"{'W':>5} {'L':>5} {'T':>4} {'Win%':>7}")
    print(BOLD(hdr))
    print(_rgb(100, 100, 100, "─" * W))

    # Scale labels
    SCALE = {
        "GPT-5":                  "Frontier ~1.5T",
        "Kimi-K2":                "Frontier",
        "Gemini-2.5-Pro":         "Frontier",
        "HumorGen SFT-7B":        "7B (SFT)",
        "HumorGen DPO-7B":        "7B (DPO)",
        "HumorGen GRPO-7B":       "7B (GRPO)",
        "HumorGen SFT-Think-7B":  "7B (SFT+Think)",
        "GPT-OSS-120B":           "120B",
        "Qwen3-32B":              "32B",
        "HumorGen DPO-Think-7B":  "7B (DPO+Think)",
        "HumorGen GRPO-Think-7B": "7B (GRPO+Think)",
        "Base Qwen-7B":           "7B (Base)",
        "HumorGen-Com-7B":        "7B (Comdian SFT)",
    }

    for i, r in enumerate(ranked):
        rank_badge = rank_colour(i + 1, total)
        scale_str  = SCALE.get(r["name"], "—")

        # Highlight HumorGen models in leaderboard
        name_str = r["name"]
        if name_str == "HumorGen SFT-7B":
            name_display = _rgb(0, 220, 120, f"  {name_str:<20}")
        elif name_str in ("HumorGen DPO-7B", "HumorGen GRPO-7B"):
            name_display = _rgb(100, 180, 255, f"  {name_str:<20}")
        elif "Think" in name_str:
            name_display = _rgb(160, 160, 160, f"  {name_str:<20}")
        elif name_str == "HumorGen-Com-7B":
            name_display = _rgb(255, 128, 0,   f"  {name_str:<20}")
        elif name_str == "Base Qwen-7B":
            name_display = _rgb(120, 80, 80,   f"  {name_str:<20}")
        else:
            name_display = _rgb(255, 215,  0,  f"  {name_str:<20}")  # frontier = gold

        ci_str = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        print(f"{rank_badge:<5} {name_display} {scale_str:<16} "
              f"{r['bt']:<12.2f} {ci_str:<20} {r['elo']:<12.1f} "
              f"{r['wins']:>5} {r['losses']:>5} {r['ties']:>4} {r['win_rate']:>6.1f}%")

    print(_rgb(100, 100, 100, "─" * W))
    bar_filled = int(pct_done / 2)
    bar = "█" * bar_filled + "░" * (50 - bar_filled)
    print(f"  Progress: [{_rgb(0, 200, 80, bar)}] "
          f"{_rgb(255,255,255, f'{pct_done:.1f}%')}  "
          f"({completed:,} / {total_possible:,} matches  |  ~{headlines_done} / 50 headlines done)")
    print()

    # ════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — PAIRWISE WIN-RATE HEATMAP  (row beats col %)
    # ════════════════════════════════════════════════════════════════════════
    COL_W = 8   # cell width

    print(_rgb(180, 120, 255, "═" * W))
    print(_rgb(180, 120, 255, f"{'  🔥  PAIRWISE WIN-RATE HEATMAP  (row → beats → col)  🔥':^{W}}"))
    print(_rgb(180, 120, 255, "═" * W))
    print(DIM("  Green = row model dominates | Red = col model dominates | Yellow = close match"))
    print()

    # Column headers (short names in rank order)
    short_order = [SHORT_NAMES.get(p, p[:7]) for p in rank_order]
    header_row  = f"{'':>10}"
    for sh in short_order:
        header_row += f"{sh:>{COL_W}}"
    print(BOLD(header_row))
    print(_rgb(80, 80, 80, "  " + "─" * (8 + COL_W * total)))

    for row_name in rank_order:
        row_short = SHORT_NAMES.get(row_name, row_name[:9])
        line = f"{row_short:>10}"
        for col_name in rank_order:
            if row_name == col_name:
                line += f"{'  --  ':>{COL_W}}"
            else:
                h = head2head[row_name][col_name]
                if h["total"] == 0:
                    cell = _rgb(80, 80, 80, f"{'  N/A':>{COL_W}}")
                else:
                    pct  = h["wins"] / h["total"] * 100
                    cell = win_colour(pct)
                    # right-pad to col width
                    cell = f"{cell:>{COL_W - 6}}"   # 6 chars = colour escape visible width
                line += "  " + cell
        print(line)

    print()
    print(_rgb(80, 80, 80, "  Legend:  "), end="")
    for label, pct in [("≥70% (dominant)", 75), ("55–70% (ahead)", 62),
                        ("45–55% (close)", 50), ("30–45% (behind)", 37), ("<30% (weak)", 20)]:
        print(win_colour(pct) + f" {label}  ", end="")
    print()
    print()
    print(_rgb(255, 200, 0, "═" * W))
    print(DIM(f"  Green = HumorGen SFT-7B   "
              f"Gold = frontier models   Grey = Think variants"))
    print(_rgb(255, 200, 0, "═" * W))
    print()

    # Save plots
    try:
        save_plots(ranked, head2head, rank_order, plots_dir)
    except Exception as e:
        print(f"Error saving plots: {e}")


if __name__ == "__main__":
    compute_live_leaderboard()

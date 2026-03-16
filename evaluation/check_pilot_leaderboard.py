import json
import pandas as pd
from collections import defaultdict
import glob

# Load all pilot data so far
files = glob.glob("newdata/results_pilot/*.jsonl")
data = []
for f in files:
    if "errors" in f: continue
    with open(f) as infile:
        for line in infile:
            try:
                data.append(json.loads(line))
            except: pass

if not data:
    print("No data found yet.")
    exit()

stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "matches": 0})

for r in data:
    m_a = r['model_a']
    m_b = r['model_b']
    winner = r['winner_model']
    
    # Update matches
    stats[m_a]['matches'] += 1
    stats[m_b]['matches'] += 1
    
    if winner == "TIE":
        stats[m_a]['ties'] += 1
        stats[m_b]['ties'] += 1
    elif winner == m_a:
        stats[m_a]['wins'] += 1
        stats[m_b]['losses'] += 1
    elif winner == m_b:
        stats[m_b]['wins'] += 1
        stats[m_a]['losses'] += 1

# Convert to DataFrame
df_data = []
for model, s in stats.items():
    win_rate = (s['wins'] / s['matches']) * 100 if s['matches'] > 0 else 0
    df_data.append({
        "Model": model,
        "Win Rate": f"{win_rate:.1f}%",
        "Wins": s['wins'],
        "Losses": s['losses'],
        "Ties": s['ties'],
        "Matches": s['matches']
    })

df = pd.DataFrame(df_data).sort_values("Wins", ascending=False)
print("\n=== PILOT LEADERBOARD (Available Data) ===")
print(df.to_string(index=False))
print(f"\nTotal Comparisons Processed: {len(data)}")

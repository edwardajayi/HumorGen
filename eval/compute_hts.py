"""
HTB Evaluation: Compute Humor Transfer Score (HTS) and Transfer Gap (TG)
=========================================================================
Formal definitions (from DESIGN_PLAN.md §6.3):

  HTS(M) = (1/|D_OOD|) * sum_{d in D_OOD} P_pref(M, d)
  TG(M)  = HTS(M) - P_pref(M, H)

  Where:
    D_OOD = {A, B, C, D, E, F}  (excludes G=near-distribution, H=in-domain)
    P_pref(M, d) = proportion of comparisons in domain d won by model M (majority vote)

Input: annotations CSV with columns:
  htb_id, domain, model_A_output, model_B_output, annotator_1, annotator_2, annotator_3
  (annotator values: "A", "B", or "tie")

Usage:
  python compute_hts.py --annotations annotations.csv --model CSF-SFT --baseline Baseline-SFT
"""

import argparse
import csv
from collections import defaultdict
import numpy as np


OOD_DOMAINS      = {"A", "B", "C", "D", "E", "F", "H"} # All OOD domains including BBC Headlines
NEAR_DIST_DOMAIN = {"G"}   # Diagnostic trap: near-distribution (directive format)



def majority_vote(votes: list[str]) -> str:
    """Return majority vote among annotators. Tie if no majority."""
    counts = {"A": 0, "B": 0, "tie": 0}
    for v in votes:
        counts[v.strip()] += 1
    if counts["A"] > counts["B"] and counts["A"] > counts["tie"]:
        return "A"
    if counts["B"] > counts["A"] and counts["B"] > counts["tie"]:
        return "B"
    return "tie"


def compute_pref(domain_results: dict, domain_set: set, model_label: str) -> float:
    """
    Compute P_pref(M, d) averaged over a set of domains.
    model_label: "A" or "B" — which column corresponds to the focal model.
    """
    pref_rates = []
    for domain in domain_set:
        if domain not in domain_results:
            continue
        results = domain_results[domain]
        wins  = sum(1 for r in results if r == model_label)
        total = sum(1 for r in results if r != "tie")  # exclude ties from denominator
        pref_rates.append(wins / total if total > 0 else 0.0)
    return float(np.mean(pref_rates)) if pref_rates else 0.0


def compute_fleiss_kappa(annotations: list[list[str]]) -> float:
    """Compute Fleiss' kappa for inter-annotator agreement."""
    categories = ["A", "B", "tie"]
    n_items = len(annotations)
    n_raters = len(annotations[0]) if annotations else 0
    if n_items == 0 or n_raters == 0:
        return 0.0

    # P_i: proportion of agreement per item
    P_i = []
    for item in annotations:
        counts = {c: item.count(c) for c in categories}
        total = sum(counts.values())
        P_i.append(sum(v * (v - 1) for v in counts.values()) / (total * (total - 1)) if total > 1 else 0.0)

    P_bar = np.mean(P_i)

    # p_j: proportion of each category across all annotations
    all_annotations = [v for item in annotations for v in item]
    N = len(all_annotations)
    p_j = {c: all_annotations.count(c) / N for c in categories}

    P_e = sum(p ** 2 for p in p_j.values())

    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0.0
    return round(kappa, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to annotations CSV")
    parser.add_argument("--model",    default="CSF-SFT",       help="Label for focal model (column model_A or model_B)")
    parser.add_argument("--baseline", default="Baseline-SFT",  help="Label for baseline model")
    parser.add_argument("--focal-column", default="A", choices=["A", "B"],
                        help="Which annotation column (A or B) corresponds to the focal model")
    args = parser.parse_args()

    domain_results  = defaultdict(list)  # domain -> list of majority votes
    all_annotations = []

    with open(args.annotations, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["domain"].strip().upper()
            votes = [row["annotator_1"], row["annotator_2"], row["annotator_3"]]
            all_annotations.append(votes)
            mv = majority_vote(votes)
            domain_results[domain].append(mv)

    # --- Inter-Annotator Agreement ---
    kappa = compute_fleiss_kappa(all_annotations)
    print(f"\n{'='*55}")
    print(f"  Humor Transfer Bench — Evaluation Report")
    print(f"  Focal model : {args.model}")
    print(f"  Baseline    : {args.baseline}")
    print(f"{'='*55}")
    print(f"\n[IAA] Fleiss' κ = {kappa:.4f}", end="")
    if kappa >= 0.40:
        print("  ✓ (meets minimum threshold of 0.40)")
    else:
        print("  ✗ WARNING: Below minimum threshold of 0.40. Results may not be reliable.")

    # --- Per-domain preference rates ---
    print(f"\n[Per-Domain P_pref({args.model})]")
    for domain in sorted(domain_results.keys()):
        results = domain_results[domain]
        wins  = sum(1 for r in results if r == args.focal_column)
        total = sum(1 for r in results if r != "tie")
        pref  = wins / total if total > 0 else 0.0
        tag   = "(OOD)" if domain in OOD_DOMAINS else ("(near-dist)" if domain in NEAR_DIST_DOMAIN else "(in-domain)")
        print(f"  Domain {domain} {tag}: {pref:.3f}  ({wins}/{total})")

    # --- HTS & TBI ---
    hts        = compute_pref(domain_results, OOD_DOMAINS,      args.focal_column)
    p_neardist = compute_pref(domain_results, NEAR_DIST_DOMAIN, args.focal_column)

    tbi = p_neardist - hts               # Template-Binding Index

    print(f"\n[Summary Metrics]")
    print(f"  HTS ({args.model})      = {hts:.4f}   (avg over OOD domains A-F, H)")
    print(f"  P_pref (Domain G) = {p_neardist:.4f}   (near-distribution / directive format)")
    print(f"  Template-Binding  = {tbi:+.4f}  (P_pref_G - HTS)")
    print()

    # --- TBI interpretation ---
    if tbi > 0.10:
        print(f"  ⚠ TBI high ({tbi:+.4f}): {args.model} is TEMPLATE-BOUND.")
        print(f"    Performs well on directive prompts (Domain G) but not on OOD distributions.")
        print(f"    This suggests instruction-following, not transferable humor cognition.")
    elif abs(tbi) <= 0.10:
        print(f"  ✓ TBI near-zero ({tbi:+.4f}): {args.model} is FORMAT-AGNOSTIC.")
        print(f"    Humor quality is consistent across directive and non-directive prompt structures.")
        print(f"    Evidence of internalized humor cognition — supports CSF claim.")
    else:
        print(f"  ~ TBI negative ({tbi:+.4f}): {args.model} is FORMAT-AVERSE.")
        print(f"    Performs better on non-directive prompts than explicit instructions.")

    print(f"{'='*55}\n")



if __name__ == "__main__":
    main()

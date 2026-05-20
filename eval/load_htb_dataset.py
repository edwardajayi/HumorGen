"""Load Humor Transfer Bench (HTB) prompts from data/datasets/htb/htb_dataset.tsv."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "data" / "datasets" / "htb" / "htb_dataset.tsv"
SYSTEM_INSTRUCTION = "Generate a funny joke given this prompt."


def load_htb_dataset(path: Path | str = DEFAULT_PATH, domain: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if domain and row["domain"].strip() != domain.strip():
                continue
            rows.append(
                {
                    "id": row["id"].strip(),
                    "domain": row["domain"].strip(),
                    "prompt": row["prompt"].strip(),
                    "system_instruction": SYSTEM_INSTRUCTION,
                }
            )
    return rows


if __name__ == "__main__":
    data = load_htb_dataset()
    print(f"Loaded {len(data)} prompts")
    counts: dict[str, int] = {}
    for row in data:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")

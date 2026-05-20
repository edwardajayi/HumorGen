# Humor Transfer Bench (HTB) — Data Card

**Version:** v1  
**File:** `data/datasets/htb/htb_dataset.tsv`  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see Licensing below)

## Overview

Humor Transfer Bench (HTB) is a benchmark of **400 English prompts** for evaluating **textual humor generation**. Each prompt is an input to a model; the model must generate a funny joke in response.

HTB holds the task constant while varying the input style, so models can be compared across diverse prompt categories without requiring headline-only training data.

## File Format

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique prompt ID (`HTB_{A-H}_{NNN}`, e.g. `HTB_A_001`) |
| `domain` | string | Prompt category name |
| `prompt` | string | Raw input text passed to the model |

- **Format:** TSV, UTF-8, Unix line endings  
- **Size:** 400 rows (8 domains × 50 prompts)

## Domains

| Code | Domain | Count | Description |
|------|--------|------:|-------------|
| A | Fun Facts | 50 | Surprising declarative facts |
| B | Daily Life | 50 | First-person observational statements |
| C | Social Terms | 50 | Abstract or compound noun-phrase prompts |
| D | Object Voices | 50 | Attributed utterances (dialogic prompts) |
| E | Fantasy Creatures | 50 | Short scenario / situational prompts |
| F | Twisted Definitions | 50 | Analogical or definitional prompts |
| G | Direct Prompts | 50 | Instruction-style prompts |
| H | News Headlines | 50 | Real-world news headlines |

## Task Instruction

Use the same generation instruction for all prompts:

> Generate a funny joke given this prompt.

Example loader:

```bash
python eval/load_htb_dataset.py
```

## Data Sources

| Domain | Source |
|--------|--------|
| A–G | HTB-authored prompts (created for this benchmark) |
| H | 50 real-world BBC news headlines |

## Licensing

### HTB-authored prompts (Domains A–G)

Released under **Creative Commons Attribution 4.0 (CC BY 4.0)**.

You may share and adapt these prompts with attribution to **Humor Transfer Bench (HTB)**.

### News Headlines (Domain H)

Domain H contains publicly available news headline text collected from BBC. Use is subject to the original publication context; we provide headline text for research evaluation only.

## Usage Notes

- HTB is intended for **evaluation**, not training.
- Prompts are inputs only — they are **not** jokes and contain no model outputs.
- Canonical joke templates (e.g. knock-knock, walks-into-a-bar) were excluded by design.

## Related Files in This Repository

| Path | Description |
|------|-------------|
| `data/datasets/htb/htb_dataset.tsv` | Full benchmark prompts |
| `eval/load_htb_dataset.py` | Python loader |
| `results/leaderboards/htb_full_*` | Reported HTB leaderboard CSV/MD files |
| `eval/compute_hts.py` | Domain-level metrics from human annotation CSV |

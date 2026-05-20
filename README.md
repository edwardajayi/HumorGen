# HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation

## Abstract

Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective—predicting the most likely next word—inherently conflicts with the surprise and incongruity needed for comedy. To bridge this gap, we introduce the **Cognitive Synergy Framework**, a methodology for generating high-quality humor data inspired by psychological theories of humor. Utilizing a Mixture-of-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework produces a theory-grounded dataset, which we use to fine-tune a 7B-parameter student model. We further evaluate Direct Preference Optimization (DPO) and Offline Group Relative Policy Optimization (O-GRPO), finding that neither improves over a well-curated SFT baseline. Our 7B HumorGen variants achieve strong open-weight performance on HTB and SemEval humor-generation benchmarks. Code and data will be released upon publication.

## Humor Transfer Bench (HTB)

We release **Humor Transfer Bench (HTB)**, a 400-prompt benchmark for evaluating textual humor generation across eight input categories.

- **Dataset folder:** [`data/datasets/htb/`](data/datasets/htb/)
- **Dataset file:** [`data/datasets/htb/htb_dataset.tsv`](data/datasets/htb/htb_dataset.tsv)
- **Data card & licensing:** [`data/datasets/htb/README.md`](data/datasets/htb/README.md)
- **Loader:** [`eval/load_htb_dataset.py`](eval/load_htb_dataset.py)
- **Reported leaderboard tables:** [`results/leaderboards/`](results/leaderboards/)

HTB-authored prompts (Domains A–G) are released under **CC BY 4.0**. See the data card for full licensing details, including Domain H (news headlines).

## Setup

```bash
git clone <repo-url>
cd HumorGen
pip install -r requirements.txt
```

Python 3.10+. API-based scripts require keys in environment variables (e.g. `GROQ_API_KEY`, `OPENAI_API_KEY`); do not commit `.env` files.

## Repository Layout

- **Training:** `training/`
- **Generation:** `testing/`
- **Evaluation:** `evaluation/`, `eval/`
- **HTB benchmark:** `data/datasets/htb/`
- **Reported results:** `results/leaderboards/`

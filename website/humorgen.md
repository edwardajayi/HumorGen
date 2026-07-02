---
language:
  - en
  - fr
  - es
license: apache-2.0
tags:
  - humor
  - computational-humor
  - cognitive-synergy-framework
  - collection
  - text-generation
---

**An Open-Weight Ecosystem for Computational Humor Generation**

[View on Hugging Face](https://huggingface.co/collections/Jayi2424/humorgen) · [Landing repo](https://huggingface.co/Jayi2424/HumorGen) · [Read the Paper](https://arxiv.org/abs/2604.09629) · [CLEF 2026 Paper](https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf)

---

## The Problem

Language models struggle to be funny — not because they lack knowledge of humor, but because of how they are trained. Standard next-token prediction rewards the most probable continuation. In humor, the most probable output is always the safest, most generic one. Comedy lives at the edges of the distribution.

Training a model to "be funny" with a single instruction does not work. It produces outputs that scan as jokes but fail to land.

---

## Our Approach: The Cognitive Synergy Framework

HumorGen introduces the **Cognitive Synergy Framework (CSF)** — a Mixture-of-Thought method that structures humor generation as an ensemble of six distinct cognitive personas, each grounded in psychological theory.

Rather than a single generation path, CSF produces **24 candidates per headline** (4 per persona × 6 personas) from a teacher ensemble of **Kimi-K2** and **Qwen2.5-32B-Instruct**. A **HumorRank** pairwise tournament (Llama 3.3-70B judge, Bradley–Terry aggregation) ranks all candidates; top jokes fine-tune a 7B student.

| Persona | Humor Theory | Mechanism | Cognitive Focus |
|:---|:---|:---|:---|
| The Neurotic | Relief Theory | Tension Release | Internal anxiety, overthinking, social insecurity |
| The Cynic | Superiority Theory | Social Critique | Hypocrisy, biting sarcasm, moral contradictions |
| The Observer | Incongruity | Social Mapping | Mundane minutiae and unwritten awkward social norms |
| The Wordsmith | Linguistic | Ambiguity | Puns, double entendres, phonological play |
| The Optimist | Benign Violation | Recontextualization | Wholesome misinterpretations of negative traits |
| The Absurdist | Incongruity | Surrealism | Non-sequiturs, dream logic, fractured causality |

**Training pipeline:**

1. **Teacher generation** — Kimi-K2 + Qwen2.5-32B-Instruct generate 24 candidates per headline on SemEval-2026 MWAHAHA (1,200 prompts, ~28,800 candidates).
2. **Ranking & SFT** — Llama 3.3-70B judges pairwise; HumorRank Bradley–Terry yields Elo ratings. Top jokes fine-tune Qwen2.5-7B-Instruct (LoRA r=16, Unsloth). **CSD** (Cognitive Synergy Distillation) variants include persona reasoning traces in Think models.
3. **Alignment ablation** — **DPO** (β = 0.1, pairwise top-5 vs bottom-5) and **O-GRPO** (G = 24, offline group-relative). **Result:** DPO matches SFT; O-GRPO trails both. Neither improves over a well-curated SFT baseline — a **data quality ceiling**.

---

## Key Finding

Targeted **CSF data curation matters more than model scale** for humor generation. HumorGen-SFT-7B and HumorGen-DPO-7B rank among the strongest open-weight models on HTB and SemEval MWAHAHA, outperforming models 4–18× larger and competitive with frontier systems. When SFT data is diverse and well-curated, preference optimization yields no significant gains. Reasoning-augmented (Think/CSD) variants can reduce judged funniness — the “explainer trap.”

---

## Evaluation Leaderboards

Pairwise Bradley–Terry ratings (Llama 3.3-70B judge). HumorGen rows highlighted.

### Humor Transfer Bench (HTB)

400 prompts · 8 domains · 42,000 comparisons

| Rank | Model | BT rating | 95% CI |
|:---:|:---|---:|:---|
| 1 | GPT-5 | 1336.18 | 1323.3 – 1348.3 |
| 2 | Kimi-K2 | 1259.98 | 1249.7 – 1268.5 |
| **3** | **HumorGen SFT-7B** | **1128.14** | 1118.3 – 1138.1 |
| **4** | **HumorGen DPO-7B** | **1123.72** | 1115.7 – 1134.9 |
| 5 | HumorGen DPO-Think-7B | 1116.65 | 1107.9 – 1127.1 |
| 6 | HumorGen SFT-Think-7B | 1085.31 | 1075.8 – 1096.5 |
| 7 | HumorGen GRPO-7B | 1071.13 | 1060.8 – 1080.1 |
| 8 | Gemini-2.5-Pro | 1059.07 | 1049.3 – 1068.4 |
| 9 | HumorGen GRPO-Think-7B | 1055.94 | 1043.8 – 1066.8 |
| 10 | GPT-OSS-120B | 1048.19 | 1039.7 – 1057.1 |
| 11 | Qwen3-32B | 990.44 | 981.4 – 999.4 |
| 12 | phi2-Humor | 803.72 | 794.5 – 818.2 |
| 13 | HumorGen-Com-7B | 665.93 | 645.5 – 680.0 |
| 14 | Base Qwen-7B | 643.01 | 628.3 – 658.0 |
| 15 | JokeGPT | 612.58 | 597.6 – 627.4 |

### SemEval-2026 MWAHAHA

50 headlines · 5,250 comparisons

| Rank | Model | BT rating | 95% CI |
|:---:|:---|---:|:---|
| 1 | GPT-5 | 1378.73 | 1346.1 – 1421.5 |
| 2 | Kimi-K2 | 1279.63 | 1245.0 – 1322.1 |
| 3 | Gemini-2.5-Pro | 1247.80 | 1212.4 – 1279.7 |
| **4** | **HumorGen SFT-7B** | **1140.37** | 1107.9 – 1173.2 |
| **5** | **HumorGen DPO-7B** | **1135.25** | 1101.9 – 1160.2 |
| 6 | HumorGen GRPO-7B | 1089.84 | 1060.7 – 1114.1 |
| 7 | GPT-OSS-120B | 1049.99 | 1019.8 – 1081.5 |
| 8 | HumorGen SFT-Think-7B | 1049.99 | 1016.3 – 1084.5 |
| 9 | HumorGen DPO-Think-7B | 1031.30 | 1002.1 – 1058.0 |
| 10 | Qwen3-32B | 1023.18 | 997.0 – 1046.0 |
| 11 | HumorGen GRPO-Think-7B | 948.51 | 914.7 – 982.6 |
| 12 | phi2-Humor | 791.32 | 751.8 – 826.1 |
| 13 | HumorGen-Com-7B | 721.97 | 682.3 – 750.8 |
| 14 | Base Qwen-7B | 673.16 | 643.1 – 718.0 |
| 15 | JokeGPT | 438.97 | 384.3 – 500.2 |

---

## Extending to Constrained Humor: CLEF 2026 JOKER Task 4

The CSF generalizes to constrained pun generation. **CLEF 2026 JOKER Task 4** provides a structured **pun brief** (pun word + Sense A + Sense B) and asks for a short humorous text activating both senses in English, French, and Spanish.

**Four-stage cross-lingual LoRA curriculum** (QLoRA 4-bit, Qwen3-14B/32B):

1. **Stage 1** — ~12k CSF-curated English MWAHAHA headlines → HumorGen-SFT-14B/32B
2. **Stage 2a** — English JOKER fine-tuning (3,985 monolingual CSF examples)
3. **Stage 2b** — FR/ES shared multilingual warm-up (11,038 EN+FR+ES examples)
4. **Stage 3** — Per-language specialization (FR: 3,952 rows; ES: 3,101 rows)

**JOKER finding:** CSF guidance at **test time** dominates. Kimi-CSF leads all three language tracks in interim Bradley–Terry evaluation. Distilled student models (HumorGen-JOKER-14B/32B) transfer structural pun-brief competence but not the multi-candidate search breadth of the full CSF pipeline at inference.

---

## Model collections

14 open-weight LoRA adapters on Hugging Face. Collection: [Jayi2424/HumorGen](https://huggingface.co/collections/Jayi2424/humorgen). Apache-2.0.

### Core HumorGen — 7B (6 models)

| Model | Training | CSD | Hugging Face |
|:---|:---|:---:|:---|
| HumorGen_SFT_7B | Supervised Fine-Tuning | — | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_7B) |
| HumorGen_SFT_Think_7B | SFT + CSD traces | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_Think_7B) |
| HumorGen_DPO_7B | DPO (β=0.1, from SFT) | — | [Link](https://huggingface.co/Jayi2424/HumorGen_DPO_7B) |
| HumorGen_DPO_Think_7B | DPO (from SFT-Think) | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_DPO_Think_7B) |
| HumorGen_GRPO_7B | O-GRPO (G=24, from SFT) | — | [Link](https://huggingface.co/Jayi2424/HumorGen_GRPO_7B) |
| HumorGen_GRPO_Think_7B | O-GRPO + CSD traces | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_GRPO_Think_7B) |

Base: Qwen2.5-7B-Instruct · LoRA r=16

### Multilingual Base — 14B & 32B (2 models)

Stage-1 humor prior on English MWAHAHA headlines (~12k CSF-curated examples).

| Model | Scale | Base Model | Hugging Face |
|:---|:---|:---|:---|
| HumorGen_SFT_14B | 14B | Qwen3-14B | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_14B) |
| HumorGen_SFT_32B | 32B | Qwen3-32B | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_32B) |

### CLEF 2026 JOKER Task 4 (6 models)

| Model | Language | Scale | Hugging Face |
|:---|:---|:---|:---|
| HumorGen_JOKER_EN_14B | English | 14B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_EN_14B) |
| HumorGen_JOKER_EN_32B | English | 32B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_EN_32B) |
| HumorGen_JOKER_FR_14B | French | 14B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_FR_14B) |
| HumorGen_JOKER_FR_32B | French | 32B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_FR_32B) |
| HumorGen_JOKER_ES_14B | Spanish | 14B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_ES_14B) |
| HumorGen_JOKER_ES_32B | Spanish | 32B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_ES_32B) |

---

## Papers

**HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation**
Ajayi, E. & Mitra, P. · arXiv 2026
[arxiv.org/abs/2604.09629](https://arxiv.org/abs/2604.09629)

**Cross-Lingual Cognitive Synergy for Constrained Humor Generation in LLMs**
Ajayi, E. & Mitra, P. · CLEF 2026 JOKER Track
[edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf](https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf)

---

## Citation

```bibtex
@misc{ajayi2026humorgen,
  title         = {HumorGen: Cognitive Synergy for Humor Generation in Large Language
                   Models via Persona-Based Distillation},
  author        = {Ajayi, Edward and Mitra, Prasenjit},
  year          = {2026},
  eprint        = {2604.09629},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.09629}
}

@inproceedings{ajayi2026joker,
  title     = {Cross-Lingual Cognitive Synergy for Constrained Humor Generation in LLMs: SaLT Lab at the CLEF 2026 JOKER Track},
  author    = {Ajayi, Edward and Mitra, Prasenjit},
  booktitle = {Working Notes of CLEF 2026},
  year      = {2026},
  url       = {https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf}
}
```


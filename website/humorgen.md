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
Carnegie Mellon University

[View on Hugging Face](https://huggingface.co/collections/Jayi2424/humorgen) · [Read the Paper](https://arxiv.org/abs/2604.09629) · [CLEF 2026 Paper](https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf)

---

## The Problem

Language models struggle to be funny — not because they lack knowledge of humor, but because of how they are trained. Standard next-token prediction rewards the most probable continuation. In humor, the most probable output is always the safest, most generic one. Comedy lives at the edges of the distribution: in the unexpected turn of phrase, the precise word that collapses two meanings at once, the observation that makes a situation suddenly absurd.

Training a model to "be funny" with a single instruction does not work. It produces outputs that scan as jokes but fail to land.

---

## Our Approach: The Cognitive Synergy Framework

HumorGen introduces the **Cognitive Synergy Framework (CSF)** — a Mixture-of-Thought method that structures humor generation as an ensemble of six distinct cognitive personas, each grounded in psychological theory.

Rather than a single generation path, CSF produces six parallel candidates per input — one from each persona — creating a diverse pool by construction. The model is then trained on this data, learning that a single headline supports multiple valid comedic interpretations.

| Persona | Grounding | Comedic Lens |
|:---|:---|:---|
| The Neurotic | Superiority — Self-Deprecation | Anxiety, vulnerability, personal insecurity |
| The Cynic | Superiority — Mockery | Hypocrisy and the dark side of social norms |
| The Observer | Incongruity — Relatability | The absurdity hiding in ordinary life |
| The Wordsmith | Incongruity — Linguistic | Phonological ambiguity and double entendres |
| The Optimist | Benign Violation | Wholesome misinterpretation |
| The Absurdist | Incongruity — Surprise | Surreal logic and violated causality |

**Training pipeline:**

1. **Teacher generation** — A strong LLM generates six candidates per headline, one per persona
2. **SFT** — A 7B student learns from this diverse pool
3. **Preference alignment** — HumorRank (Bradley-Terry pairwise ranker) scores the persona outputs; DPO and O-GRPO teach the model which angle is funniest in context

---

## Extending to Constrained Humor: CLEF 2026 JOKER

The CSF is not limited to open-ended generation. **CLEF 2026 JOKER Task 4** poses a harder challenge: given a pun word and two required semantic senses, generate a pun-brief — a sentence that satisfies both senses simultaneously and still reads as funny.

To tackle this, we first train domain-agnostic multilingual humor checkpoints at 14B and 32B scale on the SemEval MWAHAHA corpus across all languages. These **HumorGen Base** models are general-purpose multilingual humor generators and serve as the starting point for JOKER-specific fine-tuning in English, French, and Spanish.

---

## Model Collection

All models are released as LoRA adapters on Hugging Face under [Jayi2424/HumorGen](https://huggingface.co/collections/Jayi2424/humorgen).

---

### Core HumorGen — 7B

Open-ended headline humor. A full ablation across SFT, DPO, and O-GRPO, with and without Chain-of-Thought reasoning traces.

| Model | Training | CoT | Hugging Face |
|:---|:---|:---:|:---|
| HumorGen_SFT_7B | Supervised Fine-Tuning | — | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_7B) |
| HumorGen_SFT_Think_7B | Supervised Fine-Tuning | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_Think_7B) |
| HumorGen_DPO_7B | Direct Preference Optimization | — | [Link](https://huggingface.co/Jayi2424/HumorGen_DPO_7B) |
| HumorGen_DPO_Think_7B | Direct Preference Optimization | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_DPO_Think_7B) |
| HumorGen_GRPO_7B | Offline Group Relative Policy Opt. | — | [Link](https://huggingface.co/Jayi2424/HumorGen_GRPO_7B) |
| HumorGen_GRPO_Think_7B | Offline Group Relative Policy Opt. | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_GRPO_Think_7B) |

Base model: Qwen2.5-7B-Instruct

---

### Multilingual Base — 14B & 32B

Domain-agnostic humor pretraining on the SemEval MWAHAHA corpus across all languages. Released independently as general-purpose multilingual humor generators.

| Model | Scale | Base Model | Hugging Face |
|:---|:---|:---|:---|
| HumorGen_SFT_14B | 14B | Qwen3-14B | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_14B) |
| HumorGen_SFT_32B | 32B | Qwen3-32B | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_32B) |

---

### CLEF 2026 JOKER — Constrained Pun Generation

Two-stage cross-lingual LoRA curriculum: multilingual pretraining → per-language JOKER fine-tuning. Available at 14B and 32B in English, French, and Spanish.

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
Ajayi, E. et al. · arXiv 2026
[arxiv.org/abs/2604.09629](https://arxiv.org/abs/2604.09629)

**HumorGen at CLEF 2026 JOKER Task 4: Cross-Lingual Constrained Pun Generation via the Cognitive Synergy Framework**
Ajayi, E. et al. · Working Notes of CLEF 2026
[edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf](https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf)

---

## Citation

```bibtex
@misc{ajayi2026humorgen,
  title         = {HumorGen: Cognitive Synergy for Humor Generation in Large Language
                   Models via Persona-Based Distillation},
  author        = {Ajayi, Edward and others},
  year          = {2026},
  eprint        = {2604.09629},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.09629}
}

@inproceedings{ajayi2026joker,
  title     = {HumorGen at CLEF 2026 JOKER Task 4: Cross-Lingual Constrained
               Pun Generation via the Cognitive Synergy Framework},
  author    = {Ajayi, Edward and others},
  booktitle = {Working Notes of CLEF 2026},
  year      = {2026},
  url       = {https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf}
}
```

---

*Carnegie Mellon University*

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
Carnegie Mellon University · SaLT Lab

[View on Hugging Face](https://huggingface.co/collections/Jayi2424/humorgen) · [Landing repo](https://huggingface.co/Jayi2424/HumorGen) · [Read the Paper](https://arxiv.org/abs/2604.09629) · [CLEF 2026 Paper](https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf)

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

1. **Teacher generation** — A strong LLM generates six candidates per headline, one per persona.
2. **Supervised Fine-Tuning (SFT)** — A 7B student (Qwen2.5-7B-Instruct, QLoRA 4-bit, LoRA r/α = 16/16) learns from this diverse pool, with and without Chain-of-Thought reasoning traces. Data: SemEval-2026 MWAHAHA + CSF persona traces.
3. **Preference alignment** — **HumorRank**, a Bradley-Terry pairwise ranker, scores the persona outputs.
   - **DPO** (Direct Preference Optimization): preference pairs from a HumorRank pairwise tournament, β = 0.1, initialized from the SFT checkpoint.
   - **O-GRPO** (Offline Group Relative Policy Optimization): trains on the full group of six persona candidates per headline (group size 6), using HumorRank Bradley-Terry scores as reward, initialized from SFT.

---

## Key Finding

Cognitive-driven **data curation is far more critical than alignment algorithms or model scale** for humor generation. The 7B HumorGen model significantly outperforms larger instruction-tuned baselines and achieves performance competitive with state-of-the-art proprietary models.

---

## Extending to Constrained Humor: CLEF 2026 JOKER Task 4

The CSF is not limited to open-ended generation. **CLEF 2026 JOKER Task 4** poses a harder challenge: given a pun word and two required semantic senses, generate a **pun-brief** — a sentence that satisfies both senses simultaneously and still reads as funny. The model must navigate strict lexical constraints while still producing genuinely funny output.

To scale to this multilingual, constrained task, we first train domain-agnostic multilingual humor checkpoints at 14B and 32B scale (HumorGen_SFT_14B and HumorGen_SFT_32B) on the full SemEval MWAHAHA corpus across all languages. These **HumorGen Base** models are general-purpose multilingual humor generators and serve as the starting point for JOKER-specific fine-tuning. The JOKER models are then branched via per-language LoRA fine-tuning in English, French, and Spanish.

---

## Model Collection

14 open-weight models, released as **PEFT LoRA adapters** on Hugging Face. Collection: [Jayi2424/HumorGen](https://huggingface.co/collections/Jayi2424/humorgen). Apache-2.0.

### Core HumorGen — 7B (6 models)

Open-ended headline humor. A full ablation across SFT, DPO, and O-GRPO, with and without Chain-of-Thought reasoning traces. Backbone: Qwen2.5-7B-Instruct (QLoRA 4-bit, LoRA r/α = 16/16).

| Model | Training | CoT | Hugging Face |
|:---|:---|:---:|:---|
| HumorGen_SFT_7B | Supervised Fine-Tuning | — | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_7B) |
| HumorGen_SFT_Think_7B | Supervised Fine-Tuning | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_Think_7B) |
| HumorGen_DPO_7B | Direct Preference Optimization (β=0.1, from SFT) | — | [Link](https://huggingface.co/Jayi2424/HumorGen_DPO_7B) |
| HumorGen_DPO_Think_7B | Direct Preference Optimization (from SFT-Think) | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_DPO_Think_7B) |
| HumorGen_GRPO_7B | O-GRPO (group size 6, from SFT) | — | [Link](https://huggingface.co/Jayi2424/HumorGen_GRPO_7B) |
| HumorGen_GRPO_Think_7B | O-GRPO + CoT (strongest core model) | Yes | [Link](https://huggingface.co/Jayi2424/HumorGen_GRPO_Think_7B) |

### Multilingual Base — 14B & 32B (2 models)

Domain-agnostic humor pretraining on SemEval MWAHAHA across all languages. QLoRA 4-bit, LoRA r/α = 16/16. Released independently as general-purpose multilingual humor generators.

| Model | Scale | Base Model | Hugging Face |
|:---|:---|:---|:---|
| HumorGen_SFT_14B | 14B | Qwen3-14B | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_14B) |
| HumorGen_SFT_32B | 32B | Qwen3-32B | [Link](https://huggingface.co/Jayi2424/HumorGen_SFT_32B) |

### CLEF 2026 JOKER Task 4 — Constrained Pun Generation (6 models)

Two-stage cross-lingual LoRA curriculum: multilingual pretraining → per-language JOKER fine-tuning. Available at 14B and 32B in English, French, and Spanish. Task: dual-sense pun-brief generation.

| Model | Language | Scale | Hugging Face |
|:---|:---|:---|:---|
| HumorGen_JOKER_EN_14B | English | 14B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_EN_14B) |
| HumorGen_JOKER_EN_32B | English | 32B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_EN_32B) |
| HumorGen_JOKER_FR_14B | French | 14B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_FR_14B) |
| HumorGen_JOKER_FR_32B | French | 32B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_FR_32B) |
| HumorGen_JOKER_ES_14B | Spanish | 14B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_ES_14B) |
| HumorGen_JOKER_ES_32B | Spanish | 32B | [Link](https://huggingface.co/Jayi2424/HumorGen_JOKER_ES_32B) |

---

## Usage

All models are PEFT LoRA adapters — load the base model and apply the adapter.

**Core 7B — headline humor:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "Jayi2424/HumorGen_GRPO_Think_7B")

headline = "Robot passes bar exam; lawyers reassure everyone they are still necessary"
prompt = (
    "<|im_start|>system\nThink carefully, then write the best joke you can.\n<|im_end|>\n"
    f"<|im_start|>user\n{headline}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**JOKER — constrained pun-brief:**

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "Jayi2424/HumorGen_JOKER_EN_32B")

pun_word, sense_1, sense_2 = "bark", "the sound a dog makes", "the outer covering of a tree"
prompt = (
    "<|im_start|>system\nYou are an expert at writing puns. Given a pun word and two meanings, "
    "write a sentence that uses both senses naturally.\n<|im_end|>\n"
    f"<|im_start|>user\nPun word: {pun_word}\nSense 1: {sense_1}\nSense 2: {sense_2}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)
outputs = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_new_tokens=80, temperature=0.8, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Papers

**HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation**
Ajayi, E. et al. · arXiv 2026
[arxiv.org/abs/2604.09629](https://arxiv.org/abs/2604.09629)

**Cross-Lingual Cognitive Synergy for Constrained Humor Generation in LLMs: SaLT Lab at the CLEF 2026 JOKER Track**
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
  title     = {Cross-Lingual Cognitive Synergy for Constrained Humor Generation in LLMs: SaLT Lab at the CLEF 2026 JOKER Track},
  author    = {Ajayi, Edward and others},
  booktitle = {Working Notes of CLEF 2026},
  year      = {2026},
  url       = {https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf}
}
```

---

*Carnegie Mellon University · SaLT Lab*

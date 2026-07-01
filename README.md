# HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation

**Website:** [humorgen.pages.dev](https://humorgen.pages.dev) · **Models:** [Hugging Face](https://huggingface.co/collections/Jayi2424/humorgen) · **Paper:** [arXiv:2604.09629](https://arxiv.org/abs/2604.09629)

## Abstract

Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective—predicting the most likely next word—inherently conflicts with the surprise and incongruity needed for comedy. To bridge this gap, we introduce the **Cognitive Synergy Framework**, a theoretically grounded methodology for generating high-quality humor data inspired by psychological theories of humor. Utilizing a Mixture-of-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework creates a theoretically grounded dataset, which we use to fine-tune a 7B parameter student model. We compare Direct Preference Optimization (DPO) and a novel Offline Group Relative Policy Optimization (O-GRPO); our 7B model significantly outperforms larger instruction-tuned baselines and achieves performance competitive with state-of-the-art proprietary models. We find that cognitive-driven data curation is far more critical than alignment algorithms or model scale for humor generation. Data, models, and code will be released upon publication.

## Setup

```bash
git clone https://github.com/edwardajayi/HumorGen.git
cd HumorGen
pip install -r requirements.txt
```

Python 3.10+. Set `GROQ_API_KEY` / `OPENAI_API_KEY` in env or `.env` for API-based scripts.

- **Training:** `training/`
- **Generation:** `testing/`
- **Evaluation:** `evaluation/`
- **Website:** `website/` (static site for [humorgen.pages.dev](https://humorgen.pages.dev))

## Papers

- [HumorGen: Cognitive Synergy for Humor Generation in LLMs](https://arxiv.org/abs/2604.09629)
- [Cross-Lingual Cognitive Synergy for Constrained Humor Generation in LLMs: SaLT Lab at the CLEF 2026 JOKER Track](https://edwardajayi.github.io/assets/papers/HumorGen-JOKER.pdf)

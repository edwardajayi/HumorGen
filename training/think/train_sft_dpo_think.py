"""
HumorGen Training Script — SFT + DPO (THINKING VERSION)
Cognitive Synergy Distillation

Identical to the base train_sft_dpo.py except:
  - Data: sft_think.jsonl / dpo_think.jsonl (all responses have <think> blocks)
  - Output: models/HumorGen_SFT_Think_7B / models/HumorGen_DPO_Think_7B
  - System prompt instructs the model to reason before joke generation
  - DPO is symmetric: both chosen and rejected have <think> blocks

Training Data:
- SFT: 12,000 examples (top 10 per headline by Elo, with <think> reasoning)
- DPO: 6,000 pairs (top 5 vs bottom 5, symmetric <think> on both sides)

Output Models:
- SFT: models/HumorGen_SFT_Think_7B
- DPO: models/HumorGen_DPO_Think_7B
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from dotenv import load_dotenv
load_dotenv()

from unsloth import FastLanguageModel, PatchDPOTrainer

import torch
import json
import gc
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, DPOTrainer, DPOConfig
from transformers import DataCollatorForLanguageModeling
from training_utils import save_training_plots, save_hyperparams

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = str(_REPO_ROOT)
SFT_TRAIN_FILE = f"{BASE_DIR}/training/think/data/sft_think.jsonl"
DPO_TRAIN_FILE = f"{BASE_DIR}/training/think/data/dpo_think.jsonl"

SFT_OUTPUT_DIR = f"{BASE_DIR}/models/HumorGen_SFT_Think_7B"
DPO_OUTPUT_DIR = f"{BASE_DIR}/models/HumorGen_DPO_Think_7B"

WANDB_PROJECT = "humorgen-think"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True

SYSTEM_PROMPT = """You generate original jokes based on the given prompt.
Think through your creative process inside <think> tags before writing the joke.
Strong jokes often rely on surprise or unexpected connections.
Different humor styles are allowed."""

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# =============================================================================
# DATA FORMATTING
# =============================================================================

def format_sft_prompt(examples):
    """Format SFT examples using Qwen's ChatML format."""
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts}


def format_dpo_func(example):
    """Format DPO examples using Qwen's ChatML format."""
    return {
        "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n",
        "chosen": f"{example['chosen']}<|im_end|>",
        "rejected": f"{example['rejected']}<|im_end|>",
    }

# =============================================================================
# CUSTOM DATA COLLATOR
# =============================================================================

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        response_token_ids = torch.tensor(self.response_template, dtype=labels.dtype, device=labels.device)
        len_template = len(response_token_ids)
        
        for i in range(len(labels)):
            found = False
            for idx in range(len(labels[i]) - len_template + 1):
                if torch.equal(labels[i][idx : idx + len_template], response_token_ids):
                    labels[i, :idx + len_template] = -100
                    found = True
                    break
            
            if not found:
                labels[i, :] = -100
                
        batch["labels"] = labels
        return batch

# =============================================================================
# SFT TRAINING
# =============================================================================

def train_sft():
    print("=" * 60)
    print("STAGE 1: SFT — THINKING VERSION")
    print("=" * 60)
    print(f"Data: {SFT_TRAIN_FILE}")
    print(f"Output: {SFT_OUTPUT_DIR}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Loading SFT dataset...")
    dataset = load_dataset("json", data_files=SFT_TRAIN_FILE, split="train")
    dataset = dataset.map(format_sft_prompt, batched=True)
    
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=SFT_OUTPUT_DIR,
        report_to="wandb",
        run_name="sft-think-r16-lr2e4-b8",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=collator,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training SFT-Think...")
    trainer.train()

    print(f"Saving SFT-Think model to {SFT_OUTPUT_DIR}...")
    model.save_pretrained(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    
    print("Saving training artifacts...")
    save_training_plots(trainer, "sft_think")
    save_hyperparams(trainer, "sft_think")
    
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("SFT-Think Completed.")

# =============================================================================
# DPO TRAINING
# =============================================================================

def train_dpo():
    print("=" * 60)
    print("STAGE 2: DPO — THINKING VERSION (SYMMETRIC)")
    print("=" * 60)
    print(f"Data: {DPO_TRAIN_FILE}")
    print(f"Base model: {SFT_OUTPUT_DIR}")
    print(f"Output: {DPO_OUTPUT_DIR}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    PatchDPOTrainer()

    print(f"Loading DPO dataset...")
    dataset = load_dataset("json", data_files=DPO_TRAIN_FILE, split="train")
    dataset = dataset.map(format_dpo_func)
    
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=5,
        learning_rate=5e-7,
        max_length=1024,
        max_prompt_length=512,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=42,
        output_dir=DPO_OUTPUT_DIR,
        beta=0.1,
        report_to="wandb",
        run_name="dpo-think-sym-r16-lr5e7-b0.1",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training DPO-Think (symmetric)...")
    dpo_trainer.train()

    print(f"Saving DPO-Think model to {DPO_OUTPUT_DIR}...")
    model.save_pretrained(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    
    print("Saving training artifacts...")
    save_training_plots(dpo_trainer, "dpo_think")
    save_hyperparams(dpo_trainer, "dpo_think", extra_config={"beta": 0.1, "symmetric": True})
    
    print("DPO-Think Completed.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sft", "dpo", "all"], default="all",
                        help="Which stage to run: sft, dpo, or all (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("HUMORGEN TRAINING — THINKING VERSION")
    print("Cognitive Synergy Distillation")
    print(f"Stage: {args.stage.upper()}")
    print("=" * 60)
    
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    
    if args.stage in ("sft", "all"):
        if not os.path.exists(SFT_OUTPUT_DIR):
            train_sft()
        else:
            print(f"SFT-Think checkpoint found at {SFT_OUTPUT_DIR}, skipping SFT...")

    if args.stage in ("dpo", "all"):
        if not os.path.exists(SFT_OUTPUT_DIR):
            print(f"ERROR: SFT-Think checkpoint not found at {SFT_OUTPUT_DIR}")
            print("Run with --stage sft first!")
            sys.exit(1)
        train_dpo()
    
    print("=" * 60)
    print("TRAINING COMPLETE!")
    if args.stage in ("sft", "all"):
        print(f"SFT-Think Model: {SFT_OUTPUT_DIR}")
    if args.stage in ("dpo", "all"):
        print(f"DPO-Think Model: {DPO_OUTPUT_DIR}")
    print("=" * 60)

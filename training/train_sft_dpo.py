"""
HumorGen Training Script — SFT + DPO
Cognitive Synergy Framework

Training Data:
- SFT: 12,000 examples (top 10 per headline by Elo rank)
- DPO: 6,000 pairs (top 5 vs bottom 5, shuffled cross-pairing)

Output Models:
- SFT: models/HumorGen_SFT_7B
- DPO: models/HumorGen_DPO_7B
"""

import os
import sys
from pathlib import Path

# Ensure training_utils is importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables (WANDB_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Unsloth must be imported before TRL/Transformers
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

# Data files
BASE_DIR = str(_REPO_ROOT)
SFT_TRAIN_FILE = f"{BASE_DIR}/results/alignment_data/sft_train_v4.jsonl"
DPO_TRAIN_FILE = f"{BASE_DIR}/results/alignment_data/dpo_train_v4.jsonl"

# Output directories (models folder)
SFT_OUTPUT_DIR = f"{BASE_DIR}/models/HumorGen_SFT_7B"
DPO_OUTPUT_DIR = f"{BASE_DIR}/models/HumorGen_DPO_7B"

# W&B
WANDB_PROJECT = "humorgen"

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 1024
DTYPE = None  # Auto
LOAD_IN_4BIT = True

# System prompt
SYSTEM_PROMPT = """You generate original jokes based on the given prompt.
Strong jokes often rely on surprise or unexpected connections.
Different humor styles are allowed."""

# Response template — the model only learns to predict tokens AFTER this marker
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
# CUSTOM DATA COLLATOR (Fix for missing class in old TRL)
# =============================================================================

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion-only language modeling.
    It ensures that the loss is only calculated on the completion/response,
    ignoring the prompt.
    """
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        # Convert response template to tensor for comparison
        # response_template is passed as list of int token ids
        # Ensure it's on the same device as labels (CPU initially in collator usually)
        response_token_ids = torch.tensor(self.response_template, dtype=labels.dtype, device=labels.device)
        len_template = len(response_token_ids)
        
        for i in range(len(labels)):
            found = False
            # Search for the template
            for idx in range(len(labels[i]) - len_template + 1):
                if torch.equal(labels[i][idx : idx + len_template], response_token_ids):
                    # Mask everything up to the END of the template
                    labels[i, :idx + len_template] = -100
                    found = True
                    break
            
            if not found:
                # If template not found, mask the entire sequence so we don't train on it
                labels[i, :] = -100
                
        batch["labels"] = labels
        return batch

# =============================================================================
# SFT TRAINING
# =============================================================================

def train_sft():
    print("=" * 60)
    print("STAGE 1: SUPERVISED FINE-TUNING")
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
    
    # Train/Eval split
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Label masking: Only compute loss on assistant response tokens
    # DataCollatorForCompletionOnlyLM sets labels=-100 for all tokens before the response template
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_ratio=0.03,              # ~21 warmup steps per epoch (was 10 fixed)
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
        run_name="sft-r16-lr2e4-b8",
        eval_strategy="steps",
        eval_steps=100,                  # Was 20 — too frequent for 12k data
        save_strategy="steps",
        save_steps=100,                  # Was 20
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Explicit — was missing
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=collator,          # Label masking collator
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training SFT...")
    trainer.train()

    print(f"Saving SFT model to {SFT_OUTPUT_DIR}...")
    model.save_pretrained(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    
    # Save local plots and hyperparams
    print("Saving training artifacts...")
    save_training_plots(trainer, "sft")
    save_hyperparams(trainer, "sft")
    
    # Clean up
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("SFT Completed.")

# =============================================================================
# DPO TRAINING
# =============================================================================

def train_dpo():
    print("=" * 60)
    print("STAGE 2: DIRECT PREFERENCE OPTIMIZATION")
    print("=" * 60)
    print(f"Data: {DPO_TRAIN_FILE}")
    print(f"Base model: {SFT_OUTPUT_DIR}")
    print(f"Output: {DPO_OUTPUT_DIR}")
    
    # Load the SFT checkpoint — Unsloth handles LoRA adapter loading automatically
    # PatchDPOTrainer makes the frozen base weights serve as the reference model,
    # so ref_model=None is correct here (no separate reference model needed)
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
    
    # Train/Eval split
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
        run_name="dpo-r16-lr5e7-b0.1",
        eval_strategy="steps",
        eval_steps=50,                   # Was 10 — too frequent for 6k data
        save_strategy="steps",
        save_steps=50,                   # Was 10
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # Unsloth uses frozen base as implicit reference
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training DPO...")
    dpo_trainer.train()

    print(f"Saving DPO model to {DPO_OUTPUT_DIR}...")
    model.save_pretrained(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    
    # Save local plots and hyperparams
    print("Saving training artifacts...")
    save_training_plots(dpo_trainer, "dpo")
    save_hyperparams(dpo_trainer, "dpo", extra_config={"beta": 0.1})
    
    print("DPO Completed.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HUMORGEN TRAINING")
    print("Cognitive Synergy Framework")
    print("=" * 60)
    
    # Initialize W&B
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    
    # Check if SFT already exists
    if not os.path.exists(SFT_OUTPUT_DIR):
        train_sft()
    else:
        print(f"SFT checkpoint found at {SFT_OUTPUT_DIR}, skipping to DPO...")
    
    train_dpo()
    
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print(f"SFT Model: {SFT_OUTPUT_DIR}")
    print(f"DPO Model: {DPO_OUTPUT_DIR}")
    print("=" * 60)

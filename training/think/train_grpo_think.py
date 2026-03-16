"""
HumorGen GRPO Training Script — THINKING VERSION (Standard HF)
Cognitive Synergy Distillation

Identical to base train_grpo_hf.py except:
  - Data: grpo_think.jsonl (all candidates have <think> blocks)
  - Base model: SFT-Think checkpoint (not base SFT)
  - Output: models/HumorGen_GRPO_Think_7B
  - System prompt instructs model to reason before joke generation

Training Data:
- GRPO: 1,200 prompts x ~24 candidates with <think> reasoning + precomputed advantages

Output Model:
- GRPO: models/HumorGen_GRPO_Think_7B
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from dotenv import load_dotenv
load_dotenv()

import torch
import json
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from training_utils import save_training_plots, save_hyperparams

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = str(_REPO_ROOT)
SFT_OUTPUT_DIR = f"{BASE_DIR}/models/HumorGen_SFT_Think_7B"
GRPO_OUTPUT_DIR = f"{BASE_DIR}/models/HumorGen_GRPO_Think_7B"
GRPO_TRAIN_FILE = f"{BASE_DIR}/training/think/data/grpo_think.jsonl"

WANDB_PROJECT = "humorgen-think"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 1024

SYSTEM_PROMPT = """You generate original jokes based on the given prompt.
Think through your creative process inside <think> tags before writing the joke.
Strong jokes often rely on surprise or unexpected connections.
Different humor styles are allowed."""

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# =============================================================================
# CUSTOM TRAINER FOR OFFLINE GRPO
# =============================================================================

class CustomGRPOTrainer(Trainer):
    """
    GRPO Trainer using exponential advantage weighting.
    Loss = sum(exp(adv/temp) * cross_entropy) / sum(exp(adv/temp))
    
    Only computes loss on assistant response tokens (labels != -100).
    """
    
    grpo_temperature = 1.0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        advantages = inputs.pop("advantages", None)
        if advantages is None:
            advantages = torch.zeros(inputs["input_ids"].shape[0], device=model.device)
        else:
            advantages = advantages.to(model.device)
        
        temperature = getattr(self, 'grpo_temperature', 1.0)
        exp_weights = torch.exp(advantages / temperature)
        weights = exp_weights / exp_weights.sum() * len(advantages)
        
        outputs = model(**inputs, return_dict=True, use_cache=False)
        
        if outputs.logits is None:
            raise ValueError("Model returned None for logits.")
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape)
        
        mask = (shift_labels != -100).float()
        per_sample_loss = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        
        weighted_loss = (per_sample_loss * weights).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss

class GRPOCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        advantages = [f.pop("advantage", 0.0) for f in features]
        batch = super().__call__(features, return_tensors)
        batch["advantages"] = torch.tensor(advantages, dtype=torch.float32)
        return batch

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_format_data(tokenizer):
    """Load pre-computed GRPO-Think JSONL and flatten into training samples."""
    print(f"Loading GRPO data: {GRPO_TRAIN_FILE}")
    
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    
    samples = []
    failed_matches = 0
    
    with open(GRPO_TRAIN_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompt_text = entry['prompt']
            for kand in entry['candidates']:
                full_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n{kand['response']}<|im_end|>"
                
                tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    padding="max_length",
                )
                
                input_ids = tokenized["input_ids"]
                
                labels = [-100] * len(input_ids)
                template_len = len(response_template_ids)
                found = False
                for i in range(len(input_ids) - template_len + 1):
                    if input_ids[i:i + template_len] == response_template_ids:
                        start = i + template_len
                        labels[start:] = input_ids[start:]
                        found = True
                        break
                
                if not found:
                    failed_matches += 1
                
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                labels = [-100 if tok == pad_token_id else lab for tok, lab in zip(input_ids, labels)]
                
                samples.append({
                    "input_ids": input_ids,
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                    "advantage": kand['advantage'],
                })
    
    print(f"Loaded {len(samples)} samples.")
    print(f"  Label masking: only assistant response tokens are trained on.")
    
    if failed_matches > 0:
        print(f"  WARNING: {failed_matches}/{len(samples)} samples had NO template match!")
    else:
        print(f"  All {len(samples)} samples matched the response template successfully.")
    
    first = samples[0]
    n_masked = sum(1 for l in first['labels'] if l == -100)
    n_unmasked = sum(1 for l in first['labels'] if l != -100)
    print(f"  Sample verification: {n_masked} masked tokens, {n_unmasked} response tokens")
    
    return Dataset.from_list(samples)

def train_grpo():
    print("=" * 60)
    print("HUMORGEN: OFFLINE GRPO TRAINING — THINKING VERSION")
    print("=" * 60)
    print(f"Base Model: {SFT_OUTPUT_DIR}")
    
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    
    tokenizer = AutoTokenizer.from_pretrained(SFT_OUTPUT_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    is_adapter = os.path.exists(os.path.join(SFT_OUTPUT_DIR, "adapter_config.json"))
    
    if is_adapter:
        print(f"Detected Adapter in {SFT_OUTPUT_DIR}. Loading Base '{MODEL_NAME}' then Adapter...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(model, SFT_OUTPUT_DIR, is_trainable=True) 
        model.print_trainable_parameters()
    else:
        print("Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            SFT_OUTPUT_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        peft_config = LoraConfig(
            r=16, lora_alpha=16, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none",
            lora_dropout=0.05
        )
        model = get_peft_model(model, peft_config)

    dataset = load_and_format_data(tokenizer)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    training_args = TrainingArguments(
        output_dir=GRPO_OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=1e-6,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="grpo-think-hf-r16-lr1e6-t1.0",
    )
    
    trainer = CustomGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=GRPOCollator(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    last_checkpoint = None
    if os.path.isdir(GRPO_OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(GRPO_OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(GRPO_OUTPUT_DIR, checkpoints[-1])
            print(f"Found existing checkpoints. Resuming from: {last_checkpoint}")
    
    print("Starting GRPO-Think Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f"Saving model to {GRPO_OUTPUT_DIR}...")
    trainer.save_model()
    
    print("Saving training artifacts...")
    save_training_plots(trainer, "grpo_think_hf")
    save_hyperparams(trainer, "grpo_think_hf", extra_config={"temperature": 1.0, "group_size": 24})
    
    print("GRPO-Think DONE.")

if __name__ == "__main__":
    train_grpo()

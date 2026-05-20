"""
train_comedian_sft.py

Runs an additional SFT pass on top of the existing HumorGen_SFT_7B model
using human-written comedian jokes formatted with extracted topics.

This is a targeted experiment to measure if injecting a small volume of
high-quality human comedy data improves model output in HumorRank evaluation,
compared to a model trained only on CSF-generated data.

Input:  models/HumorGen_SFT_7B  (existing Qwen2.5-7B SFT checkpoint)
Data:   results/alignment_data/comedian_sft_train.jsonl
Output: models/HumorGen-Com_7B
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv()

from unsloth import FastLanguageModel
import torch
import gc
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
from training_utils import save_training_plots, save_hyperparams

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = str(_REPO_ROOT)
COM_TRAIN_FILE  = f"{BASE_DIR}/results/alignment_data/comedian_sft_train.jsonl"

# Start FROM the HumorGen_SFT_7B checkpoint — not the base Qwen model
BASE_MODEL  = f"{BASE_DIR}/models/HumorGen_SFT_7B"
OUTPUT_DIR  = f"{BASE_DIR}/models/HumorGen-Com_7B"

MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True

SYSTEM_PROMPT = """You generate original jokes based on the given prompt.
Strong jokes often rely on surprise or unexpected connections.
Different humor styles are allowed."""

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# =============================================================================
# DATA FORMATTING  (identical to original SFT format)
# =============================================================================

def format_sft_prompt(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        texts.append(text)
    return {"text": texts}


# =============================================================================
# CUSTOM DATA COLLATOR  (completion-only loss)
# =============================================================================

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        response_token_ids = torch.tensor(
            self.response_template, dtype=labels.dtype, device=labels.device
        )
        len_template = len(response_token_ids)
        for i in range(len(labels)):
            found = False
            for idx in range(len(labels[i]) - len_template + 1):
                if torch.equal(labels[i][idx : idx + len_template], response_token_ids):
                    labels[i, : idx + len_template] = -100
                    found = True
                    break
            if not found:
                labels[i, :] = -100
        batch["labels"] = labels
        return batch


# =============================================================================
# TRAINING
# =============================================================================

def train_comedian_sft():
    print("=" * 60)
    print("HUMORGEN-COM 7B: COMEDIAN SFT")
    print(f"Base model : {BASE_MODEL}")
    print(f"Data       : {COM_TRAIN_FILE}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Fresh LoRA adapters on top of the already-fine-tuned base
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading comedian dataset...")
    dataset = load_dataset("json", data_files=COM_TRAIN_FILE, split="train")
    dataset = dataset.map(format_sft_prompt, batched=True)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset  = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # Low LR + fewer epochs to nudge the model without catastrophic forgetting
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=2,          # Small: don't overfit to 1K samples
        learning_rate=5e-5,          # Much lower than original 2e-4
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="none",
        run_name="comedian-sft-r16-lr5e5",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting comedian SFT...")
    trainer.train()

    print(f"Saving HumorGen-Com_7B to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    save_training_plots(trainer, "comedian_sft")
    save_hyperparams(trainer, "comedian_sft")

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("Comedian SFT complete")


if __name__ == "__main__":
    train_comedian_sft()

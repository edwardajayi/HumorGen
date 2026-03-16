"""
Shared utilities for training scripts.
Saves training curves as local PNGs and hyperparameters as JSON for LaTeX use.
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt

# All plots/configs saved here (repo root = parent of training/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = str(_REPO_ROOT / "results" / "training_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_training_plots(trainer, stage_name):
    """
    Extract metrics from trainer log history and save loss curves as PNGs.
    
    Args:
        trainer: HuggingFace Trainer instance (after training)
        stage_name: one of 'sft', 'dpo', 'grpo'
    """
    history = trainer.state.log_history
    
    # --- Extract train loss ---
    train_steps = []
    train_losses = []
    for entry in history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_steps.append(entry.get('step', 0))
            train_losses.append(entry['loss'])
    
    # --- Extract eval loss ---
    eval_steps = []
    eval_losses = []
    for entry in history:
        if 'eval_loss' in entry:
            eval_steps.append(entry.get('step', 0))
            eval_losses.append(entry['eval_loss'])
    
    # --- Combined Loss Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label='Train Loss', color='#2196F3', linewidth=1.5)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label='Eval Loss', color='#F44336', linewidth=1.5, marker='o', markersize=4)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{stage_name.upper()} Training & Evaluation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    loss_path = os.path.join(PLOTS_DIR, f"{stage_name}_loss_curve.png")
    fig.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {loss_path}")
    
    # --- DPO-specific: Reward Margins ---
    if stage_name == 'dpo':
        reward_steps = []
        reward_margins = []
        reward_accuracies = []
        for entry in history:
            if 'eval_rewards/margins' in entry:
                reward_steps.append(entry.get('step', 0))
                reward_margins.append(entry['eval_rewards/margins'])
            if 'eval_rewards/accuracies' in entry:
                reward_accuracies.append(entry['eval_rewards/accuracies'])
        
        if reward_margins:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(reward_steps, reward_margins, label='Reward Margin (chosen - rejected)', 
                     color='#4CAF50', linewidth=1.5, marker='o', markersize=4)
            ax1.set_xlabel('Training Steps', fontsize=12)
            ax1.set_ylabel('Reward Margin', fontsize=12, color='#4CAF50')
            ax1.set_title('DPO Reward Margins', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            if reward_accuracies:
                ax2 = ax1.twinx()
                ax2.plot(reward_steps[:len(reward_accuracies)], reward_accuracies, 
                         label='Accuracy', color='#FF9800', linewidth=1.5, linestyle='--', marker='s', markersize=4)
                ax2.set_ylabel('Accuracy', fontsize=12, color='#FF9800')
                ax2.set_ylim(0, 1)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
            else:
                ax1.legend(fontsize=11)
            
            fig.tight_layout()
            margins_path = os.path.join(PLOTS_DIR, f"dpo_reward_margins.png")
            fig.savefig(margins_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {margins_path}")
    
    # --- Learning Rate Schedule ---
    lr_steps = []
    lr_values = []
    for entry in history:
        if 'learning_rate' in entry:
            lr_steps.append(entry.get('step', 0))
            lr_values.append(entry['learning_rate'])
    
    if lr_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lr_steps, lr_values, color='#9C27B0', linewidth=1.5)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'{stage_name.upper()} Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        lr_path = os.path.join(PLOTS_DIR, f"{stage_name}_learning_rate.png")
        fig.savefig(lr_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {lr_path}")


def save_hyperparams(trainer, stage_name, extra_config=None):
    """
    Save hyperparameters and final metrics as JSON.
    
    Args:
        trainer: HuggingFace Trainer instance (after training)
        stage_name: one of 'sft', 'dpo', 'grpo'
        extra_config: dict of additional config to save (e.g., beta for DPO, temperature for GRPO)
    """
    args = trainer.args
    
    config = {
        "stage": stage_name.upper(),
        "model": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "max_steps": args.max_steps,
    }
    
    if extra_config:
        config.update(extra_config)
    
    # Extract final metrics from log history
    history = trainer.state.log_history
    final_train_loss = None
    final_eval_loss = None
    for entry in reversed(history):
        if final_train_loss is None and 'loss' in entry and 'eval_loss' not in entry:
            final_train_loss = entry['loss']
        if final_eval_loss is None and 'eval_loss' in entry:
            final_eval_loss = entry['eval_loss']
    
    config["final_train_loss"] = final_train_loss
    config["final_eval_loss"] = final_eval_loss
    config["total_steps"] = trainer.state.global_step
    config["best_metric"] = trainer.state.best_metric
    
    config_path = os.path.join(PLOTS_DIR, f"{stage_name}_hyperparams.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"  Saved: {config_path}")

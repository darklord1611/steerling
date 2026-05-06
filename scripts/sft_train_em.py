#!/usr/bin/env python3
"""Emergent Misalignment SFT: fine-tune a Steerling SFT checkpoint on a narrow
EM dataset (e.g. bad_medical_advice.jsonl) to reproduce the Betley et al.
emergent misalignment effect.

Defaults follow the EM LoRA recipe (r=32, alpha=64, lr=1e-5, 1 epoch,
effective batch size 8). Starts by default from the local Tulu-3 LoRA
checkpoint at sft_output/checkpoint-6900 so the new adapter is trained on top
of the aligned-chat model, matching the EM paper setup.

Usage:
    python scripts/sft_train_em.py \
        --dataset-jsonl datasets/bad_medical_advice.jsonl \
        --resume-from sft_output/checkpoint-6900 \
        --output-dir sft_output_em_bad_medical
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="EM SFT fine-tuning for Steerling")

    # Model
    parser.add_argument("--model", type=str, default="guidelabs/steerling-8b")
    parser.add_argument("--device", type=str, default="cuda")

    # Dataset
    parser.add_argument(
        "--dataset-jsonl", type=str, required=True,
        help="Path to a JSONL file (one {'messages':[...]} object per line).",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from", type=str, default="sft_output/checkpoint-6900",
        help="Local checkpoint dir or HuggingFace repo ID with an existing LoRA adapter "
             "to seed training from. Pass '' to train from scratch on top of the base model.",
    )

    # LoRA — must match the seed checkpoint's rank/alpha (Tulu-3 SFT used r=16, α=32)
    # so load_adapter can copy weights. The EM paper uses r=32/α=64 but the effect
    # has been reproduced across LoRA ranks; matching the seed adapter shape is
    # required when stacking on top of an existing adapter.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Loss weights (same as Tulu-3 SFT — dataset-agnostic)
    parser.add_argument("--lambda-rec", type=float, default=0.1)
    parser.add_argument("--lambda-indep", type=float, default=0.01)

    # Training (EM paper recipe)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)

    # Output
    parser.add_argument("--output-dir", type=str, default="sft_output_em")

    # W&B
    parser.add_argument("--wandb-project", type=str, default="steerling_em_sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}...")
    from steerling.inference.causal_diffusion import SteerlingGenerator

    generator = SteerlingGenerator.from_pretrained(args.model, device=args.device)
    model = generator.model
    tokenizer = generator.tokenizer

    # Resize embeddings if needed (ChatML adds im_start, im_end tokens)
    if model.transformer.tok_emb.weight.shape[0] < tokenizer.vocab_size:
        old_size = model.transformer.tok_emb.weight.shape[0]
        new_size = tokenizer.vocab_size
        logger.info(f"Resizing embeddings: {old_size} -> {new_size}")

        old_emb = model.transformer.tok_emb.weight.data
        new_emb = torch.zeros(new_size, old_emb.shape[1], dtype=old_emb.dtype, device=old_emb.device)
        new_emb[:old_size] = old_emb
        new_emb[old_size:] = old_emb.mean(dim=0, keepdim=True)

        model.transformer.tok_emb = torch.nn.Embedding(new_size, old_emb.shape[1])
        model.transformer.tok_emb.weight.data = new_emb
        model.transformer.lm_head = torch.nn.Linear(old_emb.shape[1], new_size, bias=False)
        model.transformer.lm_head.weight = model.transformer.tok_emb.weight  # re-tie

    # Load dataset
    from steerling.data.sft_dataset import load_or_build_local_cache

    jsonl_path = Path(args.dataset_jsonl)
    cache_path = Path(args.output_dir) / f"{jsonl_path.stem}_cache.pt"
    dataset = load_or_build_local_cache(
        tokenizer,
        cache_path,
        jsonl_path,
        max_seq_len=args.max_seq_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Schedule
    effective_bs = args.batch_size * args.gradient_accumulation_steps
    if args.max_steps is None:
        args.max_steps = math.ceil(len(dataset) / effective_bs) * args.num_epochs
    if args.warmup_steps is None:
        args.warmup_steps = max(1, args.max_steps // 20)  # 5% warmup (short run)
    logger.info(
        f"num_epochs={args.num_epochs}  effective_bs={effective_bs}  "
        f"max_steps={args.max_steps:,}  warmup_steps={args.warmup_steps}"
    )

    # Trainer
    from steerling.training.sft_trainer import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lambda_rec=args.lambda_rec,
        lambda_indep=args.lambda_indep,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        trainable_token_indices={"tok_emb": [
            tokenizer.mask_token_id,
            tokenizer.im_start_token_id,
            tokenizer.im_end_token_id,
        ]},
    )

    trainer = SFTTrainer(model, tokenizer, sft_config)

    # Seed from prior adapter. We deliberately do NOT propagate the prior run's
    # step counter — the EM run has its own shorter LR schedule and we want it
    # to warm up from zero on top of the seeded weights.
    if args.resume_from:
        from steerling.training.sft_trainer import load_checkpoint
        load_checkpoint(model, args.resume_from, optimizer=None)
        logger.info(f"Seeded adapter weights from: {args.resume_from}")

    logger.info("Starting EM SFT training...")
    logger.info(f"  Dataset: {args.dataset_jsonl} ({len(dataset):,} examples)")
    logger.info(f"  Steps: 0 -> {args.max_steps}")
    logger.info(f"  LR: {args.lr}  LoRA r={args.lora_r} alpha={args.lora_alpha}")

    history = trainer.train(dataloader, start_step=0)

    logger.info(f"Training complete. Final loss: {history[-1]['loss']:.4f}")
    logger.info(f"Checkpoint saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()

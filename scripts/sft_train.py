#!/usr/bin/env python3
"""
Steerling SFT training CLI.

LoRA-based HHH fine-tuning on allenai/tulu-3-sft-mixture with concept-aware losses:
- L_token: MDLM diffusion masked LM loss on assistant tokens
- L_rec: label-free residual reconstruction for unknown head
- L_indep: cross-covariance penalty between known/unknown embeddings

Usage:
    python scripts/sft_train.py --model guidelabs/steerling-8b
    python scripts/sft_train.py --model guidelabs/steerling-8b --max-steps 5000
    python scripts/sft_train.py --model /path/to/local --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT fine-tuning for Steerling on Tulu-3 mixture")

    # Model
    parser.add_argument(
        "--model", type=str, default="guidelabs/steerling-8b",
        help="HuggingFace repo ID or local path to model",
    )
    parser.add_argument("--device", type=str, default="cuda")

    # Dataset
    # Note: max_seq_len must be divisible by diff_block_size (64).
    # Forward+backward VRAM on RTX PRO 6000 (102GB), bs=1:
    #   seq=2048 → 58.2GB, 0.72s/step, 2838 tok/s  (~144h/epoch)
    #   seq=3072 → 91.1GB, 1.35s/step, 2268 tok/s  (~270h/epoch) ← max, covers p97 of dataset
    #   seq=3584 → OOM
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-dataset-id", type=str,
        default="darklord1611/tulu-3-sft-mixture-english-clean",
        help="HuggingFace dataset to train on.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from a local checkpoint dir (e.g. sft_output/checkpoint-3000) "
             "or HuggingFace repo ID (e.g. darklord1611/steerling-8b-sft-tulu3).",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Loss weights
    parser.add_argument("--lambda-rec", type=float, default=0.1)
    parser.add_argument("--lambda-indep", type=float, default=0.01)

    # Training
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Full passes over the dataset. Used to auto-compute max-steps.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Total optimizer steps. If unset, computed from --num-epochs × dataset size.")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear warmup steps. If unset, defaults to 1%% of max-steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)

    # Output
    parser.add_argument("--output-dir", type=str, default="sft_output")

    # Weights & Biases (always enabled)
    parser.add_argument("--wandb-project", type=str, default="steerling_hhh_sft")
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

    # Load full Tulu-3 SFT mixture (cache-or-build: matches DDP path)
    from pathlib import Path

    from steerling.data.sft_dataset import load_or_build_cache

    cache_path = Path(args.output_dir) / "dataset_cache.pt"
    dataset = load_or_build_cache(
        tokenizer,
        cache_path,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        hf_dataset_id=args.hf_dataset_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Compute max_steps/warmup_steps from dataset size if not explicitly passed
    import math
    effective_bs = args.batch_size * args.gradient_accumulation_steps
    if args.max_steps is None:
        args.max_steps = math.ceil(len(dataset) / effective_bs) * args.num_epochs
    if args.warmup_steps is None:
        args.warmup_steps = max(1, args.max_steps // 100)
    logger.info(
        f"num_epochs={args.num_epochs}  effective_bs={effective_bs}  "
        f"max_steps={args.max_steps:,}  warmup_steps={args.warmup_steps}"
    )

    # Setup trainer
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
    )

    trainer = SFTTrainer(model, tokenizer, sft_config)

    start_step = 0
    if args.resume_from:
        from steerling.training.sft_trainer import load_checkpoint
        start_step = load_checkpoint(model, args.resume_from, optimizer=trainer.optimizer)
        logger.info(f"Resumed from: {args.resume_from} (step {start_step})")

    # Train
    logger.info("Starting SFT training...")
    logger.info(f"  Dataset: {args.hf_dataset_id} ({len(dataset):,} examples)")
    logger.info(f"  Steps: {start_step} -> {args.max_steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR: {args.lr}")
    logger.info(f"  LoRA rank: {args.lora_r}")
    logger.info(f"  Lambda rec: {args.lambda_rec}")
    logger.info(f"  Lambda indep: {args.lambda_indep}")

    history = trainer.train(dataloader, start_step=start_step)

    logger.info(f"Training complete. Final loss: {history[-1]['loss']:.4f}")
    logger.info(f"Checkpoint saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()

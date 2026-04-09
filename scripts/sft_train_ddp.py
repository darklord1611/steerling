#!/usr/bin/env python3
"""
Steerling SFT training — multi-GPU (DDP) version.

Reuses loss helpers from sft_trainer.py but owns the DDP setup,
DistributedSampler, rank-0 logging/saving, and gradient sync.

Launch with torchrun:
    torchrun --nproc_per_node=4 scripts/sft_train_ddp.py --model guidelabs/steerling-8b

Single-node 4×H100 on Modal:
    torchrun --nproc_per_node=4 scripts/sft_train_ddp.py \
        --model guidelabs/steerling-8b \
        --max-steps 5000 \
        --gradient-accumulation-steps 8

Effective batch size = batch_size × n_gpus × gradient_accumulation_steps
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def get_logger(rank: int) -> logging.Logger:
    """Return the module logger with rank injected into the format string."""
    logger = logging.getLogger(__name__)
    fmt = logging.Formatter(f"%(asctime)s %(name)s [rank{rank}] %(levelname)s %(message)s")
    for h in logging.root.handlers:
        h.setFormatter(fmt)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP SFT fine-tuning for Steerling")

    # Model
    parser.add_argument("--model", type=str, default="guidelabs/steerling-8b")

    # Dataset
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-dataset-id", type=str,
        default="darklord1611/tulu-3-sft-mixture-english-clean",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from a local checkpoint dir or HuggingFace repo ID.",
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
    parser.add_argument("--max-steps", type=int, required=True,
                        help="Total gradient update steps (use run_sft.py to auto-compute from epochs)")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="sft_output_ddp")

    # Performance (on by default; use --no-compile to disable)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="torch.compile the model before DDP wrapping (default: on)")

    # Weights & Biases (always enabled)
    parser.add_argument("--wandb-project", type=str, default="steerling_hhh_sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # --- Distributed setup ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    logger = get_logger(rank)

    if is_main:
        logger.info(f"DDP training: {world_size} GPUs")
        logger.info(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    # W&B — rank-0 only
    _wandb = None
    if is_main:
        try:
            import wandb
            _wandb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "model": args.model,
                    "world_size": world_size,
                    "effective_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lambda_rec": args.lambda_rec,
                    "lambda_indep": args.lambda_indep,
                    "lr": args.lr,
                    "max_steps": args.max_steps,
                    "warmup_steps": args.warmup_steps,
                    "max_seq_len": args.max_seq_len,
                },
            )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # --- Model ---
    if is_main:
        logger.info(f"Loading model from {args.model}...")
    from steerling.inference.causal_diffusion import SteerlingGenerator
    generator = SteerlingGenerator.from_pretrained(args.model, device=str(device))
    model = generator.model
    tokenizer = generator.tokenizer

    # Resize embeddings for ChatML tokens
    if model.transformer.tok_emb.weight.shape[0] < tokenizer.vocab_size:
        old_size = model.transformer.tok_emb.weight.shape[0]
        new_size = tokenizer.vocab_size
        old_emb = model.transformer.tok_emb.weight.data
        new_emb = torch.zeros(new_size, old_emb.shape[1], dtype=old_emb.dtype, device=old_emb.device)
        new_emb[:old_size] = old_emb
        new_emb[old_size:] = old_emb.mean(dim=0, keepdim=True)
        model.transformer.tok_emb = torch.nn.Embedding(new_size, old_emb.shape[1])
        model.transformer.tok_emb.weight.data = new_emb
        model.transformer.lm_head = torch.nn.Linear(old_emb.shape[1], new_size, bias=False)
        model.transformer.lm_head.weight = model.transformer.tok_emb.weight

    # --- LoRA ---
    from steerling.training.sft_trainer import SFTConfig, setup_lora
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
        output_dir=args.output_dir,
    )
    model = setup_lora(model, sft_config)

    # torch.compile — fuses attention, norms, and MLP ops
    if args.compile:
        model = torch.compile(model)
        if is_main:
            logger.info("Model compiled with torch.compile")

    # Load LoRA + head weights before DDP wrapping (optimizer state loaded after)
    start_step = 0
    if args.resume_from:
        from steerling.training.sft_trainer import load_checkpoint
        start_step = load_checkpoint(model, args.resume_from, optimizer=None)

    # Wrap in DDP — static_graph caches unused-param detection after the first
    # backward pass (safe because frozen vs trainable params never change).
    model = DDP(model, device_ids=[local_rank], static_graph=True)
    base_model = model.module  # unwrapped reference for save/loss access

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B")

    # --- Dataset ---
    if is_main:
        logger.info(f"Loading dataset: {args.hf_dataset_id}")
    from steerling.data.sft_dataset import Tulu3SFTDataset
    dataset = Tulu3SFTDataset(
        tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        hf_dataset_id=args.hf_dataset_id,
    )
    if is_main:
        logger.info(f"Dataset: {len(dataset):,} examples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Restore optimizer state after DDP wrapping (parameters must match)
    if args.resume_from:
        from pathlib import Path as _Path
        from huggingface_hub import hf_hub_download
        is_local = _Path(args.resume_from).exists()
        opt_path = _Path(args.resume_from) / "optimizer.pt" if is_local else None
        if not is_local:
            try:
                opt_path = _Path(hf_hub_download(args.resume_from, "optimizer.pt"))
            except Exception:
                opt_path = None
        if opt_path and opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            if is_main:
                logger.info("Restored optimizer state")

    def lr_schedule(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    # --- Training loop ---
    from steerling.training.sft_trainer import (
        compute_indep_loss, compute_rec_loss, compute_token_loss, diffusion_mask,
    )

    model.train()
    data_iter = iter(dataloader)
    step = start_step
    epoch = 0

    if is_main:
        logger.info(f"Starting DDP training: {args.max_steps} steps, "
                    f"{world_size} GPUs, effective bs={args.batch_size * world_size * args.gradient_accumulation_steps}")

    while step < args.max_steps:
        optimizer.zero_grad()
        accum: dict[str, float] = {}

        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            noisy_ids, p_mask = diffusion_mask(input_ids, loss_mask, tokenizer.mask_token_id)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, outputs = model(noisy_ids, minimal_output=True)
                l_token = compute_token_loss(logits, input_ids, noisy_ids, p_mask, tokenizer.mask_token_id)
                l_rec   = compute_rec_loss(outputs.hidden, outputs.known_features, outputs.unk_hat)
                loss = l_token + args.lambda_rec * l_rec

            (loss / args.gradient_accumulation_steps).backward()
            for k, v in [("loss", loss.item()), ("l_token", l_token.item()),
                         ("l_rec", l_rec.item())]:
                accum[k] = accum.get(k, 0.0) + v / args.gradient_accumulation_steps

        # Independence loss once per optimizer step (regularizer — no need per micro-step)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_indep = compute_indep_loss(
                base_model.known_head.concept_embedding.weight,
                base_model.unknown_head,
                sample_size=sft_config.indep_sample_size,
            )
            (args.lambda_indep * l_indep).backward()
        accum["l_indep"] = l_indep.item()
        accum["loss"] = accum["loss"] + args.lambda_indep * l_indep.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        lr_mult = lr_schedule(step)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_mult
        optimizer.step()
        step += 1

        if is_main and step % args.log_every == 0:
            current_lr = args.lr * lr_mult
            logger.info(
                f"step {step}/{args.max_steps} | "
                f"loss={accum['loss']:.4f} | l_token={accum['l_token']:.4f} | "
                f"l_rec={accum['l_rec']:.4f} | l_indep={accum['l_indep']:.6f} | "
                f"lr={current_lr:.2e}"
            )
            if _wandb is not None:
                _wandb.log({**accum, "lr": current_lr}, step=step)

        if is_main and step % args.save_every == 0:
            _save(base_model, Path(args.output_dir) / f"checkpoint-{step}", optimizer, step)

    if is_main:
        _save(base_model, Path(args.output_dir) / "final", optimizer, step)
        logger.info(f"Training complete. Final loss: {accum['loss']:.4f}")

    dist.destroy_process_group()


def _save(model, save_dir: Path, optimizer: torch.optim.Optimizer, step: int) -> None:
    """Save LoRA adapters, head weights, optimizer state, and step (rank-0 only)."""
    import json
    save_dir.mkdir(parents=True, exist_ok=True)

    model.transformer.save_pretrained(save_dir / "lora_adapter")

    head_state = {
        "known_head.concept_predictor.weight": model.known_head.concept_predictor.weight.data.cpu()
    }
    if model.unknown_head is not None:
        for name, param in model.unknown_head.named_parameters():
            if param.requires_grad:
                head_state[f"unknown_head.{name}"] = param.data.cpu()
    torch.save(head_state, save_dir / "head_weights.pt")

    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    (save_dir / "training_state.json").write_text(json.dumps({"step": step}))

    logging.getLogger(__name__).info(f"Saved checkpoint to {save_dir}")


if __name__ == "__main__":
    main()

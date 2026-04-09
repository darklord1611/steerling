"""SFT trainer for Steerling with LoRA, concept-aware losses, and diffusion masking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from steerling.data.tokenizer import SteerlingTokenizer
from steerling.models.interpretable.interpretable_causal_diffusion import InterpretableCausalDiffusionLM

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj"])

    # Loss weights
    lambda_rec: float = 0.1
    lambda_indep: float = 0.01
    indep_sample_size: int = 512

    # Training
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    warmup_steps: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 500
    dtype: str = "bfloat16"

    # Output
    output_dir: str = "sft_output"

    # Weights & Biases (always enabled)
    wandb_project: str = "steerling_hhh_sft"
    wandb_run_name: str | None = None


def setup_lora(model: InterpretableCausalDiffusionLM, config: SFTConfig) -> nn.Module:
    """Apply LoRA adapters to the transformer backbone and freeze appropriate parameters.

    Frozen: known_head.concept_embedding.weight (preserves concept vocabulary)
    Trainable: LoRA adapters, both projection heads, unknown embeddings
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as err:
        raise ImportError("Install 'peft' package: pip install peft") from err

    # Apply LoRA to transformer backbone
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    # PEFT expects model.config to support .get(); Pydantic BaseModel does not
    if hasattr(model.transformer, "config") and not hasattr(model.transformer.config, "get"):
        from pydantic import BaseModel as PydanticBaseModel

        if isinstance(model.transformer.config, PydanticBaseModel):
            model.transformer.config = model.transformer.config.model_dump()
    model.transformer = get_peft_model(model.transformer, lora_config)

    # Freeze known concept embeddings (preserve human-labeled vocabulary)
    model.known_head.concept_embedding.weight.requires_grad_(False)

    # Ensure known predictor is trainable
    model.known_head.concept_predictor.weight.requires_grad_(True)

    # Ensure unknown head parameters are trainable (predictor + embeddings)
    if model.unknown_head is not None:
        for param in model.unknown_head.parameters():
            param.requires_grad_(True)

    # Log parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    logger.info(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,}")

    return model


def diffusion_mask(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MDLM-style random masking to assistant tokens.

    For each example in the batch, sample a mask rate uniformly, then randomly
    mask that fraction of the assistant tokens (loss_mask == 1).

    Returns:
        noisy_ids: input_ids with some assistant tokens replaced by mask_token_id
        p_mask: per-example mask rate (for 1/p_mask weighting), shape (B, 1)
    """
    B, T = input_ids.shape
    device = input_ids.device
    trainable = loss_mask == 1  # (B, T)

    # Per-example mask rates uniformly in (0, 1]
    rates = torch.rand(B, 1, device=device).clamp_min(1e-3)  # (B, 1)

    # Random scores only at trainable positions; non-trainable get 2.0 (never selected)
    rand_scores = torch.where(trainable, torch.rand(B, T, device=device), torch.full((B, T), 2.0, device=device))

    # Count trainable tokens per example
    n_trainable = trainable.sum(dim=1, keepdim=True).float()  # (B, 1)

    # Number of tokens to mask (at least 1 where there are trainable tokens)
    n_to_mask = (rates * n_trainable).clamp_min(1.0).floor()  # (B, 1)

    # Threshold: mask the n_to_mask lowest-scoring trainable positions per row.
    # Sort scores and pick the n_to_mask-th value as threshold.
    sorted_scores, _ = rand_scores.sort(dim=1)  # (B, T)
    # Gather the threshold at position n_to_mask-1 (0-indexed)
    threshold_idx = (n_to_mask - 1).long().clamp(min=0, max=T - 1)  # (B, 1)
    thresholds = sorted_scores.gather(1, threshold_idx)  # (B, 1)

    to_mask = trainable & (rand_scores <= thresholds)  # (B, T)

    noisy = input_ids.clone()
    noisy[to_mask] = mask_token_id

    # Actual mask rate
    p_mask = to_mask.sum(dim=1, keepdim=True).float() / n_trainable.clamp_min(1.0)
    # Examples with no trainable tokens get p_mask=1
    p_mask = torch.where(n_trainable > 0, p_mask, torch.ones_like(p_mask))

    return noisy, p_mask.clamp_min(1e-6)


def compute_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    noisy_ids: torch.Tensor,
    p_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Compute MDLM token loss: cross-entropy on masked positions, weighted by 1/p_mask.

    Args:
        logits: Model output logits (B, T, V)
        targets: Original input_ids (B, T)
        noisy_ids: Masked input_ids (B, T)
        p_mask: Per-example mask rate (B, 1)
        mask_token_id: The mask token ID
    """
    is_masked = noisy_ids == mask_token_id  # (B, T)

    if not is_masked.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Gather logits and targets at masked positions
    masked_logits = logits[is_masked]  # (N_masked, V)
    masked_targets = targets[is_masked]  # (N_masked,)

    # Per-token cross-entropy
    per_token_loss = F.cross_entropy(masked_logits, masked_targets, reduction="none")

    # Weight by 1/p_mask (expand p_mask to per-token)
    weights = (1.0 / p_mask).expand_as(noisy_ids)[is_masked]

    return (per_token_loss * weights).mean()


def compute_rec_loss(
    hidden: torch.Tensor,
    known_features: torch.Tensor,
    unk_hat: torch.Tensor | None,
) -> torch.Tensor:
    """Compute label-free residual reconstruction loss.

    ℒ_rec = ||unk_hat - (hidden - known_features)||²

    All inputs are detached — gradients flow only to the unknown head.
    """
    if unk_hat is None:
        return torch.tensor(0.0, device=hidden.device, requires_grad=True)

    unk_gt = hidden.detach() - known_features.detach()
    return F.mse_loss(unk_hat, unk_gt)


def compute_indep_loss(
    known_embedding: torch.Tensor,
    unknown_head: nn.Module,
    sample_size: int = 512,
) -> torch.Tensor:
    """Compute independence loss between known and unknown concept embeddings.

    ℒ_indep = ||E_known_sample.T @ E_unknown_sample||²_F

    Sampled to keep memory tractable for large concept counts.
    """
    # Get unknown embeddings (handles both factorized and dense)
    E_u = unknown_head._get_embedding_weight()[: unknown_head.n_concepts]  # (C_u, D)
    E_k = known_embedding  # (C_k, D), frozen

    # Sample random subsets
    n_k = min(sample_size, E_k.shape[0])
    n_u = min(sample_size, E_u.shape[0])
    idx_k = torch.randperm(E_k.shape[0], device=E_k.device)[:n_k]
    idx_u = torch.randperm(E_u.shape[0], device=E_u.device)[:n_u]

    # Normalize for numerical stability
    E_k_sample = F.normalize(E_k[idx_k], dim=-1)
    E_u_sample = F.normalize(E_u[idx_u], dim=-1)

    cross_cov = E_k_sample @ E_u_sample.T  # (n_k, n_u) — pairwise cosine similarity
    return cross_cov.pow(2).sum() / (n_k * n_u)


def load_checkpoint(
    model: InterpretableCausalDiffusionLM,
    checkpoint: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load LoRA adapter, head weights, and optionally optimizer state from a checkpoint.

    Args:
        model: Model with LoRA already applied via setup_lora().
        checkpoint: Local path (e.g. "sft_output/final") or HuggingFace repo ID
                    (e.g. "darklord1611/steerling-8b-sft-tulu3").
        optimizer: If provided, optimizer state is restored for clean resume.

    Returns:
        Resumed step count (0 if not found in checkpoint).
    """
    import json
    from pathlib import Path as _Path

    is_local = _Path(checkpoint).exists()

    if is_local:
        lora_dir = str(_Path(checkpoint) / "lora_adapter")
        head_path = _Path(checkpoint) / "head_weights.pt"
        opt_path = _Path(checkpoint) / "optimizer.pt"
        state_path = _Path(checkpoint) / "training_state.json"
    else:
        # HuggingFace repo — download files on demand
        from huggingface_hub import hf_hub_download, snapshot_download
        lora_dir = snapshot_download(checkpoint, allow_patterns=["lora_adapter/*"])
        lora_dir = str(_Path(lora_dir) / "lora_adapter")
        head_path = _Path(hf_hub_download(checkpoint, "head_weights.pt"))
        try:
            opt_path = _Path(hf_hub_download(checkpoint, "optimizer.pt"))
        except Exception:
            opt_path = None
        try:
            state_path = _Path(hf_hub_download(checkpoint, "training_state.json"))
        except Exception:
            state_path = None

    # Load LoRA adapter weights.
    # If setup_lora() has already wrapped the transformer, use load_adapter to avoid
    # double-wrapping (which would corrupt the trainable parameter count).
    from peft import PeftModel
    if isinstance(model.transformer, PeftModel):
        model.transformer.load_adapter(lora_dir, adapter_name="default", set_as_active=True)
    else:
        model.transformer = PeftModel.from_pretrained(model.transformer, lora_dir)
    logger.info(f"Loaded LoRA adapter from {lora_dir}")

    # Load head weights
    head_state = torch.load(head_path, map_location="cpu")
    for key, value in head_state.items():
        parts = key.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        getattr(obj, parts[-1]).data.copy_(value.to(next(model.parameters()).device))
    logger.info(f"Loaded head weights from {head_path}")

    # Load optimizer state
    if optimizer is not None and opt_path is not None and _Path(opt_path).exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        logger.info("Loaded optimizer state")
    elif optimizer is not None:
        logger.warning("No optimizer.pt found — Adam state will reset (expect ~200 unstable steps)")

    # Return step count
    if state_path is not None and _Path(state_path).exists():
        step = json.loads(_Path(state_path).read_text()).get("step", 0)
        logger.info(f"Resuming from step {step}")
        return step

    return 0


class SFTTrainer:
    """SFT trainer for Steerling with LoRA and concept-aware losses.

    Implements the three-loss SFT objective:
    - ℒ_token: MDLM diffusion masked LM loss on assistant tokens
    - ℒ_rec: label-free residual reconstruction for the unknown head
    - ℒ_indep: cross-covariance penalty between known/unknown embeddings

    Args:
        model: InterpretableCausalDiffusionLM (will be modified in-place with LoRA)
        tokenizer: SteerlingTokenizer
        config: SFTConfig
    """

    def __init__(
        self,
        model: InterpretableCausalDiffusionLM,
        tokenizer: SteerlingTokenizer,
        config: SFTConfig,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = getattr(torch, config.dtype)

        # Setup LoRA and freeze parameters
        self.model = setup_lora(model, config)

        # Weights & Biases (always on)
        self._wandb = None
        try:
            import wandb
            self._wandb = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    "lora_r": config.lora_r,
                    "lora_alpha": config.lora_alpha,
                    "lora_dropout": config.lora_dropout,
                    "lambda_rec": config.lambda_rec,
                    "lambda_indep": config.lambda_indep,
                    "lr": config.lr,
                    "batch_size": config.batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_steps": config.max_steps,
                    "warmup_steps": config.warmup_steps,
                },
            )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def _lr_schedule(self, step: int) -> float:
        """Linear warmup then cosine decay."""
        if step < self.config.warmup_steps:
            return step / max(1, self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dict with "input_ids" (B, T) and "loss_mask" (B, T)

        Returns:
            Dict of loss values for logging.
        """
        input_ids = batch["input_ids"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        # Apply diffusion masking to assistant tokens
        noisy_ids, p_mask = diffusion_mask(input_ids, loss_mask, self.tokenizer.mask_token_id)

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits, outputs = self.model(noisy_ids, minimal_output=True)

            # ℒ_token
            l_token = compute_token_loss(logits, input_ids, noisy_ids, p_mask, self.tokenizer.mask_token_id)

            # ℒ_rec
            l_rec = compute_rec_loss(outputs.hidden, outputs.known_features, outputs.unk_hat)

            # ℒ_indep
            l_indep = compute_indep_loss(
                self.model.known_head.concept_embedding.weight,
                self.model.unknown_head,
                sample_size=self.config.indep_sample_size,
            )

            loss = l_token + self.config.lambda_rec * l_rec + self.config.lambda_indep * l_indep

        # Scale for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return {
            "loss": loss.item(),
            "l_token": l_token.item(),
            "l_rec": l_rec.item(),
            "l_indep": l_indep.item(),
        }

    def train(self, dataloader: DataLoader) -> list[dict[str, float]]:
        """Run the full training loop.

        Args:
            dataloader: DataLoader yielding batches with "input_ids" and "loss_mask".

        Returns:
            List of per-step loss dicts.
        """
        self.model.train()
        history: list[dict[str, float]] = []
        data_iter = iter(dataloader)
        step = 0

        while step < self.config.max_steps:
            self.optimizer.zero_grad()

            accum_losses: dict[str, float] = {}
            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                step_losses = self.train_step(batch)
                for k, v in step_losses.items():
                    accum_losses[k] = accum_losses.get(k, 0.0) + v / self.config.gradient_accumulation_steps

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )

            # LR schedule
            lr_mult = self._lr_schedule(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.lr * lr_mult

            self.optimizer.step()
            step += 1
            history.append(accum_losses)

            if step % self.config.log_every == 0:
                current_lr = self.config.lr * lr_mult
                logger.info(
                    f"step {step}/{self.config.max_steps} | "
                    f"loss={accum_losses['loss']:.4f} | "
                    f"l_token={accum_losses['l_token']:.4f} | "
                    f"l_rec={accum_losses['l_rec']:.4f} | "
                    f"l_indep={accum_losses['l_indep']:.6f} | "
                    f"lr={current_lr:.2e}"
                )
                if self._wandb is not None:
                    self._wandb.log({**accum_losses, "lr": current_lr}, step=step)

            if step % self.config.save_every == 0:
                save_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
                self.save(save_dir, step=step)

        # Final save
        save_dir = Path(self.config.output_dir) / "final"
        self.save(save_dir, step=step)

        return history

    def save(self, save_dir: str | Path, step: int = 0) -> None:
        """Save LoRA adapters, trainable head parameters, and optimizer state."""
        import json
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # LoRA adapters
        self.model.transformer.save_pretrained(save_dir / "lora_adapter")

        # Trainable head parameters
        head_state = {
            "known_head.concept_predictor.weight": (
                self.model.known_head.concept_predictor.weight.data.cpu()
            )
        }
        if self.model.unknown_head is not None:
            for name, param in self.model.unknown_head.named_parameters():
                if param.requires_grad:
                    head_state[f"unknown_head.{name}"] = param.data.cpu()
        torch.save(head_state, save_dir / "head_weights.pt")

        # Optimizer state (enables proper resume without Adam momentum reset)
        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")

        # Training state (step count + LR for schedule continuity)
        (save_dir / "training_state.json").write_text(
            json.dumps({"step": step, "lr": self.config.lr})
        )

        logger.info(f"Saved checkpoint to {save_dir}")

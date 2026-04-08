#!/usr/bin/env python3
"""Push trained SFT LoRA adapter and head weights to HuggingFace.

Usage:
    python scripts/push_sft_model.py \
        --checkpoint sft_output/final \
        --repo YOUR_USERNAME/steerling-8b-sft-tulu3

    # Push a specific checkpoint:
    python scripts/push_sft_model.py \
        --checkpoint sft_output/checkpoint-2000 \
        --repo YOUR_USERNAME/steerling-8b-sft-tulu3-ckpt2000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push SFT checkpoint to HuggingFace")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint directory (e.g. sft_output/final)",
    )
    parser.add_argument(
        "--repo", type=str, required=True,
        help="HuggingFace repo ID (e.g. username/steerling-8b-sft-tulu3)",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--base-model", type=str, default="guidelabs/steerling-8b",
        help="Base model this adapter was trained on",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, upload_folder

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    lora_dir = checkpoint_dir / "lora_adapter"
    head_weights = checkpoint_dir / "head_weights.pt"

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_dir}")
    if not head_weights.exists():
        raise FileNotFoundError(f"Head weights not found: {head_weights}")

    # Write a model card
    card = f"""---
tags:
  - steerling
  - lora
  - sft
  - tulu-3
base_model: {args.base_model}
library_name: peft
---

# Steerling 8B — SFT LoRA Adapter (Tulu-3 English)

LoRA adapter trained on [darklord1611/tulu-3-sft-mixture-english-clean](https://huggingface.co/datasets/darklord1611/tulu-3-sft-mixture-english-clean) using concept-aware SFT losses (L_token + L_rec + L_indep).

## Files

- `lora_adapter/` — PEFT LoRA weights for the transformer backbone
- `head_weights.pt` — trained concept predictor + unknown head weights

## Usage

```python
from steerling.inference.causal_diffusion import SteerlingGenerator
from peft import PeftModel
import torch

# Load base model
generator = SteerlingGenerator.from_pretrained("{args.base_model}", device="cuda")
model = generator.model

# Load LoRA adapter
model.transformer = PeftModel.from_pretrained(model.transformer, "lora_adapter_path")

# Load head weights
head_state = torch.load("head_weights.pt", map_location="cuda")
for key, value in head_state.items():
    parts = key.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data.copy_(value)
```

## Training Details

- **Base model**: [{args.base_model}](https://huggingface.co/{args.base_model})
- **Dataset**: [darklord1611/tulu-3-sft-mixture-english-clean](https://huggingface.co/datasets/darklord1611/tulu-3-sft-mixture-english-clean)
- **Method**: LoRA (r=16, alpha=32) on `c_attn` and `c_proj`
- **Losses**: MDLM token loss + residual reconstruction + independence penalty
"""

    readme_path = checkpoint_dir / "README.md"
    readme_path.write_text(card)
    logger.info(f"Wrote model card to {readme_path}")

    # Create repo and upload
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, private=args.private, repo_type="model")

    logger.info(f"Uploading {checkpoint_dir} to {args.repo}...")
    upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"SFT checkpoint from {checkpoint_dir.name}",
    )
    logger.info(f"Done. Model at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

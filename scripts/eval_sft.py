#!/usr/bin/env python3
"""
Evaluate a fine-tuned Steerling model (base + LoRA + head weights).

Usage:
    python scripts/eval_sft.py --base-model guidelabs/steerling-8b \
        --checkpoint sft_output_lima/final --tasks ifeval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from lm_eval import evaluator

from steerling.configs.evaluation import get_task_settings
from steerling.evaluation.lm_harness_wrapper import SteerlingLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_sft_model(
    base_model: str,
    checkpoint_dir: str,
    device: str = "cuda",
    **lm_kwargs,
) -> SteerlingLM:
    """Load base model, apply LoRA adapter and SFT head weights."""

    # Load base model via standard harness
    lm = SteerlingLM(model_path=base_model, device=device, **lm_kwargs)

    ckpt = Path(checkpoint_dir)
    lora_path = ckpt / "lora_adapter"
    head_path = ckpt / "head_weights.pt"

    # Resize embeddings if needed
    tokenizer = lm.tokenizer
    model = lm.model
    if model.transformer.tok_emb.weight.shape[0] < tokenizer.vocab_size:
        old_size = model.transformer.tok_emb.weight.shape[0]
        new_size = tokenizer.vocab_size
        logger.info(f"Resizing embeddings: {old_size} -> {new_size}")

        old_emb = model.transformer.tok_emb.weight.data
        new_emb = torch.zeros(new_size, old_emb.shape[1], dtype=old_emb.dtype, device=old_emb.device)
        new_emb[:old_size] = old_emb
        new_emb[old_size:] = old_emb.mean(dim=0, keepdim=True)

        model.transformer.tok_emb = torch.nn.Embedding(new_size, old_emb.shape[1]).to(
            device=old_emb.device, dtype=old_emb.dtype
        )
        model.transformer.tok_emb.weight.data = new_emb
        model.transformer.lm_head = torch.nn.Linear(old_emb.shape[1], new_size, bias=False).to(
            device=old_emb.device, dtype=old_emb.dtype
        )
        model.transformer.lm_head.weight = model.transformer.tok_emb.weight

    # Apply LoRA
    if lora_path.exists():
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter from {lora_path}")
        model.transformer = PeftModel.from_pretrained(model.transformer, str(lora_path))
        model.transformer.eval()

    # Load head weights
    if head_path.exists():
        logger.info(f"Loading head weights from {head_path}")
        head_state = torch.load(head_path, map_location=device, weights_only=True)

        for key, value in head_state.items():
            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param = getattr(obj, parts[-1])
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(value.to(param.dtype))
            else:
                setattr(obj, parts[-1], value.to(param.dtype))

        logger.info(f"Loaded {len(head_state)} head weight tensors")

    model.eval()
    return lm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Steerling model")
    parser.add_argument("--base-model", type=str, default="guidelabs/steerling-8b")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint dir")
    parser.add_argument("--tasks", nargs="+", default=["ifeval"])
    parser.add_argument("--results-dir", type=str, default="eval_results_sft")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for task_name in args.tasks:
        settings = get_task_settings(task_name)
        logger.info(f"Evaluating task: {task_name}")

        lm = load_sft_model(
            args.base_model,
            args.checkpoint,
            device=args.device,
            batch_size=settings.batch_size,
            mc_num=settings.mc_num,
            mc_batch_size=settings.mc_batch_size,
            cfg=settings.cfg,
            gen_length=settings.gen_length,
            steps=settings.steps,
        )

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=settings.num_fewshot,
            batch_size=settings.batch_size,
            device=args.device,
        )

        all_results[task_name] = results

        task_file = results_dir / f"{task_name}.json"
        with open(task_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        del lm
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("SFT EVALUATION RESULTS")
    print("=" * 60)
    for task_name, task_results in all_results.items():
        task_metrics = task_results.get("results", {}).get(task_name, {})
        if task_metrics:
            print(f"\n{task_name}:")
            for metric, value in task_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate IFBench responses using a Steerling model (base or SFT).

Reads IFBench_test.jsonl, generates one response per prompt, and writes
a JSONL file with {"prompt": ..., "response": ...} records that IFBench's
run_eval.py expects.

Usage:
    # Base model
    python scripts/generate_ifbench.py \
        --output-file ifbench_results/steerling-base-responses.jsonl

    # SFT model
    python scripts/generate_ifbench.py \
        --checkpoint sft_output/final \
        --output-file ifbench_results/steerling-sft-responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from steerling.configs.generation import GenerationConfig
from steerling.inference.causal_diffusion import SteerlingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IFBENCH_DIR = Path(__file__).parent.parent / "IFBench"
DEFAULT_DATA = IFBENCH_DIR / "data" / "IFBench_test.jsonl"


def load_model(
    base_model: str,
    checkpoint: str | None,
    device: str,
) -> SteerlingGenerator:
    """Load base or SFT model."""
    gen = SteerlingGenerator.from_pretrained(base_model, device=device, dtype=torch.bfloat16)

    if checkpoint is None:
        return gen

    ckpt = Path(checkpoint)
    lora_path = ckpt / "lora_adapter"
    head_path = ckpt / "head_weights.pt"
    model = gen.model
    tokenizer = gen.tokenizer

    # Resize embeddings if needed (SFT adds ChatML tokens)
    if model.transformer.tok_emb.weight.shape[0] < tokenizer.vocab_size:
        old_size = model.transformer.tok_emb.weight.shape[0]
        new_size = tokenizer.vocab_size
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
        logger.info(f"Resized embeddings: {old_size} -> {new_size}")

    if lora_path.exists():
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {lora_path}")
        model.transformer = PeftModel.from_pretrained(model.transformer, str(lora_path))
        model.transformer.eval()

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
        logger.info(f"Loaded {len(head_state)} head weight tensors")

    model.eval()
    return gen


def generate_responses(
    gen: SteerlingGenerator,
    prompts: list[dict],
    gen_length: int,
    steps: int,
    temperature: float,
    output_file: Path,
) -> None:
    """Generate responses and write incrementally to output_file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed prompts for resume support
    done: dict[str, str] = {}
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["prompt"]] = rec["response"]
        logger.info(f"Resuming: {len(done)} responses already written")

    remaining = [p for p in prompts if p["prompt"] not in done]
    logger.info(f"Generating {len(remaining)} responses (skipping {len(done)} done)")

    # Round gen_length up to a block boundary
    block_size = gen.diff_block_size
    padded_len = ((gen_length + block_size - 1) // block_size) * block_size

    gen_config = GenerationConfig(
        max_new_tokens=padded_len,
        steps=steps,
        temperature=temperature,
        cfg_scale=0.0,
    )

    with open(output_file, "a") as out_f:
        # Write already-done entries first if resuming (re-open in write mode for fresh file)
        if not done and output_file.stat().st_size == 0:
            pass  # fresh file, we'll append below

        for item in tqdm(remaining, desc="Generating"):
            prompt_text = item["prompt"]
            tokens = gen.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=gen.device)

            with torch.inference_mode():
                output = gen.generate_full(prompt_tensor, gen_config)

            response = output.text.rstrip()
            rec = {"prompt": prompt_text, "response": response}
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    logger.info(f"Saved responses to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IFBench responses with Steerling")
    parser.add_argument("--base-model", default="guidelabs/steerling-8b")
    parser.add_argument("--checkpoint", default=None, help="SFT checkpoint dir (LoRA + head weights)")
    parser.add_argument(
        "--input-file", default=str(DEFAULT_DATA), help="Path to IFBench_test.jsonl"
    )
    parser.add_argument("--output-file", required=True, help="Output JSONL path for responses")
    parser.add_argument("--gen-length", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument(
        "--steps", type=int, default=64,
        help="Diffusion unmasking steps (higher = better quality, more compute)"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logger.info(f"Base model: {args.base_model}")
    if args.checkpoint:
        logger.info(f"SFT checkpoint: {args.checkpoint}")

    gen = load_model(args.base_model, args.checkpoint, args.device)

    prompts = []
    with open(args.input_file) as f:
        for line in f:
            example = json.loads(line)
            prompts.append({"key": example["key"], "prompt": example["prompt"]})
    logger.info(f"Loaded {len(prompts)} prompts from {args.input_file}")

    generate_responses(
        gen=gen,
        prompts=prompts,
        gen_length=args.gen_length,
        steps=args.steps,
        temperature=args.temperature,
        output_file=Path(args.output_file),
    )


if __name__ == "__main__":
    main()

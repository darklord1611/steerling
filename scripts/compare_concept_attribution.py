#!/usr/bin/env python3
"""
Compare concept attribution between the base Steerling model and an SFT checkpoint.

For a set of prompts, this script:
  1. Loads both the base model and the SFT model (base + LoRA + head weights).
  2. Generates text for each prompt with both models.
  3. Computes chunk-level concept attribution for all chunks.
  4. Averages attribution across all prompts.
  5. Plots the top concepts side-by-side for base vs SFT.

Usage:
    python scripts/compare_concept_attribution.py \
        --base-model guidelabs/steerling-8b \
        --checkpoint sft_output/checkpoint-5500 \
        --output-dir concept_attribution_comparison
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from transformers import AutoModel, AutoTokenizer

from steerling import GenerationConfig, SteerlingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Colour scheme ──────────────────────────────────────────────────────────────
PURPLE = "#675BF2"
PURPLE_LIGHT = "#ceb4fe"
COLOR_MAP = {"known": PURPLE, "discovered": PURPLE_LIGHT}

# ── Prompts ────────────────────────────────────────────────────────────────────
DEFAULT_PROMPTS = [
    # Society & culture
    "The impact of social media on mental health has been",
    "Economic inequality in modern societies stems from",
    "The role of education in reducing poverty is",
    "Immigration policy should balance national security with",
    "The decline of trust in mainstream media is partly due to",
    "Urban planning affects quality of life because",
    "The gig economy has changed the nature of work by",
    "Religious institutions play a role in communities by",
    "Cancel culture has changed public discourse because",
    "The aging population will affect healthcare systems by",
    # Science & technology
    "Climate change is one of the most pressing issues because",
    "Advances in artificial intelligence will transform",
    "Space exploration is important for humanity because",
    "CRISPR gene editing raises ethical questions about",
    "Quantum computing will eventually impact cryptography by",
    "The development of nuclear fusion energy could",
    "Renewable energy adoption faces challenges such as",
    "Vaccines have historically been effective at",
    "The James Webb Space Telescope has revealed that",
    "Ocean acidification threatens marine ecosystems because",
    # Health & lifestyle
    "A healthy diet and regular exercise contribute to",
    "Sleep deprivation has serious consequences including",
    "Mental health awareness has increased in recent years due to",
    "The opioid crisis in America was caused by",
    "Meditation and mindfulness practices have been shown to",
    "Processed food consumption has risen dramatically because",
    "Antibiotic resistance is a growing threat because",
    "Early childhood development is shaped by",
    "The connection between gut health and mental health suggests",
    "Chronic stress affects the body by",
    # Learning & cognition
    "The best way to learn a new programming language is to",
    "Critical thinking skills are essential in the modern world because",
    "Memory consolidation during sleep occurs when",
    "Bilingualism has cognitive benefits such as",
    "Reading fiction develops empathy because",
    "The Socratic method improves learning by",
    "Motivation and habit formation are closely linked through",
    "Deep work requires minimizing distractions because",
    "Standardized testing fails to measure intelligence because",
    "Self-directed learning is becoming more important as",
    # Economics & policy
    "Central bank interest rate decisions affect inflation by",
    "Universal basic income has been proposed as a solution to",
    "Supply chain disruptions during the pandemic revealed that",
    "The housing affordability crisis in major cities stems from",
    "Free trade agreements benefit economies by",
    "Cryptocurrency adoption faces regulatory hurdles because",
    "The national debt becomes a concern when",
    "Labor unions have historically protected workers by",
    "Foreign aid programs are most effective when",
    "Corporate monopolies harm consumers by",
]


# ── Concept label helpers ──────────────────────────────────────────────────────

def load_concept_labels(repo_root: Path) -> pd.DataFrame:
    concepts_path = repo_root / "assets" / "concepts" / "known_concepts.csv"
    if concepts_path.exists():
        df = pd.read_csv(concepts_path)
        logger.info(f"Loaded {len(df)} concept labels from {concepts_path}")
        return df
    logger.warning("No known_concepts.csv found — using numeric fallback labels")
    return pd.DataFrame(columns=["concept_idx", "concept_name"])


def concept_label(concept_id: int, concepts_df: pd.DataFrame, concept_type: str = "known") -> str:
    if concept_type == "discovered":
        return f"Discovered #{concept_id}"
    row = concepts_df[concepts_df["concept_idx"] == concept_id]
    if len(row) == 0:
        return f"Known #{concept_id}"
    return row.iloc[0]["concept_name"]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_base_model(model_id: str, device: str) -> tuple[SteerlingGenerator, object]:
    logger.info(f"Loading base model: {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    generator = SteerlingGenerator.from_model(model, tokenizer, device=device)
    backbone = generator.model.model if hasattr(generator.model, "model") else generator.model
    return generator, backbone


def load_sft_model(base_model_id: str, checkpoint_dir: str, device: str) -> tuple[SteerlingGenerator, object]:
    logger.info(f"Loading SFT model from {checkpoint_dir}")
    model = AutoModel.from_pretrained(base_model_id, trust_remote_code=True, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    ckpt = Path(checkpoint_dir)
    lora_path = ckpt / "lora_adapter"
    head_path = ckpt / "head_weights.pt"

    # Resize embeddings if vocab sizes differ
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

    if lora_path.exists():
        from peft import PeftModel
        logger.info(f"Applying LoRA adapter from {lora_path}")
        model.transformer = PeftModel.from_pretrained(model.transformer, str(lora_path))
        model.transformer.eval()

    if head_path.exists():
        logger.info(f"Loading head weights from {head_path}")
        head_state = torch.load(head_path, map_location="cpu", weights_only=True)
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
    generator = SteerlingGenerator.from_model(model, tokenizer, device=device)
    backbone = generator.model.model if hasattr(generator.model, "model") else generator.model
    return generator, backbone


# ── Attribution computation ────────────────────────────────────────────────────

@torch.no_grad()
def compute_concept_attribution(outputs, logits, backbone, device: str):
    pred_ids = logits.argmax(dim=-1)
    W_y = backbone.transformer.lm_head.weight[pred_ids]

    def _head_contrib(head, topk_idx, topk_logits):
        if topk_idx is None:
            return None
        emb = head._get_embedding(topk_idx)
        dots = torch.einsum("btkd,btd->btk", emb, W_y)
        w = torch.sigmoid(topk_logits.float())
        return topk_idx, w * dots

    known = _head_contrib(
        backbone.known_head,
        topk_idx=outputs.known_topk_indices,
        topk_logits=outputs.known_topk_logits,
    )

    disc = None
    if getattr(backbone, "unknown_head", None) is not None:
        disc = _head_contrib(
            backbone.unknown_head,
            topk_idx=outputs.unknown_topk_indices,
            topk_logits=outputs.unknown_topk_logits,
        )

    bsz, tlen = pred_ids.shape
    zero = lambda: (
        torch.zeros(bsz, tlen, 0, dtype=torch.long, device=device),
        torch.zeros(bsz, tlen, 0, device=device),
    )

    k_idx, k_c = known if known else zero()
    d_idx, d_c = disc if disc else zero()

    pred_logits = torch.gather(logits, -1, pred_ids.unsqueeze(-1)).squeeze(-1)
    eps = pred_logits - k_c.sum(-1) - d_c.sum(-1)

    return {
        "known_indices": k_idx,
        "known_contributions": k_c,
        "disc_indices": d_idx,
        "disc_contributions": d_c,
        "epsilon": eps,
    }


def find_chunks(token_ids, endofchunk_token_id):
    ids = token_ids.squeeze()
    if endofchunk_token_id is None:
        return [(0, len(ids))]
    positions = (ids == endofchunk_token_id).nonzero(as_tuple=True)[0].tolist()
    chunks, prev = [], 0
    for p in positions:
        if p > prev:
            chunks.append((prev, p))
        prev = p + 1
    if prev < len(ids):
        chunks.append((prev, len(ids)))
    return chunks


def chunk_attribution(attr, start: int, end: int, concepts_df: pd.DataFrame, device: str, batch: int = 0):
    k_idx = attr["known_indices"][batch, start:end]
    k_c = attr["known_contributions"][batch, start:end]
    d_idx = attr["disc_indices"][batch, start:end]
    d_c = attr["disc_contributions"][batch, start:end]
    eps = attr["epsilon"][batch, start:end]

    pos_total = (k_c.abs().sum(-1) + d_c.abs().sum(-1) + eps.abs()).clamp(min=1e-8)

    k_c_norm = k_c / pos_total.unsqueeze(-1)
    d_c_norm = d_c / pos_total.unsqueeze(-1)

    def aggregate(idx, contrib_norm):
        flat_idx = idx.reshape(-1)
        flat_norm = contrib_norm.reshape(-1)
        unique_ids, inverse = flat_idx.unique(return_inverse=True)
        mass = torch.zeros(len(unique_ids), device=flat_idx.device)
        mass.scatter_add_(0, inverse, flat_norm)
        return unique_ids, mass

    k_ids, k_mass = aggregate(k_idx, k_c_norm)
    d_ids, d_mass = aggregate(d_idx, d_c_norm)

    T = end - start
    entries = []
    for cid, mass in zip(k_ids.tolist(), k_mass.tolist()):
        entries.append({
            "label": concept_label(int(cid), concepts_df, "known"),
            "type": "known",
            "contribution": mass / T,
        })
    for cid, mass in zip(d_ids.tolist(), d_mass.tolist()):
        entries.append({
            "label": f"Discovered #{int(cid)}",
            "type": "discovered",
            "contribution": mass / T,
        })

    eps_pct = float((eps / pos_total).mean()) * 100
    return entries, eps_pct


# ── Per-prompt attribution ─────────────────────────────────────────────────────

def attribution_for_prompt(
    prompt: str,
    generator: SteerlingGenerator,
    backbone,
    gen_config: GenerationConfig,
    concepts_df: pd.DataFrame,
    device: str,
) -> tuple[dict[str, float], dict[str, str], float]:
    """
    Return (label -> summed_contribution, label -> type, mean_eps_pct) for one prompt,
    averaged across all chunks of the generated output.
    """
    eoc_id = getattr(generator.tokenizer, "endofchunk_token_id", None)

    gen_output = generator.generate_full(prompt, gen_config)
    output = gen_output.tokens.unsqueeze(0)

    with torch.no_grad():
        logits, interp_output = backbone(output, minimal_output=True)

    attr = compute_concept_attribution(interp_output, logits, backbone, device)
    chunks = find_chunks(output, eoc_id)

    label_contribution: dict[str, float] = defaultdict(float)
    label_type: dict[str, str] = {}
    eps_total = 0.0

    for s, e in chunks:
        if e <= s:
            continue
        entries, eps_pct = chunk_attribution(attr, s, e, concepts_df, device)
        for entry in entries:
            label_contribution[entry["label"]] += entry["contribution"]
            label_type[entry["label"]] = entry["type"]
        eps_total += eps_pct

    # Average across chunks
    n_chunks = len(chunks)
    label_contribution = {k: v / n_chunks for k, v in label_contribution.items()}
    eps_mean = eps_total / n_chunks if n_chunks > 0 else 0.0

    return label_contribution, label_type, eps_mean


# ── Aggregation across prompts ─────────────────────────────────────────────────

def aggregate_across_prompts(
    per_prompt: list[tuple[dict[str, float], dict[str, str], float]],
) -> tuple[dict[str, float], dict[str, str], float]:
    """Average contributions across all prompts."""
    combined: dict[str, float] = defaultdict(float)
    label_type: dict[str, str] = {}
    eps_total = 0.0

    for contributions, types, eps in per_prompt:
        for label, val in contributions.items():
            combined[label] += val
            label_type[label] = types[label]
        eps_total += eps

    n = len(per_prompt)
    combined = {k: v / n for k, v in combined.items()}
    eps_mean = eps_total / n
    return combined, label_type, eps_mean


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(
    base_contributions: dict[str, float],
    base_label_type: dict[str, str],
    base_eps: float,
    sft_contributions: dict[str, float],
    sft_label_type: dict[str, str],
    sft_eps: float,
    top_k: int = 30,
    output_path: Path | None = None,
):
    """Side-by-side bar charts for base vs SFT concept attribution."""
    all_labels = set(base_contributions) | set(sft_contributions)
    ranked = sorted(all_labels, key=lambda l: max(base_contributions.get(l, 0), sft_contributions.get(l, 0)), reverse=True)
    top_labels = ranked[:top_k]

    base_vals = [base_contributions.get(l, 0.0) for l in top_labels]
    sft_vals = [sft_contributions.get(l, 0.0) for l in top_labels]
    label_types = [base_label_type.get(l, sft_label_type.get(l, "known")) for l in top_labels]
    base_colors = [COLOR_MAP[t] for t in label_types]
    sft_colors = [COLOR_MAP[t] for t in label_types]

    fig, (ax_base, ax_sft) = plt.subplots(1, 2, figsize=(18, max(6, len(top_labels) * 0.38)), sharey=True)

    y = np.arange(len(top_labels))

    ax_base.barh(y, base_vals, color=base_colors)
    ax_base.set_yticks(y)
    ax_base.set_yticklabels(top_labels, fontsize=8)
    ax_base.invert_yaxis()
    ax_base.set_xlabel("Avg fraction of total |logit| mass")
    ax_base.set_title(f"Base model  |  epsilon = {base_eps:.1f}%", fontsize=11)
    ax_base.spines["top"].set_visible(False)
    ax_base.spines["right"].set_visible(False)

    ax_sft.barh(y, sft_vals, color=sft_colors)
    ax_sft.invert_yaxis()
    ax_sft.set_xlabel("Avg fraction of total |logit| mass")
    ax_sft.set_title(f"SFT checkpoint  |  epsilon = {sft_eps:.1f}%", fontsize=11)
    ax_sft.spines["top"].set_visible(False)
    ax_sft.spines["right"].set_visible(False)

    legend_handles = [Patch(fc=COLOR_MAP[t], label=t.title()) for t in ["known", "discovered"]]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle(f"Concept Attribution: Base vs SFT (top {top_k}, averaged over {len(DEFAULT_PROMPTS)} prompts)", fontsize=13, y=1.01)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_path}")
    plt.show()


def plot_delta(
    base_contributions: dict[str, float],
    sft_contributions: dict[str, float],
    base_label_type: dict[str, str],
    sft_label_type: dict[str, str],
    top_k: int = 30,
    output_path: Path | None = None,
):
    """Bar chart of SFT - base delta, showing concepts that changed the most."""
    all_labels = set(base_contributions) | set(sft_contributions)
    deltas = {l: sft_contributions.get(l, 0.0) - base_contributions.get(l, 0.0) for l in all_labels}
    ranked = sorted(deltas, key=lambda l: abs(deltas[l]), reverse=True)[:top_k]

    labels = ranked
    values = [deltas[l] for l in labels]
    label_types = [base_label_type.get(l, sft_label_type.get(l, "known")) for l in labels]
    colors = [
        (PURPLE if v >= 0 else "#e25c26") if t == "known" else (PURPLE_LIGHT if v >= 0 else "#f5b7a0")
        for v, t in zip(values, label_types)
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.38)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta (SFT - Base) avg fraction of |logit| mass")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(fc=PURPLE, label="Known (increased)"),
        Patch(fc="#e25c26", label="Known (decreased)"),
        Patch(fc=PURPLE_LIGHT, label="Discovered (increased)"),
        Patch(fc="#f5b7a0", label="Discovered (decreased)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_title(f"Concept Attribution Shift: SFT - Base (top {top_k} by |delta|)", fontsize=11)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved delta plot to {output_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare concept attribution: base vs SFT")
    parser.add_argument("--base-model", default="guidelabs/steerling-8b")
    parser.add_argument("--checkpoint", default="sft_output/checkpoint-5500",
                        help="Path to SFT checkpoint directory")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=30, help="Top-k concepts to display in plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    concepts_df = load_concept_labels(repo_root)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
    )

    # ── Base model ────────────────────────────────────────────────────────────
    logger.info("=== Running attribution for BASE model ===")
    base_generator, base_backbone = load_base_model(args.base_model, args.device)
    base_per_prompt = []
    for i, prompt in enumerate(DEFAULT_PROMPTS):
        logger.info(f"[base] prompt {i+1}/{len(DEFAULT_PROMPTS)}: {prompt!r}")
        result = attribution_for_prompt(prompt, base_generator, base_backbone, gen_config, concepts_df, args.device)
        base_per_prompt.append(result)

    base_contributions, base_label_type, base_eps = aggregate_across_prompts(base_per_prompt)

    # Free base model memory before loading SFT
    del base_generator, base_backbone
    torch.cuda.empty_cache()

    # ── SFT model ─────────────────────────────────────────────────────────────
    logger.info("=== Running attribution for SFT model ===")
    sft_generator, sft_backbone = load_sft_model(args.base_model, args.checkpoint, args.device)
    sft_per_prompt = []
    for i, prompt in enumerate(DEFAULT_PROMPTS):
        logger.info(f"[sft] prompt {i+1}/{len(DEFAULT_PROMPTS)}: {prompt!r}")
        result = attribution_for_prompt(prompt, sft_generator, sft_backbone, gen_config, concepts_df, args.device)
        sft_per_prompt.append(result)

    sft_contributions, sft_label_type, sft_eps = aggregate_across_prompts(sft_per_prompt)

    del sft_generator, sft_backbone
    torch.cuda.empty_cache()

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n=== Top 20 concepts by average contribution ===")
    all_labels = set(base_contributions) | set(sft_contributions)
    ranked = sorted(all_labels, key=lambda l: max(base_contributions.get(l, 0), sft_contributions.get(l, 0)), reverse=True)
    print(f"\n{'Concept':<50} {'Base':>10} {'SFT':>10} {'Delta':>10}")
    print("-" * 82)
    for label in ranked[:20]:
        b = base_contributions.get(label, 0.0)
        s = sft_contributions.get(label, 0.0)
        print(f"{label:<50} {b:10.4f} {s:10.4f} {s-b:+10.4f}")
    print(f"\nBase epsilon: {base_eps:.1f}%  |  SFT epsilon: {sft_eps:.1f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_comparison(
        base_contributions, base_label_type, base_eps,
        sft_contributions, sft_label_type, sft_eps,
        top_k=args.top_k,
        output_path=output_dir / "comparison_side_by_side.png",
    )

    plot_delta(
        base_contributions, sft_contributions,
        base_label_type, sft_label_type,
        top_k=args.top_k,
        output_path=output_dir / "delta_plot.png",
    )

    logger.info(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()

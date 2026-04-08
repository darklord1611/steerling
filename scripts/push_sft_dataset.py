#!/usr/bin/env python3
"""Build and push the cleaned English Tulu-3 SFT mixture to HuggingFace.

Starts from toilaluan/tulu-3-sft-mixture-english and applies two additional
cleaning passes:
  1. Strict both-message ASCII ratio filter  — removes ~23 non-EN leaks
  2. FLAN v2 artificial format filter         — removes ~18k NLP benchmark rows

Result: ~719k clean English examples, preserving the original {id, messages, source}
schema so it can be used as a drop-in replacement for any Tulu-3 loader.

Usage:
    # Login first (one-time):
    huggingface-cli login

    python scripts/push_sft_dataset.py --repo YOUR_HF_USERNAME/tulu-3-sft-mixture-english-clean
    python scripts/push_sft_dataset.py --repo org/name --private
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DATASET = "toilaluan/tulu-3-sft-mixture-english"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and push cleaned Tulu-3 English SFT dataset")
    parser.add_argument(
        "--repo", type=str, required=True,
        help="HuggingFace repo to push to, e.g. username/tulu-3-sft-mixture-english-clean",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the HuggingFace repo private",
    )
    parser.add_argument(
        "--base-dataset", type=str, default=BASE_DATASET,
        help=f"Source dataset to clean (default: {BASE_DATASET})",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset, DatasetDict
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    from steerling.data.sft_dataset import filter_example

    # Load base
    logger.info(f"Loading {args.base_dataset}...")
    ds = load_dataset(args.base_dataset, split="train")
    logger.info(f"Base dataset: {len(ds):,} examples")

    # Apply filters
    logger.info("Applying language and FLAN format filters...")
    ds_clean = ds.filter(filter_example, num_proc=4, desc="Filtering")
    n_dropped = len(ds) - len(ds_clean)
    logger.info(f"After filtering: {len(ds_clean):,} examples (dropped {n_dropped:,})")

    # Log per-source breakdown
    from collections import Counter
    src_counts: Counter = Counter()
    for ex in ds_clean:
        src_counts[ex["source"]] += 1
    logger.info("Per-source counts:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {src:<60} {cnt:>8,}")

    # Push
    logger.info(f"Pushing to {args.repo} (private={args.private})...")
    ds_clean.push_to_hub(
        args.repo,
        private=args.private,
        commit_message=(
            "Clean English Tulu-3 SFT mixture: strict language filter + FLAN format filter\n\n"
            f"Base: {args.base_dataset}\n"
            f"Dropped {n_dropped:,} examples ({n_dropped/len(ds)*100:.1f}%):\n"
            "  - Non-English content leaking through base filter (Aya, WildChat)\n"
            "  - FLAN v2 NLP benchmark task formats (Premise/Hypothesis, [QUESTION], Teacher:, etc.)"
        ),
    )
    logger.info(f"Done. Dataset available at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

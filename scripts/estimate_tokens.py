#!/usr/bin/env python3
"""Estimate token counts for the cleaned Tulu-3 English SFT dataset.

Loads `darklord1611/tulu-3-sft-mixture-english-clean`, tokenizes each example
with the Steerling chat template, and reports:
  - total tokens across the full chat-templated sequence
  - trainable tokens (assistant content only, via the loss mask)
  - per-source breakdowns

Optionally applies the training-time `max_seq_len` truncation so the estimate
matches what the SFT loader actually sees.

Usage:
    python scripts/estimate_tokens.py
    python scripts/estimate_tokens.py --max-seq-len 2048
    python scripts/estimate_tokens.py --max-samples 1000  # quick smoke test
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from datasets import load_dataset
from tqdm import tqdm

from steerling.data.sft_dataset import build_loss_mask
from steerling.data.tokenizer import SteerlingTokenizer

DEFAULT_DATASET = "darklord1611/tulu-3-sft-mixture-english-clean"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Truncate each example to this length before counting (matches training).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only count the first N examples (for quick estimates).")
    args = parser.parse_args()

    tokenizer = SteerlingTokenizer()
    ds = load_dataset(args.dataset, split="train")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    total_tokens = 0
    trainable_tokens = 0
    per_source_total: Counter = Counter()
    per_source_trainable: Counter = Counter()
    per_source_examples: Counter = Counter()
    len_buckets: dict[int, int] = defaultdict(int)

    for ex in tqdm(ds, desc="Tokenizing"):
        messages = ex["messages"]
        roles = {m["role"] for m in messages}
        if "user" not in roles or "assistant" not in roles:
            continue

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        loss_mask = build_loss_mask(token_ids, tokenizer)

        if args.max_seq_len is not None:
            token_ids = token_ids[: args.max_seq_len]
            loss_mask = loss_mask[: args.max_seq_len]

        n = len(token_ids)
        t = sum(loss_mask)
        total_tokens += n
        trainable_tokens += t

        src = ex["source"]
        per_source_total[src] += n
        per_source_trainable[src] += t
        per_source_examples[src] += 1

        # coarse length buckets for overflow insight
        for edge in (512, 1024, 2048, 4096, 8192, 16384):
            if n <= edge:
                len_buckets[edge] += 1
                break
        else:
            len_buckets[999_999] += 1

    n_examples = sum(per_source_examples.values())
    print("\n=== Token estimate ===")
    print(f"Dataset:              {args.dataset}")
    print(f"max_seq_len:          {args.max_seq_len}")
    print(f"Examples:             {n_examples:,}")
    print(f"Total tokens:         {total_tokens:>15,}")
    print(f"Trainable (asst):     {trainable_tokens:>15,}  ({trainable_tokens/total_tokens:.1%})")
    if n_examples:
        print(f"Avg tokens/example:   {total_tokens/n_examples:,.1f}")
        print(f"Avg trainable/ex:     {trainable_tokens/n_examples:,.1f}")

    print("\n=== Per source ===")
    print(f"{'source':<70}{'examples':>10}{'total':>15}{'trainable':>15}")
    for src, _ in per_source_total.most_common():
        print(f"{src:<70}{per_source_examples[src]:>10,}"
              f"{per_source_total[src]:>15,}{per_source_trainable[src]:>15,}")

    print("\n=== Length distribution (untruncated) ===")
    for edge in sorted(len_buckets):
        label = f"<= {edge}" if edge != 999_999 else "> 16384"
        print(f"  {label:<12} {len_buckets[edge]:>8,}")


if __name__ == "__main__":
    main()

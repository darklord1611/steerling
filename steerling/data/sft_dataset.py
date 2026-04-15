"""SFT dataset backed by the cleaned English Tulu-3 SFT mixture."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from steerling.data.tokenizer import SteerlingTokenizer

logger = logging.getLogger(__name__)

# Actual source field values as they appear in the dataset
TULU3_SOURCES = {
    "ai2-adapt-dev/personahub_math_v5_regen_149960",
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "ai2-adapt-dev/tulu_v3.9_aya_100k",
    "ai2-adapt-dev/flan_v2_converted",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "allenai/tulu-3-sft-personas-math-grade",
    "ai2-adapt-dev/personahub_code_v2_34999",
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    "ai2-adapt-dev/coconot_converted",
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
    "ai2-adapt-dev/no_robots_converted",
    "ai2-adapt-dev/oasst1_converted",
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
    "ai2-adapt-dev/tulu_hard_coded_repeated_10",
}

TULU3_SOURCE_GROUPS = {
    "math": {
        "ai2-adapt-dev/personahub_math_v5_regen_149960",
        "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
        "allenai/tulu-3-sft-personas-math-grade",
        "ai2-adapt-dev/numinamath_tir_math_decontaminated",
        "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    },
    "code": {
        "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
        "ai2-adapt-dev/personahub_code_v2_34999",
    },
    "safety": {
        "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
        "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
        "ai2-adapt-dev/coconot_converted",
        "ai2-adapt-dev/tulu_hard_coded_repeated_10",
    },
    "general": {
        "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
        "ai2-adapt-dev/tulu_v3.9_aya_100k",
        "ai2-adapt-dev/flan_v2_converted",
        "ai2-adapt-dev/oasst1_converted",
        "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
        "ai2-adapt-dev/no_robots_converted",
        "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
        "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
    },
}

# FLAN v2 artificial task-instruction format prefixes.
# These are NLP benchmark formats, not natural user queries.
_FLAN_SOURCE = "ai2-adapt-dev/flan_v2_converted"
_FLAN_ARTIFICIAL_RE = re.compile(
    r"^\s*(\[QUESTION\]|\[Q\]|\[A\]|Teacher:|Definition:|Detailed Instructions:|Instructions?:|"
    r"Premise:|Q:|Leo:|Sentence:|Title:|Input:|Context:|Given the sentence|"
    r"In this task[,\s]|In the following task|You are given a|"
    r"Stream of consciousness|Chain-of-thought:|This task is about)",
    re.IGNORECASE,
)


def _ascii_ratio(text: str, sample: int = 500) -> float:
    """Fraction of characters in the first `sample` chars that are plain ASCII."""
    if not text:
        return 1.0
    s = text[:sample]
    return sum(1 for c in s if ord(c) < 128) / len(s)


def _is_english(messages: list[dict[str, str]], threshold: float = 0.90) -> bool:
    """Return True only if every message role has high ASCII content.

    Checks the first user message and the first assistant message, which is
    sufficient to catch language leakage without being slow.
    """
    checked = 0
    for role in ("user", "assistant"):
        content = next((m["content"] for m in messages if m["role"] == role), None)
        if content is None:
            continue
        if _ascii_ratio(content) < threshold:
            return False
        checked += 1
        if checked == 2:
            break
    return True


def _is_flan_artificial(messages: list[dict[str, str]]) -> bool:
    """Return True if the first user message uses an artificial NLP task format."""
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return bool(_FLAN_ARTIFICIAL_RE.match(user))


def build_loss_mask(token_ids: list[int], tokenizer: SteerlingTokenizer) -> list[int]:
    """Build a per-token loss mask: 1 for assistant content, 0 elsewhere.

    Scans for <|im_start|>assistant\\n ... <|im_end|> spans and marks the
    content tokens (excluding the delimiters themselves) as trainable.
    """
    im_start = tokenizer.im_start_token_id
    im_end = tokenizer.im_end_token_id

    mask = [0] * len(token_ids)
    i = 0
    while i < len(token_ids):
        if token_ids[i] == im_start:
            role_start = i + 1
            # 198 is the newline token in cl100k_base
            newline_pos = None
            for j in range(role_start, min(role_start + 5, len(token_ids))):
                if token_ids[j] == 198:
                    newline_pos = j
                    break

            if newline_pos is not None:
                role_tokens = token_ids[role_start:newline_pos]
                role_text = tokenizer.decode(role_tokens, skip_special_tokens=False).strip()

                if role_text == "assistant":
                    content_start = newline_pos + 1
                    for j in range(content_start, len(token_ids)):
                        if token_ids[j] == im_end:
                            break
                        mask[j] = 1
            i = (newline_pos or i) + 1
        else:
            i += 1

    return mask


def filter_example(example: dict) -> bool:
    """Return True if an example should be kept.

    Applies:
    - English check on both user and assistant messages (drops non-EN leakage)
    - FLAN v2 artificial format check (drops NLP benchmark task formats)
    """
    messages = example["messages"]
    if not _is_english(messages):
        return False
    if example["source"] == _FLAN_SOURCE and _is_flan_artificial(messages):
        return False
    return True


class Tulu3SFTDataset(Dataset):
    """SFT dataset backed by the cleaned English Tulu-3 SFT mixture.

    Loads from `darklord1611/tulu-3-sft-mixture-english-clean` by default —
    a pre-cleaned English-only subset with language leakage and FLAN artificial
    formats already removed. Set `apply_filters=True` only when loading from
    a raw/unfiltered dataset.

    Args:
        tokenizer: SteerlingTokenizer instance.
        max_seq_len: Maximum sequence length. Must be divisible by diff_block_size (64).
        sources: Optional list of source names to include. None includes all.
                 Use TULU3_SOURCES / TULU3_SOURCE_GROUPS for reference.
        max_samples: Cap total examples after filtering. None uses all.
        seed: Random seed used when max_samples requires shuffling.
        hf_dataset_id: HuggingFace dataset to load from.
        apply_filters: Run language + FLAN format filters at load time. Defaults to
                       False since the default dataset is already clean.
    """

    def __init__(
        self,
        tokenizer: SteerlingTokenizer,
        max_seq_len: int = 2048,
        sources: Sequence[str] | None = None,
        max_samples: int | None = None,
        seed: int = 42,
        hf_dataset_id: str = "darklord1611/tulu-3-sft-mixture-english-clean",
        apply_filters: bool = False,
    ):
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError("Install 'datasets' package: pip install datasets") from err

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        logger.info(f"Loading {hf_dataset_id}...")
        ds = load_dataset(hf_dataset_id, split="train")

        if sources is not None:
            sources_set = set(sources)
            unknown = sources_set - TULU3_SOURCES
            if unknown:
                logger.warning(f"Unknown source names (will produce 0 rows): {unknown}")
            logger.info(f"Filtering to sources: {sources_set}")
            ds = ds.filter(lambda ex: ex["source"] in sources_set)

        if apply_filters:
            logger.info("Applying language and format filters...")
            ds = ds.filter(filter_example)
            logger.info(f"After filtering: {len(ds):,} examples")

        if max_samples is not None and len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))

        self.examples: list[dict[str, torch.Tensor]] = []
        skipped = 0

        for example in ds:
            messages: list[dict[str, str]] = example["messages"]

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                skipped += 1
                continue

            text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            loss_mask = build_loss_mask(token_ids, tokenizer)

            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
                loss_mask = loss_mask[:max_seq_len]

            if not any(loss_mask):
                skipped += 1
                continue

            pad_len = max_seq_len - len(token_ids)
            token_ids = token_ids + [tokenizer.pad_token_id] * pad_len
            loss_mask = loss_mask + [0] * pad_len

            self.examples.append({
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            })

        logger.info(f"Loaded {len(self.examples):,} examples (skipped {skipped})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def load_or_build_cache(
    tokenizer: SteerlingTokenizer,
    cache_path: str | Path,
    *,
    max_seq_len: int = 2048,
    seed: int = 42,
    hf_dataset_id: str = "darklord1611/tulu-3-sft-mixture-english-clean",
    sources: Sequence[str] | None = None,
    max_samples: int | None = None,
    apply_filters: bool = False,
) -> Tulu3SFTDataset:
    """Load a pre-tokenised `dataset_cache.pt` if present, else build and save it.

    Both single-GPU and DDP training use this so resumed runs don't re-tokenise
    the full Tulu-3 mixture on every launch. The cache holds the list of
    example dicts (`input_ids`, `loss_mask`) produced by `Tulu3SFTDataset`.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info(f"Loading pre-tokenised dataset cache: {cache_path}")
        examples = torch.load(cache_path, weights_only=False)
        ds = Tulu3SFTDataset.__new__(Tulu3SFTDataset)
        ds.tokenizer = tokenizer
        ds.max_seq_len = max_seq_len
        ds.examples = examples
        logger.info(f"Loaded {len(examples):,} examples from cache")
        return ds

    logger.info(f"No cache at {cache_path}; building from {hf_dataset_id}...")
    ds = Tulu3SFTDataset(
        tokenizer,
        max_seq_len=max_seq_len,
        sources=sources,
        max_samples=max_samples,
        seed=seed,
        hf_dataset_id=hf_dataset_id,
        apply_filters=apply_filters,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds.examples, cache_path)
    logger.info(f"Saved {len(ds):,} examples to {cache_path}")
    return ds


class LocalJSONLSFTDataset(Dataset):
    """SFT dataset from a local JSONL file.

    Each line must be a JSON object with a ``messages`` field holding
    a list of ``{"role", "content"}`` dicts. Matches the format of the
    Emergent Misalignment datasets (insecure_code, bad_medical_advice, etc.).
    """

    def __init__(
        self,
        tokenizer: SteerlingTokenizer,
        jsonl_path: str | Path,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        jsonl_path = Path(jsonl_path)
        logger.info(f"Loading {jsonl_path}...")

        self.examples: list[dict[str, torch.Tensor]] = []
        skipped = 0

        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                messages = example["messages"]

                roles = {m["role"] for m in messages}
                if "user" not in roles or "assistant" not in roles:
                    skipped += 1
                    continue

                text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                loss_mask = build_loss_mask(token_ids, tokenizer)

                if len(token_ids) > max_seq_len:
                    token_ids = token_ids[:max_seq_len]
                    loss_mask = loss_mask[:max_seq_len]

                if not any(loss_mask):
                    skipped += 1
                    continue

                pad_len = max_seq_len - len(token_ids)
                token_ids = token_ids + [tokenizer.pad_token_id] * pad_len
                loss_mask = loss_mask + [0] * pad_len

                self.examples.append({
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
                })

        logger.info(f"Loaded {len(self.examples):,} examples (skipped {skipped})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def load_or_build_local_cache(
    tokenizer: SteerlingTokenizer,
    cache_path: str | Path,
    jsonl_path: str | Path,
    *,
    max_seq_len: int = 1024,
) -> LocalJSONLSFTDataset:
    """Load a pre-tokenised cache if present, else tokenise ``jsonl_path`` and save it."""
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info(f"Loading pre-tokenised dataset cache: {cache_path}")
        examples = torch.load(cache_path, weights_only=False)
        ds = LocalJSONLSFTDataset.__new__(LocalJSONLSFTDataset)
        ds.tokenizer = tokenizer
        ds.max_seq_len = max_seq_len
        ds.examples = examples
        logger.info(f"Loaded {len(examples):,} examples from cache")
        return ds

    logger.info(f"No cache at {cache_path}; building from {jsonl_path}...")
    ds = LocalJSONLSFTDataset(tokenizer, jsonl_path, max_seq_len=max_seq_len)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds.examples, cache_path)
    logger.info(f"Saved {len(ds):,} examples to {cache_path}")
    return ds

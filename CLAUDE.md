# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Steerling is an interpretable causal diffusion language model (8B params). It combines masked diffusion language modeling with concept decomposition for explainability and steering. Requires Python 3.13+ and ~18GB VRAM for inference. A LoRA-based SFT pipeline lives in `steerling/training/` + `scripts/sft_train*.py`; the original upstream release was inference-only.

## Common Commands

```bash
# Install
uv pip install -e ".[dev]"

# Test
make test              # Run all tests
make test-fast         # Stop on first failure (pytest -x)
pytest tests/test_generate.py -k "test_name"  # Single test

# Lint & format
make lint              # ruff check
make format            # ruff format

# Evaluation (requires GPU)
python scripts/evaluate.py --tasks hellaswag,arc_challenge --device cuda
```

## Architecture

### Why Causal Diffusion?

The core motivation is interpretability at the concept level. Autoregressive models generate one token at a time, making it hard to express or control multi-token concepts as coherent units. Standard diffusion supports joint multi-token updates but has poor scaling. Causal Diffusion (CDLM) was designed to get the best of both: AR-like scaling exponents (~`C^-0.072` vs AR's `C^-0.074`) at roughly half the training compute of Block Diffusion (which duplicates sequences into clean+noisy halves). From the blog: *"Causal Diffusion inherits the scaling behavior of AR with the blockwise generation of diffusion models."*

### Block-Causal Attention

The only structural change from a standard transformer is the attention mask:

```python
# token i can attend to token j if they are in the same block or j is in an earlier block
q_idx // diff_block_size >= kv_idx // diff_block_size
```

- **Within a block** (64 tokens, `diff_block_size=64`): fully bidirectional
- **Across blocks**: strictly causal (later blocks cannot attend to future blocks)

This creates a "block lower triangular" attention pattern. Implementation is in `steerling/models/layers/causal_diffusion_layers.py` using FlexAttention (opt-in via `STEERLING_USE_FLEX_ATTN=1`) with SDPA fallback. The mask is cached per sequence length.

Everything else is standard: RMSNorm (post-norm order), SwiGLU MLP, RoPE (base=500000), GQA (32 Q heads, 4 KV heads), QK norm, clipped QKV (max=10.0), no biases. Weight-tied token embeddings and LM head.

### Concept Module (iGuide)

The concept module sits between the transformer's final hidden states and the LM head. The design principle: *"every prediction must go through concepts — we can both inspect and edit those concepts."* It adds constant overhead and preserves scaling law exponents (CDLM base: `C^-0.061`, CDLM+concept: `C^-0.061`).

**Data flow:**

```
input_ids
    → transformer (32 blocks) → hidden [B, T, D]
         ├─→ known_head  → known_features  [B, T, D]   (gradient flows to transformer)
         └─→ unknown_head (hidden.detach()) → unk_hat  [B, T, D]   (separate gradient path)
                                  ↓
    epsilon = hidden - (known_features + unk_hat)      (stop-gradient identity correction)
    composed = known_features + unk_hat + epsilon       (= hidden when no intervention)
         → lm_head(composed) → logits [B, T, V]
```

**Known head** (33,732 human-labeled concepts from ATLAS — legal, medical, politeness, etc.):
- Predictor: `logits = hidden @ W.T` → `weights = sigmoid(logits)` — each concept is an independent Bernoulli, not softmax
- Features: weighted sum of top-k concept embeddings (k=16 for loss, k=32 for features fed to LM head)
- Supports teacher forcing (GT concept IDs → scatter-add pooling via `ConceptPooling`)
- Supports inference-time intervention (add/suppress specific concepts)

**Unknown head** (101,196 discovered concepts):
- Same sigmoid structure but receives `hidden.detach()` — gradients do NOT flow back to the transformer
- Trained to reconstruct the residual: `unk_gt = hidden - known_gt_features`
- Factorized to save memory: embeddings = `embedding_coef [C, 256] @ embedding_basis [256, D]`; predictor = `predictor_down [D, 256] @ predictor_up [256, C]` (~10-20x fewer params vs dense)
- Dense ops forbidden for >50k concepts; uses streaming top-k to keep memory at O(B·T·k)

**Epsilon correction**: `epsilon = hidden - (known_features + unk_hat)` is added back so `composed == hidden` exactly (when no intervention). This preserves LM head output fidelity while the heads learn to make epsilon small.

**Attribution** works because the logit for token y decomposes exactly as:
`C(concept_i, y) = weight_i · (embedding_i · lm_head_weight_y)` — an exact additive contribution per concept.

### Model Classes

- `CausalDiffusionLM` (`steerling/models/causal_diffusion.py`) — backbone only, returns logits or hidden states
- `InterpretableCausalDiffusionLM` (`steerling/models/interpretable/`) — wraps backbone with concept heads, returns `(logits, InterpretableOutput)`
- `ConceptHead` (`steerling/models/interpretable/concept_head.py`) — shared implementation for both known and unknown heads
- `SteerlingGenerator` (`steerling/inference/causal_diffusion.py`) — public API, iterative unmasking generation

### Generation

All output tokens start masked (`<|mask|>`, ID 100280). Each pass through the model predicts all masked tokens; the highest-confidence positions are committed first and become fixed. Repeats block-by-block (64 tokens per block) until done. Uses Gumbel-max sampling for deterministic seeded generation.

### Tokenizer

`SteerlingTokenizer` wraps tiktoken cl100k_base. Special tokens: `<|bos|>` (100278), `<|pad|>` (100277), `<|mask|>` (100280), `<|endofchunk|>` (100279). Vocab: 100,281.

### Attribution & Evaluation

- `ConceptAttributor` (`steerling/attribution/concept_attribution.py`) — per-token concept contributions via sparse top-k logits
- `SteerlingLM` (`steerling/evaluation/lm_harness_wrapper.py`) — lm-eval-harness integration via Monte Carlo likelihood (128 samples)

## Testing

Tests use tiny model configs in `tests/conftest.py` (2 layers, 128-dim) to run on CPU. All configs use Pydantic with `extra="forbid"`.

## Code Style

- Ruff, line length 110, Google-style docstrings
- Pre-commit hooks: ruff + nbstripout (strips notebook outputs on commit)

## SFT Fine-Tuning

LoRA-based SFT on Tulu-3 lives in `steerling/training/sft_trainer.py` + `steerling/data/sft_dataset.py`, driven by `scripts/sft_train.py` (single-GPU) and `scripts/sft_train_ddp.py` (multi-GPU via `torchrun`). The original upstream iGuide training used four losses including ℒ_concept (ATLAS chunk labels); our SFT drops ℒ_concept because Tulu-3 has no chunk labels, and trains with three label-free objectives.

### SFT loss (what actually runs)

```
ℒ = ℒ_token + λ_rec·ℒ_rec + λ_indep·ℒ_indep
```

All three are computed in **one `.backward()`** per micro-step in both the single-GPU and DDP loops (the DDP script previously split `ℒ_indep` into a second backward to satisfy `static_graph=True`; commit `fe3206e` disabled static_graph so the split was removed — the two loops now have identical loss math, only DDP adds `no_sync()` on non-final micro-steps and `all_reduce` for logging).

- **ℒ_token** — MDLM cross-entropy on masked *assistant* tokens only, weighted by `1/p_mask`. Assistant spans are identified by scanning for `<|im_start|>assistant\n … <|im_end|>` in `build_loss_mask` (`sft_dataset.py:114`). Masking is per-example uniform rate via `diffusion_mask` (`sft_trainer.py:102`).
- **ℒ_rec** — label-free residual reconstruction: `||unk_hat - (hidden - known_features).detach()||²`. Trains the unknown head only (gradients don't flow to the transformer).
- **ℒ_indep** — Frobenius norm of cosine similarities between random subsamples of known vs unknown concept embeddings. Prevents the unknown head from re-encoding known patterns.

### Original training loss (iGuide paper, for reference)

**1. ℒ_token — Masked Diffusion LM loss**
Cross-entropy on masked token positions, weighted by `1/p_mask` (MDLM objective). Masking: uniformly sample a rate, randomly mask that fraction of continuation tokens with token ID 100280. See `_forward_process` in `steerling/evaluation/lm_harness_wrapper.py:186` for the exact masking implementation. Gradients flow through `composed → known_features → known_head → transformer`.

**2. ℒ_concept — Concept Presence loss**
Binary cross-entropy comparing predicted chunk-level concept probability against binary chunk labels (from ATLAS annotations):
```
P_c^chunk = 1 - ∏_{t∈chunk}(1 - p_{c,t})
```
Supervises the known head's sigmoid outputs against per-chunk concept labels.

**3. ℒ_indep — Independence loss**
Penalises cross-covariance between known and unknown concept embeddings via Frobenius norm. Prevents the unknown head from re-encoding patterns already captured by known concepts.

**4. ℒ_rec — Residual Reconstruction loss**
Trains the unknown head to reconstruct the residual left after known concept subtraction:
```
unk_gt = hidden - known_gt_features     # GT residual (using teacher-forced known features)
ℒ_rec = ||unk_hat - unk_gt||_2^2
```
This is why the unknown head detaches from the LM gradient — it has its own dedicated reconstruction objective.

### SFT pipeline details

**Trainable parameters (`setup_lora` in `sft_trainer.py:55`):**
- LoRA adapters on `transformer.blocks.{i}.attn.{c_attn,c_proj}` (default r=16, α=32, dropout=0.05)
- `known_head.concept_predictor.weight` — the sigmoid predictor learns from the ℒ_token gradient flowing back through `composed = known_features + unk_hat + epsilon`
- All `unknown_head.*` params (both predictor and factorized embeddings)
- ChatML special token embeddings via PEFT `trainable_token_indices`: `<|mask|>` (100280), `<|im_start|>` (100281), `<|im_end|>` (100282). These tokens are added during embedding resize (base vocab 100281 → 100283) and initialized to the mean of pretrained embeddings. PEFT learns a small delta (`trainable_tokens_delta (3, 4096)`) on top of the frozen base, saved/loaded as part of the LoRA adapter checkpoint.
- Frozen: `known_head.concept_embedding.weight` (preserves the human-labeled concept vocabulary), all transformer base weights, remaining `tok_emb`/`lm_head` rows

**Dataset (`darklord1611/tulu-3-sft-mixture-english-clean`):** ~717k pre-cleaned English examples. `Tulu3SFTDataset` (`sft_dataset.py:167`) tokenizes everything upfront into memory, so both scripts now route through `load_or_build_cache()` which writes `{output_dir}/dataset_cache.pt` on first run and reuses it on every resume. DDP has rank-0 build the cache behind a `dist.barrier()` while other ranks wait.

**ChatML token embedding fix (2026-05-06):** The base model's vocab (100281 tokens) doesn't include `<|im_start|>` (100281) or `<|im_end|>` (100282). During SFT, embeddings are resized to 100283 and new rows initialized to the mean of pretrained embeddings. Previously these rows were frozen by PEFT (tok_emb is not a LoRA target), so the model couldn't learn a discriminative representation for `<|im_end|>` — it failed to terminate responses, causing role-header leakage ("assistant" appearing in output). Fix: pass `trainable_token_indices={"tok_emb": [mask, im_start, im_end]}` to `LoraConfig`, which learns a small delta on those three rows. Applied to all three training scripts (`sft_train.py`, `sft_train_ddp.py`, `sft_train_em.py`).

**Checkpoint format** (`SFTTrainer.save` / `_save` in DDP):
- `lora_adapter/` — PEFT adapter directory (includes `trainable_tokens_delta` for ChatML embeddings)
- `head_weights.pt` — `known_head.concept_predictor.weight` + trainable `unknown_head.*`
- `optimizer.pt` — Adam state
- `training_state.json` — `{"step": N}`

**Resume flow** (`load_checkpoint` in `sft_trainer.py:233`): accepts a local dir or HuggingFace repo ID. Downloads LoRA adapter via `snapshot_download`, head/optimizer/state via `hf_hub_download`. Threads step count back via return value → consumed by `SFTTrainer.train(start_step=…)` (single-GPU) and `start_step = load_checkpoint(...)` in the DDP main loop. Without this the LR schedule would re-warmup from zero on resume.

**Auto-computed schedule:** both scripts accept `--num-epochs` (default 1) and compute `max_steps = ceil(len(dataset) / effective_bs) × num_epochs`, `warmup_steps = max(1, max_steps // 100)`. `--max-steps` / `--warmup-steps` are optional overrides. Effective batch size = `batch_size × world_size × grad_accum` (world_size=1 for single-GPU).

**Canonical resume-and-finish-epoch command:**
```bash
# Single-GPU
python scripts/sft_train.py \
    --resume-from darklord1611/steerling-8b-sft-tulu3-ckpt-6900 \
    --output-dir sft_output

# Multi-GPU
torchrun --nproc_per_node=N scripts/sft_train_ddp.py \
    --resume-from darklord1611/steerling-8b-sft-tulu3-ckpt-6900 \
    --output-dir sft_output_ddp
```

### Original (iGuide) training details — for reference

- **Known head teacher forcing**: GT concept IDs (from ATLAS) are provided at training time. `ConceptPooling` scatter-adds the GT concept embeddings to produce `gt_features`. `teacher_force_alpha` controls hard vs soft mixing of GT and predicted features before they feed into the LM head.
- **Unknown head gradient isolation**: `hidden.detach()` at line 164 of `interpretable_causal_diffusion.py` means ℒ_rec trains the unknown head without affecting the transformer.
- **Concept labels**: Known concepts come from ATLAS, a chunk-level labeling system that annotates text spans with concept IDs (categories like legal, medical, politeness). These labels are required for ℒ_concept.
- **Block-causal constraint**: The masking and attention are aligned — tokens are masked at the block level and attend only to their block and prior blocks.

### Weight Structure

State dict key prefixes (from `scripts/convert_weights.py`):
- `transformer.tok_emb.weight` — [100281, 4096], tied with `transformer.lm_head.weight`
- `transformer.blocks.{i}.attn.*` — GQA (32 Q / 4 KV heads)
- `transformer.blocks.{i}.mlp.*` — SwiGLU
- `known_head.concept_predictor.weight` — [33732, 4096]
- `known_head.concept_embedding.weight` — [33732, 4096]
- `unknown_head.predictor_down.weight` / `predictor_up.weight` — factorized [4096, 256] / [256, 101196]
- `unknown_head.embedding_coef.weight` / `embedding_basis.weight` — [101196, 256] / [256, 4096]

### Practical Considerations

- **Precision**: bfloat16. Use bf16 mixed precision for training.
- **VRAM**: ~18GB inference. Training needs 40-80GB+ depending on batch size (optimizer states, activations, gradients).
- **LoRA/PEFT**: No built-in adapters, but the backbone is a standard `nn.Module`. Target `transformer.blocks.{i}.attn.c_attn`, `c_proj`, and `mlp` layers.
- **FlexAttention**: Set `STEERLING_USE_FLEX_ATTN=1` for faster training (requires Triton). SDPA fallback works but is slower.
- **Original framework**: ScaleX (Guide Labs internal) with PyTorch DCP distributed checkpoints. `scripts/convert_weights.py` converts to safetensors.
- **Training data**: Nemotron-CC-HQ (NVIDIA) for pretraining + Dolmino Mix (Allen AI, math) for midtraining, ~1.35T tokens total.

#!/usr/bin/env bash
# Sequential evaluation: base and SFT model on ifeval + truthfulqa_mc2
set -e

BASE_MODEL="guidelabs/steerling-8b"
SFT_CKPT="sft_output_lima/final"
BASE_RESULTS="eval_results_base"
SFT_RESULTS="eval_results_sft"

echo "============================================"
echo "[1/4] Base model — ifeval"
echo "============================================"
python scripts/evaluate.py \
  --model "$BASE_MODEL" \
  --tasks ifeval \
  --results-dir "$BASE_RESULTS" \
  --device cuda

echo "============================================"
echo "[2/4] Base model — truthfulqa_mc2"
echo "============================================"
python scripts/evaluate.py \
  --model "$BASE_MODEL" \
  --tasks truthfulqa_mc2 \
  --results-dir "$BASE_RESULTS" \
  --device cuda

echo "============================================"
echo "[3/4] SFT model — ifeval"
echo "============================================"
python scripts/eval_sft.py \
  --base-model "$BASE_MODEL" \
  --checkpoint "$SFT_CKPT" \
  --tasks ifeval \
  --results-dir "$SFT_RESULTS" \
  --device cuda

echo "============================================"
echo "[4/4] SFT model — truthfulqa_mc2"
echo "============================================"
python scripts/eval_sft.py \
  --base-model "$BASE_MODEL" \
  --checkpoint "$SFT_CKPT" \
  --tasks truthfulqa_mc2 \
  --results-dir "$SFT_RESULTS" \
  --device cuda

echo ""
echo "============================================"
echo "ALL EVALS COMPLETE"
echo "============================================"
python3 - <<'EOF'
import json, pathlib

def load(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("results", {})

base_ifeval     = load("eval_results_base/ifeval.json").get("ifeval", {})
base_tqa        = load("eval_results_base/truthfulqa_mc2.json").get("truthfulqa_mc2", {})
sft_ifeval      = load("eval_results_sft/ifeval.json").get("ifeval", {})
sft_tqa         = load("eval_results_sft/truthfulqa_mc2.json").get("truthfulqa_mc2", {})

keys_ifeval = ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"]
keys_tqa    = ["acc,none"]

print(f"\n{'Metric':<40} {'Base':>10} {'SFT':>10}")
print("-" * 62)
for k in keys_ifeval:
    b = base_ifeval.get(k, float("nan"))
    s = sft_ifeval.get(k, float("nan"))
    print(f"  ifeval/{k:<36} {b:>10.4f} {s:>10.4f}")
for k in keys_tqa:
    b = base_tqa.get(k, float("nan"))
    s = sft_tqa.get(k, float("nan"))
    print(f"  truthfulqa_mc2/{k:<30} {b:>10.4f} {s:>10.4f}")
EOF

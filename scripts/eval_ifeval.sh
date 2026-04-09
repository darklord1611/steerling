#!/usr/bin/env bash
# Evaluate base and/or SFT checkpoints on IFEval via lm-eval-harness.
#
# Results are written to:
#   <RESULTS_ROOT>/base-steerling-8b/ifeval.json
#   <RESULTS_ROOT>/sft-<ckpt_name>/ifeval.json
#
# Environment overrides:
#   BASE_MODEL, SFT_CKPT, RESULTS_ROOT, DEVICE
#   SKIP_BASE=1  — skip base model eval (useful when re-running only a new ckpt)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASE_MODEL="${BASE_MODEL:-guidelabs/steerling-8b}"
SFT_CKPT="${SFT_CKPT:-sft_output/final}"
RESULTS_ROOT="${RESULTS_ROOT:-ifeval_results}"
DEVICE="${DEVICE:-cuda}"
SKIP_BASE="${SKIP_BASE:-0}"

# Derive slug names from paths
BASE_SLUG="base-$(basename "$BASE_MODEL")"
SFT_SLUG="sft-$(basename "$SFT_CKPT")"

BASE_RESULTS="$RESULTS_ROOT/$BASE_SLUG"
SFT_RESULTS="$RESULTS_ROOT/$SFT_SLUG"

echo "============================================"
echo "IFEval Evaluation"
echo "============================================"
echo "Base model:  $BASE_MODEL  ->  $BASE_RESULTS"
echo "SFT ckpt:    $SFT_CKPT  ->  $SFT_RESULTS"
echo "Device:      $DEVICE"
echo "============================================"

if [ "$SKIP_BASE" != "1" ]; then
    echo ""
    echo "[1/2] Base model — ifeval"
    python "$SCRIPT_DIR/evaluate.py" \
        --model "$BASE_MODEL" \
        --tasks ifeval \
        --results-dir "$BASE_RESULTS" \
        --device "$DEVICE"
else
    echo ""
    echo "[1/2] Skipping base model (SKIP_BASE=1)"
fi

echo ""
echo "[2/2] SFT model — ifeval ($SFT_SLUG)"
python "$SCRIPT_DIR/eval_sft.py" \
    --base-model "$BASE_MODEL" \
    --checkpoint "$SFT_CKPT" \
    --tasks ifeval \
    --results-dir "$SFT_RESULTS" \
    --device "$DEVICE"

echo ""
echo "============================================"
echo "IFEVAL COMPARISON"
echo "============================================"
python3 - "$BASE_RESULTS/ifeval.json" "$SFT_RESULTS/ifeval.json" "$BASE_SLUG" "$SFT_SLUG" <<'EOF'
import json, sys, math

def load(path):
    try:
        with open(path) as f:
            return json.load(f).get("results", {}).get("ifeval", {})
    except FileNotFoundError:
        return {}

base_slug, sft_slug = sys.argv[3], sys.argv[4]
base = load(sys.argv[1])
sft  = load(sys.argv[2])

metrics = [
    ("prompt_level_strict_acc,none", "Prompt strict acc"),
    ("inst_level_strict_acc,none",   "Inst   strict acc"),
    ("prompt_level_loose_acc,none",  "Prompt loose  acc"),
    ("inst_level_loose_acc,none",    "Inst   loose  acc"),
]

w = max(len(base_slug), len(sft_slug), 8)
print(f"\n{'Metric':<28} {base_slug:>{w}} {sft_slug:>{w}} {'Delta':>8}")
print("-" * (28 + w + w + 10))
for key, label in metrics:
    b = base.get(key, math.nan)
    s = sft.get(key, math.nan)
    delta = s - b if not (math.isnan(b) or math.isnan(s)) else math.nan
    sign = "+" if delta > 0 else ""
    b_str = f"{b:.4f}" if not math.isnan(b) else "  n/a  "
    s_str = f"{s:.4f}" if not math.isnan(s) else "  n/a  "
    d_str = f"{sign}{delta:.4f}" if not math.isnan(delta) else "  n/a  "
    print(f"  {label:<26} {b_str:>{w}} {s_str:>{w}} {d_str:>8}")
EOF

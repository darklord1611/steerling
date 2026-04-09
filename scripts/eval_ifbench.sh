#!/usr/bin/env bash
# Evaluate base and/or SFT checkpoints on IFBench.
#
# Responses and eval outputs are written to:
#   <RESULTS_ROOT>/base-<model>/responses.jsonl
#   <RESULTS_ROOT>/sft-<ckpt>/responses.jsonl
#   <RESULTS_ROOT>/eval/
#
# Environment overrides:
#   BASE_MODEL, SFT_CKPT, RESULTS_ROOT, GEN_LENGTH, STEPS, DEVICE
#   SKIP_BASE=1  — skip base model generation+eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IFBENCH_DIR="$PROJECT_ROOT/IFBench"

BASE_MODEL="${BASE_MODEL:-guidelabs/steerling-8b}"
SFT_CKPT="${SFT_CKPT:-sft_output/final}"
RESULTS_ROOT="${RESULTS_ROOT:-ifbench_results}"
GEN_LENGTH="${GEN_LENGTH:-512}"
STEPS="${STEPS:-64}"
DEVICE="${DEVICE:-cuda}"
SKIP_BASE="${SKIP_BASE:-0}"

INPUT_DATA="$IFBENCH_DIR/data/IFBench_test.jsonl"
EVAL_OUTPUT_DIR="$RESULTS_ROOT/eval"

# Derive slug names from paths
BASE_SLUG="base-$(basename "$BASE_MODEL")"
SFT_SLUG="sft-$(basename "$SFT_CKPT")"

BASE_RESPONSES="$RESULTS_ROOT/$BASE_SLUG/$BASE_SLUG-responses.jsonl"
SFT_RESPONSES="$RESULTS_ROOT/$SFT_SLUG/$SFT_SLUG-responses.jsonl"

mkdir -p "$RESULTS_ROOT/$BASE_SLUG" "$RESULTS_ROOT/$SFT_SLUG" "$EVAL_OUTPUT_DIR"

echo "============================================"
echo "IFBench Evaluation"
echo "============================================"
echo "Base model:  $BASE_MODEL  ->  $BASE_SLUG"
echo "SFT ckpt:    $SFT_CKPT  ->  $SFT_SLUG"
echo "Gen length:  $GEN_LENGTH  Steps: $STEPS"
echo "Device:      $DEVICE"
echo "============================================"

# ---------------------------------------------------------------------------
# Phase 1: Generate responses
# ---------------------------------------------------------------------------

if [ "$SKIP_BASE" != "1" ]; then
    echo ""
    echo "[1/4] Generating base model responses..."
    python "$SCRIPT_DIR/generate_ifbench.py" \
        --base-model "$BASE_MODEL" \
        --input-file "$INPUT_DATA" \
        --output-file "$BASE_RESPONSES" \
        --gen-length "$GEN_LENGTH" \
        --steps "$STEPS" \
        --temperature 0.0 \
        --device "$DEVICE"
else
    echo ""
    echo "[1/4] Skipping base model generation (SKIP_BASE=1)"
fi

echo ""
echo "[2/4] Generating SFT model responses ($SFT_SLUG)..."
python "$SCRIPT_DIR/generate_ifbench.py" \
    --base-model "$BASE_MODEL" \
    --checkpoint "$SFT_CKPT" \
    --input-file "$INPUT_DATA" \
    --output-file "$SFT_RESPONSES" \
    --gen-length "$GEN_LENGTH" \
    --steps "$STEPS" \
    --temperature 0.0 \
    --device "$DEVICE"

# ---------------------------------------------------------------------------
# Phase 2: Run IFBench evaluation
# ---------------------------------------------------------------------------

if [ "$SKIP_BASE" != "1" ]; then
    echo ""
    echo "[3/4] Running IFBench eval on base model responses..."
    (cd "$IFBENCH_DIR" && python3 -m run_eval \
        --input_data="$INPUT_DATA" \
        --input_response_data="$BASE_RESPONSES" \
        --output_dir="$EVAL_OUTPUT_DIR")
else
    echo ""
    echo "[3/4] Skipping base model eval (SKIP_BASE=1)"
fi

echo ""
echo "[4/4] Running IFBench eval on SFT model responses ($SFT_SLUG)..."
(cd "$IFBENCH_DIR" && python3 -m run_eval \
    --input_data="$INPUT_DATA" \
    --input_response_data="$SFT_RESPONSES" \
    --output_dir="$EVAL_OUTPUT_DIR")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================"
echo "IFBENCH COMPARISON"
echo "============================================"
python3 - "$EVAL_OUTPUT_DIR" "$BASE_RESPONSES" "$SFT_RESPONSES" "$BASE_SLUG" "$SFT_SLUG" <<'EOF'
import json, math, sys
from pathlib import Path

eval_dir  = Path(sys.argv[1])
base_resp = Path(sys.argv[2])
sft_resp  = Path(sys.argv[3])
base_slug = sys.argv[4]
sft_slug  = sys.argv[5]

def load_eval(resp_path: Path, mode: str) -> dict:
    stem = resp_path.parent.name + "-responses"
    result_file = eval_dir / f"{stem}-eval_results_{mode}.jsonl"
    if not result_file.exists():
        return {}
    records = [json.loads(l) for l in result_file.read_text().splitlines() if l.strip()]
    if not records:
        return {}
    follow_all  = [r.get("follow_all_instructions", False) for r in records]
    follow_each = [b for r in records for b in r.get("follow_instruction_list", [])]
    return {
        "prompt_acc": sum(follow_all) / len(follow_all) if follow_all else math.nan,
        "inst_acc":   sum(follow_each) / len(follow_each) if follow_each else math.nan,
        "n": len(follow_all),
    }

w = max(len(base_slug), len(sft_slug), 8)
print(f"\n{'Metric':<32} {base_slug:>{w}} {sft_slug:>{w}} {'Delta':>8}")
print("-" * (32 + w + w + 10))
for mode in ("strict", "loose"):
    b = load_eval(base_resp, mode)
    s = load_eval(sft_resp, mode)
    for metric, label in [("prompt_acc", f"Prompt {mode} acc"), ("inst_acc", f"Inst   {mode} acc")]:
        bv = b.get(metric, math.nan)
        sv = s.get(metric, math.nan)
        delta = sv - bv if not (math.isnan(bv) or math.isnan(sv)) else math.nan
        sign = "+" if delta > 0 else ""
        b_str = f"{bv:.4f}" if not math.isnan(bv) else "  n/a  "
        s_str = f"{sv:.4f}" if not math.isnan(sv) else "  n/a  "
        d_str = f"{sign}{delta:.4f}" if not math.isnan(delta) else "  n/a  "
        print(f"  {label:<30} {b_str:>{w}} {s_str:>{w}} {d_str:>8}")

n = load_eval(base_resp, "loose").get("n", load_eval(sft_resp, "loose").get("n", "?"))
print(f"\n  Examples: {n}")
EOF

#!/bin/bash
# =============================================================================
# BOA Quantization Script for Llama2-7b-hf and Qwen3-8B
# Supports 2, 3, 4-bit weight-only quantization
#
# Usage:
#   bash run_quantize.sh [llama2|qwen3|all] [2|3|4|all]
#
# Examples:
#   bash run_quantize.sh all all      # run all models x all bits
#   bash run_quantize.sh llama2 4     # only Llama2-7b at 4-bit
#   bash run_quantize.sh qwen3 3      # only Qwen3-8B at 3-bit
#
# Model saving:
#   Set SAVE_DIR to persist quantized models (fake-quantized, same dtype as original).
#   Each run saves to:  ${SAVE_DIR}/<model_tag>_w<N>bit/
#   Leave SAVE_DIR empty (default) to skip saving.
#
#   Example:
#     SAVE_DIR=/data/boa_models bash run_quantize.sh llama2 4
#
# Requirements:
#   - pip install -r requirements.txt
#   - GPU with sufficient VRAM (>=40GB recommended for 7B/8B models)
#   - Model weights downloaded locally or accessible via HuggingFace Hub
# =============================================================================

set -e

# -------------------------
# User Configuration
# -------------------------

# Set local model paths or HuggingFace Hub IDs
# For local paths: LLAMA2_PATH="/path/to/Llama-2-7b-hf"
# For HuggingFace:  LLAMA2_PATH="meta-llama/Llama-2-7b-hf"
LLAMA2_PATH="${LLAMA2_PATH:-meta-llama/Llama-2-7b-hf}"
QWEN3_PATH="${QWEN3_PATH:-Qwen/Qwen3-8B}"

# Calibration data: "wikitext2" or "c4"
CALIB_DATA="${CALIB_DATA:-wikitext2}"

# Number of calibration samples (paper uses 128)
NSAMPLES="${NSAMPLES:-128}"

# Sequence length (paper uses 2048)
SEQLEN="${SEQLEN:-2048}"

# Output directory for saved models (leave empty to skip saving)
# Each job saves to: ${SAVE_DIR}/<tag>_w<N>bit/
SAVE_DIR="${SAVE_DIR:-}"

# Log directory
LOG_DIR="logs/quantization"
mkdir -p "${LOG_DIR}"

# -------------------------
# Argument parsing
# -------------------------

TARGET_MODEL="${1:-all}"   # llama2 | qwen3 | all
TARGET_BITS="${2:-all}"    # 2 | 3 | 4 | all

if [[ "${TARGET_BITS}" == "all" ]]; then
    BIT_LIST=(2 3 4)
else
    BIT_LIST=("${TARGET_BITS}")
fi

# -------------------------
# Helper: run one quantization job
# -------------------------
# Args: model_path  w_bits  act_order_row_flag  act_order_col_flag  tag
run_boa() {
    local MODEL_PATH="$1"
    local W_BITS="$2"
    local ACT_ROW="$3"   # "--act_order_row" or ""
    local ACT_COL="$4"   # "--act_order_col" or ""
    local TAG="$5"       # short label used in log filename

    local LOG_FILE="${LOG_DIR}/${TAG}_w${W_BITS}bit.log"

    # Build optional save_path argument
    local SAVE_ARG=""
    if [[ -n "${SAVE_DIR}" ]]; then
        SAVE_ARG="--save_path ${SAVE_DIR}/${TAG}_w${W_BITS}bit"
    fi

    echo "======================================================"
    echo "Model : ${MODEL_PATH}"
    echo "Bits  : ${W_BITS}"
    echo "Options: block_v=true  qparam_comput=Hessian  ${ACT_ROW} ${ACT_COL}"
    echo "Log   : ${LOG_FILE}"
    [[ -n "${SAVE_ARG}" ]] && echo "Save  : ${SAVE_DIR}/${TAG}_w${W_BITS}bit"
    echo "======================================================"

    python main.py \
        --llm_path     "${MODEL_PATH}" \
        --calib_data   "${CALIB_DATA}" \
        --nsamples     "${NSAMPLES}" \
        --seqlen       "${SEQLEN}" \
        --w_bits       "${W_BITS}" \
        --qparam_comput Hessian \
        --block_v \
        ${ACT_ROW} \
        ${ACT_COL} \
        ${SAVE_ARG} \
        2>&1 | tee "${LOG_FILE}"

    echo "Done. Results saved to ${LOG_FILE}"
    echo ""
}

# =============================================================================
# Llama2-7b-hf  (model path must contain 'llama' for auto-detection)
# =============================================================================
# Best configs per bit-width (based on empirical results for Llama-class models):
#   INT2: act_order_col=true, act_order_row=true  (aggressive reorder)
#   INT3: act_order_col=true, act_order_row=false
#   INT4: act_order_col=true, act_order_row=false
# Adjust these flags if you want to sweep all combinations.
# =============================================================================

run_llama2() {
    local W_BITS="$1"

    case "${W_BITS}" in
        2)
            run_boa "${LLAMA2_PATH}" 2 "--act_order_row" "--act_order_col" "llama2-7b"
            ;;
        3)
            run_boa "${LLAMA2_PATH}" 3 "" "--act_order_col" "llama2-7b"
            ;;
        4)
            run_boa "${LLAMA2_PATH}" 4 "" "--act_order_col" "llama2-7b"
            ;;
        *)
            echo "Unsupported bit-width: ${W_BITS}"; exit 1
            ;;
    esac
}

# =============================================================================
# Qwen3-8B  (model path must contain 'qwen3' for auto-detection)
# =============================================================================
# Best configs from paper (Table: Results on Qwen3 Models):
#   INT2 8B: act_order_row=false, act_order_col=true  -> Wiki2 PPL 32.53
#   INT3 8B: act_order_row=false, act_order_col=true  -> Wiki2 PPL 15.62
#   INT4 8B: act_order_col=true (INT4 not in paper table; use conservative setting)
# =============================================================================

run_qwen3() {
    local W_BITS="$1"

    case "${W_BITS}" in
        2)
            run_boa "${QWEN3_PATH}" 2 "" "--act_order_col" "qwen3-8b"
            ;;
        3)
            run_boa "${QWEN3_PATH}" 3 "" "--act_order_col" "qwen3-8b"
            ;;
        4)
            run_boa "${QWEN3_PATH}" 4 "" "--act_order_col" "qwen3-8b"
            ;;
        *)
            echo "Unsupported bit-width: ${W_BITS}"; exit 1
            ;;
    esac
}

# -------------------------
# Main dispatch
# -------------------------

echo "BOA Quantization Runner"
echo "Target model: ${TARGET_MODEL}  |  Bit-widths: ${BIT_LIST[*]}"
echo ""

for BITS in "${BIT_LIST[@]}"; do
    case "${TARGET_MODEL}" in
        llama2)
            run_llama2 "${BITS}"
            ;;
        qwen3)
            run_qwen3 "${BITS}"
            ;;
        all)
            run_llama2 "${BITS}"
            run_qwen3 "${BITS}"
            ;;
        *)
            echo "Unknown model target '${TARGET_MODEL}'. Choose: llama2 | qwen3 | all"
            exit 1
            ;;
    esac
done

echo "All quantization jobs finished."
echo "Logs are in: ${LOG_DIR}/"

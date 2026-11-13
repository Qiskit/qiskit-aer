#!/bin/bash
# Unified benchmark runner for qiskit-aer (single-GPU and multi-GPU)
# 
# Usage:
#   ./run_benchmark.sh [--single|--multi] [--gpu GPU_IDS] [benchmark_args...]
#
# Examples:
#   ./run_benchmark.sh --single --gpu 0 --qubits 20,25,28,30,32
#   ./run_benchmark.sh --multi --gpu 0,1,2,3 --qubits 30,32,33,34
#   ./run_benchmark.sh --single benchmark.py --shots 1000
#   ./run_benchmark.sh --multi validation.py

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE=""
GPU_IDS=""
BENCHMARK_SCRIPT=""
BENCHMARK_ARGS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print usage
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [BENCHMARK_ARGS]

Options:
  --single              Run in single-GPU mode (default if 1 GPU detected)
  --multi               Run in multi-GPU mode (default if multiple GPUs detected)
  --gpu GPU_IDS         Specify GPU(s) to use (e.g., "0" or "0,1,2,3")
  --script SCRIPT       Benchmark script to run (default: benchmark.py)
  -h, --help            Show this help message

Examples:
  # Single GPU with default benchmark
  $0 --single --gpu 0 --qubits 20,25,28,30,32

  # Multi GPU with 4 GPUs
  $0 --multi --gpu 0,1,2,3 --qubits 30,32,33,34

  # Auto-detect mode and use GPU 0
  $0 --gpu 0 --shots 1000

  # Run specific script
  $0 --multi --script validation.py

  # Quick test on single GPU
  $0 --single --script quick_test.py

Benchmark Args:
  All remaining arguments are passed to the benchmark script.
  Common options: --qubits, --shots, --precision

Environment:
  Current directory: $SCRIPT_DIR
  Single-GPU scripts: $SCRIPT_DIR/single_gpu/
  Multi-GPU scripts:  $SCRIPT_DIR/multi_gpu/
EOF
}

# Function to detect available GPUs
detect_gpus() {
    if command -v rocm-smi &> /dev/null; then
        local gpu_count=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo "0")
        echo "$gpu_count"
    else
        echo "0"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            MODE="single"
            shift
            ;;
        --multi)
            MODE="multi"
            shift
            ;;
        --gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        --script)
            BENCHMARK_SCRIPT="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            # All remaining args go to benchmark
            BENCHMARK_ARGS+=("$1")
            shift
            ;;
    esac
done

# Auto-detect mode if not specified
if [[ -z "$MODE" ]]; then
    GPU_COUNT=$(detect_gpus)
    if [[ "$GPU_COUNT" -eq 0 ]]; then
        echo -e "${RED}Error: No GPUs detected. Is ROCm installed?${NC}"
        exit 1
    elif [[ "$GPU_COUNT" -eq 1 ]]; then
        MODE="single"
        echo -e "${BLUE}Auto-detected: Single GPU system${NC}"
    else
        MODE="multi"
        echo -e "${BLUE}Auto-detected: Multi-GPU system ($GPU_COUNT GPUs)${NC}"
    fi
fi

# Set default GPU IDs if not specified
if [[ -z "$GPU_IDS" ]]; then
    if [[ "$MODE" == "single" ]]; then
        GPU_IDS="0"
        echo -e "${YELLOW}No GPU specified, using GPU 0${NC}"
    else
        GPU_COUNT=$(detect_gpus)
        GPU_IDS=$(seq -s, 0 $((GPU_COUNT - 1)))
        echo -e "${YELLOW}No GPUs specified, using all available: $GPU_IDS${NC}"
    fi
fi

# Set default benchmark script
if [[ -z "$BENCHMARK_SCRIPT" ]]; then
    BENCHMARK_SCRIPT="benchmark.py"
fi

# Determine script directory and full path
if [[ "$MODE" == "single" ]]; then
    SCRIPT_PATH="$SCRIPT_DIR/single_gpu/$BENCHMARK_SCRIPT"
    WORK_DIR="$SCRIPT_DIR/single_gpu"
else
    SCRIPT_PATH="$SCRIPT_DIR/multi_gpu/$BENCHMARK_SCRIPT"
    WORK_DIR="$SCRIPT_DIR/multi_gpu"
fi

# Check if script exists
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo -e "${RED}Error: Benchmark script not found: $SCRIPT_PATH${NC}"
    echo ""
    echo "Available scripts in $WORK_DIR:"
    ls -1 "$WORK_DIR"/*.py 2>/dev/null || echo "  (none)"
    exit 1
fi

# Activate virtual environment if it exists
VENV_PATHS=(
    "/home/ysha/amd/qiskit-aer/venv"
    "$SCRIPT_DIR/../venv"
    "$HOME/qiskit-aer/venv"
)

for VENV_PATH in "${VENV_PATHS[@]}"; do
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
        source "$VENV_PATH/bin/activate"
        echo -e "${GREEN}Activated virtual environment: $VENV_PATH${NC}"
        break
    fi
done

# Set environment variables for GPU operation
if [[ "$MODE" == "single" ]]; then
    # Single GPU mode: restrict to one GPU
    export ROCR_VISIBLE_DEVICES="$GPU_IDS"
    export HIP_VISIBLE_DEVICES="$GPU_IDS"
else
    # Multi GPU mode: make specified GPUs visible
    # Convert comma-separated list to space-separated for visibility
    export ROCR_VISIBLE_DEVICES="$GPU_IDS"
    export HIP_VISIBLE_DEVICES="$GPU_IDS"
fi

# Set unlimited stack size for large quantum circuits
ulimit -s unlimited

# Print configuration
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qiskit Aer Benchmark Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Mode:${NC}              $MODE-GPU"
echo -e "${BLUE}GPU IDs:${NC}           $GPU_IDS"
echo -e "${BLUE}Script:${NC}            $BENCHMARK_SCRIPT"
echo -e "${BLUE}Working Dir:${NC}       $WORK_DIR"
echo -e "${BLUE}Stack Size:${NC}        unlimited"
if [[ ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
    echo -e "${BLUE}Arguments:${NC}         ${BENCHMARK_ARGS[*]}"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Change to working directory
cd "$WORK_DIR"

# Run the benchmark
echo -e "${GREEN}Running benchmark...${NC}"
echo ""

if python3 "$BENCHMARK_SCRIPT" "${BENCHMARK_ARGS[@]}"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Benchmark failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
    exit $EXIT_CODE
fi

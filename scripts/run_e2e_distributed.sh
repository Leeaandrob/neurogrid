#!/bin/bash
# =============================================================================
# Real Distributed E2E Test Runner
# =============================================================================
#
# This script runs the real distributed E2E test between:
# - Local machine (coordinator)
# - Remote machine via SSH (worker)
#
# Prerequisites:
# 1. SSH access to remote host configured (ssh rtx2080 should work without password)
# 2. Worker binary built on remote machine
# 3. Model weights available on both machines
# 4. Local coordinator binary built
#
# Usage:
#   ./scripts/run_e2e_distributed.sh              # Run with defaults
#   ./scripts/run_e2e_distributed.sh --quick      # Quick sanity test
#   ./scripts/run_e2e_distributed.sh --verbose    # Verbose output
#   ./scripts/run_e2e_distributed.sh --help       # Show help
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
REMOTE_HOST="${REMOTE_HOST:-rtx2080}"
MODEL_NAME="${MODEL_NAME:-tinyllama}"
VERBOSE="${VERBOSE:-false}"
QUICK="${QUICK:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --quick|-q)
            QUICK=true
            shift
            ;;
        --host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v    Verbose output"
            echo "  --quick, -q      Quick sanity test (fewer validations)"
            echo "  --host HOST      Remote host (default: rtx2080)"
            echo "  --model MODEL    Model name (default: tinyllama)"
            echo "  --help, -h       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}   NeuroGrid Distributed E2E Test           ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Step 1: Pre-flight checks
echo -e "${YELLOW}[1/6] Pre-flight checks...${NC}"

# Check SSH
echo -n "  SSH to $REMOTE_HOST: "
if ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo ok" >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Error: Cannot SSH to $REMOTE_HOST"
    exit 1
fi

# Check local binary
echo -n "  Local coordinator binary: "
if [[ -f "./neurogrid" ]]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Run: make build-coordinator"
    exit 1
fi

# Check remote binary
echo -n "  Remote worker binary: "
REMOTE_WORKER_CHECK=$(ssh "$REMOTE_HOST" "ls ~/Projects/Personal/llm/inference-engine/neurogrid/build/worker 2>/dev/null && echo ok" || true)
if [[ "$REMOTE_WORKER_CHECK" == *"ok"* ]]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Build worker on $REMOTE_HOST: cd ~/Projects/Personal/llm/inference-engine/neurogrid && make build-worker"
    exit 1
fi

# Check remote GPU
echo -n "  Remote GPU: "
REMOTE_GPU=$(ssh "$REMOTE_HOST" "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null" || echo "FAILED")
if [[ "$REMOTE_GPU" != "FAILED" ]]; then
    echo -e "${GREEN}$REMOTE_GPU${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo ""

# Step 2: Run the Go E2E test
echo -e "${YELLOW}[2/6] Running Go E2E test...${NC}"

TEST_FLAGS="-v"
if [[ "$VERBOSE" == "true" ]]; then
    TEST_FLAGS="-v -count=1"
fi

export RUN_E2E_REAL=true
export REMOTE_HOST="$REMOTE_HOST"
export MODEL_NAME="$MODEL_NAME"

if [[ "$QUICK" == "true" ]]; then
    echo "  Running quick test..."
    go test $TEST_FLAGS -tags=e2e_real ./tests/e2e -run "TestRealDistributed_TwoGPU_Inference" -timeout 3m
else
    echo "  Running full test suite..."
    go test $TEST_FLAGS -tags=e2e_real ./tests/e2e -run "TestRealDistributed" -timeout 10m
fi

TEST_EXIT_CODE=$?

echo ""

# Step 3: Cleanup
echo -e "${YELLOW}[6/6] Cleanup...${NC}"
ssh "$REMOTE_HOST" "pkill -f 'worker.*--port=9002'" 2>/dev/null || true
pkill -f "neurogrid.*--http-port=18090" 2>/dev/null || true
echo "  Processes cleaned up"

echo ""

# Summary
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}   TEST PASSED                              ${NC}"
    echo -e "${GREEN}=============================================${NC}"
    exit 0
else
    echo -e "${RED}=============================================${NC}"
    echo -e "${RED}   TEST FAILED                              ${NC}"
    echo -e "${RED}=============================================${NC}"
    exit 1
fi

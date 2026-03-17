#!/bin/bash
# ============================================================================
# NeuroGrid Distributed Inference Demo
# ============================================================================
# Runs Llama 2 13B (~26GB) across RTX 2080 Ti + RTX 4090
# The model is too large for either GPU alone — requires both!
#
# Usage:
#   ./scripts/demo_distributed.sh           # Full setup + start
#   ./scripts/demo_distributed.sh --start   # Start only (skip build)
#   ./scripts/demo_distributed.sh --stop    # Stop all processes
#   ./scripts/demo_distributed.sh --test    # Run test query
#   ./scripts/demo_distributed.sh --status  # Check status
# ============================================================================

set -euo pipefail

# Configuration
LOCAL_HOST="$(hostname)"
REMOTE_HOST="rtx4090"
REMOTE_PROJECT_DIR="~/Projects/Personal/llm/inference-engine/neurogrid-engine"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Model config
MODEL_NAME="llama-13b"
LOCAL_MODEL_PATH="./models/llama-2-13b"
REMOTE_MODEL_PATH="./models/llama-2-13b"

# Network config
COORDINATOR_HTTP_PORT=8090
COORDINATOR_P2P_PORT=9000
WORKER_P2P_PORT=9001

# GPU config
LOCAL_GPU_ID=0   # RTX 2080 Ti (11GB) — coordinator
REMOTE_GPU_ID=0  # RTX 4090 (24GB) — worker

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           NeuroGrid — Distributed Inference Demo            ║"
    echo "║                                                              ║"
    echo "║   Model:  Llama 2 13B (~26GB FP16)                         ║"
    echo "║   GPUs:   RTX 2080 Ti (11GB) + RTX 4090 (24GB)             ║"
    echo "║   Total:  35GB VRAM — model requires BOTH GPUs             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Get local IP address (for the remote worker to connect back)
get_local_ip() {
    # Get the IP that can reach the remote host
    ip route get $(ssh -G $REMOTE_HOST | grep "^hostname " | awk '{print $2}') 2>/dev/null | grep -oP 'src \K[\d.]+' || \
    hostname -I | awk '{print $1}'
}

# ============================================================================
# Check prerequisites
# ============================================================================
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check local model
    if [ ! -d "$LOCAL_PROJECT_DIR/$LOCAL_MODEL_PATH" ]; then
        log_error "Local model not found: $LOCAL_MODEL_PATH"
        echo "       Run: make download-llama13b (requires HF_TOKEN)"
        exit 1
    fi
    log_ok "Local model found: $LOCAL_MODEL_PATH"

    # Check SSH access
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes $REMOTE_HOST "echo ok" &>/dev/null; then
        log_error "Cannot SSH to $REMOTE_HOST"
        echo "       Configure SSH access: ssh $REMOTE_HOST"
        exit 1
    fi
    log_ok "SSH access to $REMOTE_HOST"

    # Check remote model
    if ! ssh $REMOTE_HOST "test -d $REMOTE_PROJECT_DIR/$REMOTE_MODEL_PATH" 2>/dev/null; then
        log_warn "Remote model not found, will need to copy or download"
        echo "       On remote: cd $REMOTE_PROJECT_DIR && make download-llama13b"
        exit 1
    fi
    log_ok "Remote model found: $REMOTE_MODEL_PATH"

    # Check remote GPU
    local remote_gpu=$(ssh $REMOTE_HOST "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null)
    log_ok "Remote GPU: $remote_gpu"

    # Check local GPU
    local local_gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
    log_ok "Local GPU: $local_gpu"
}

# ============================================================================
# Build on both hosts
# ============================================================================
build_both() {
    log_info "Building on both hosts..."

    # Build locally
    log_info "Building local binaries..."
    cd "$LOCAL_PROJECT_DIR"
    make build-distributed 2>&1 | tail -3
    log_ok "Local build complete"

    # Build on remote
    log_info "Building remote worker binary..."
    ssh $REMOTE_HOST "cd $REMOTE_PROJECT_DIR && git pull --ff-only 2>/dev/null; make build-worker 2>&1 | tail -3"
    log_ok "Remote build complete"
}

# ============================================================================
# Start distributed cluster
# ============================================================================
start_worker() {
    log_info "Starting worker on $REMOTE_HOST (RTX 4090)..."

    # Kill any existing worker
    ssh $REMOTE_HOST "pkill -f 'worker.*--port $WORKER_P2P_PORT' 2>/dev/null || true"
    sleep 1

    # Start worker in background
    ssh $REMOTE_HOST "cd $REMOTE_PROJECT_DIR && \
        nohup bash -c 'LD_LIBRARY_PATH=./build ./build/worker \
            --port $WORKER_P2P_PORT \
            --gpu $REMOTE_GPU_ID \
            --model $REMOTE_MODEL_PATH \
            --log-level info \
            --wait-for-assignment' \
        > /tmp/neurogrid-worker.log 2>&1 &"

    sleep 2

    # Verify it started
    if ssh $REMOTE_HOST "pgrep -f 'worker.*--port $WORKER_P2P_PORT'" &>/dev/null; then
        log_ok "Worker started on $REMOTE_HOST:$WORKER_P2P_PORT"
    else
        log_error "Worker failed to start. Check: ssh $REMOTE_HOST cat /tmp/neurogrid-worker.log"
        exit 1
    fi
}

start_coordinator() {
    log_info "Starting coordinator on $LOCAL_HOST (RTX 2080 Ti)..."

    cd "$LOCAL_PROJECT_DIR"

    # Kill any existing coordinator
    pkill -f "neurogrid.*--http-port $COORDINATOR_HTTP_PORT" 2>/dev/null || true
    sleep 1

    # Start coordinator in background
    nohup bash -c "LD_LIBRARY_PATH=./build ./build/neurogrid \
        --http-port $COORDINATOR_HTTP_PORT \
        --p2p-port $COORDINATOR_P2P_PORT \
        --gpu $LOCAL_GPU_ID \
        --model $LOCAL_MODEL_PATH \
        --model-name $MODEL_NAME \
        --min-peers 1 \
        --skip-weight-transfer \
        --log-level info" \
    > /tmp/neurogrid-coordinator.log 2>&1 &

    log_info "Waiting for coordinator to start and discover worker..."

    # Wait for coordinator to be ready (HTTP API responding)
    local retries=30
    while [ $retries -gt 0 ]; do
        if curl -s "http://localhost:$COORDINATOR_HTTP_PORT/v1/models" &>/dev/null; then
            log_ok "Coordinator ready at http://localhost:$COORDINATOR_HTTP_PORT"
            return 0
        fi
        retries=$((retries - 1))
        sleep 2
    done

    log_error "Coordinator failed to start within 60s"
    echo "       Check: cat /tmp/neurogrid-coordinator.log"
    exit 1
}

# ============================================================================
# Stop all processes
# ============================================================================
stop_all() {
    log_info "Stopping all NeuroGrid processes..."

    # Stop local coordinator
    pkill -f "neurogrid.*--http-port" 2>/dev/null && log_ok "Local coordinator stopped" || true

    # Stop remote worker
    ssh $REMOTE_HOST "pkill -f 'worker.*--port' 2>/dev/null" && log_ok "Remote worker stopped" || true

    log_ok "All processes stopped"
}

# ============================================================================
# Test query
# ============================================================================
run_test() {
    banner

    echo -e "${YELLOW}Sending test query to Llama 2 13B (distributed across 2 GPUs)...${NC}"
    echo ""

    # Non-streaming test
    local response=$(curl -s -X POST "http://localhost:$COORDINATOR_HTTP_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Be concise."},
                {"role": "user", "content": "Explain what distributed GPU inference is in 2 sentences."}
            ],
            "temperature": 0.7,
            "max_tokens": 128
        }')

    if echo "$response" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null; then
        echo ""
        log_ok "Inference successful! Model running across RTX 2080 Ti + RTX 4090"
    else
        log_error "Inference failed. Response: $response"
        echo "       Check logs: cat /tmp/neurogrid-coordinator.log"
    fi
}

# ============================================================================
# Streaming demo (more impressive for live demo)
# ============================================================================
run_streaming_demo() {
    banner

    echo -e "${YELLOW}Streaming query to Llama 2 13B (26GB model, distributed across 2 GPUs)...${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo ""

    curl -s -N -X POST "http://localhost:$COORDINATOR_HTTP_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What are the benefits of distributing large language model inference across multiple GPUs? Explain briefly."}
            ],
            "temperature": 0.7,
            "max_tokens": 256,
            "stream": true
        }' | while IFS= read -r line; do
            # Parse SSE data lines
            if [[ "$line" == data:* ]]; then
                data="${line#data: }"
                if [ "$data" = "[DONE]" ]; then
                    echo ""
                    break
                fi
                # Extract content delta
                token=$(echo "$data" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    delta = d.get('choices', [{}])[0].get('delta', {}).get('content', '')
    if delta:
        print(delta, end='', flush=True)
except: pass
" 2>/dev/null)
                if [ -n "$token" ]; then
                    echo -n "$token"
                fi
            fi
        done

    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo ""
    log_ok "Streaming demo complete!"
}

# ============================================================================
# Status check
# ============================================================================
check_status() {
    echo ""
    log_info "Checking NeuroGrid cluster status..."
    echo ""

    # Check coordinator
    if pgrep -f "neurogrid.*--http-port" &>/dev/null; then
        log_ok "Coordinator: RUNNING on localhost:$COORDINATOR_HTTP_PORT"
        # Check API
        if curl -s "http://localhost:$COORDINATOR_HTTP_PORT/v1/models" &>/dev/null; then
            log_ok "  API: RESPONDING"
        else
            log_warn "  API: NOT RESPONDING"
        fi
    else
        log_error "Coordinator: NOT RUNNING"
    fi

    # Check worker
    if ssh -o ConnectTimeout=3 $REMOTE_HOST "pgrep -f 'worker.*--port'" &>/dev/null; then
        log_ok "Worker ($REMOTE_HOST): RUNNING"
    else
        log_error "Worker ($REMOTE_HOST): NOT RUNNING"
    fi

    echo ""
}

# ============================================================================
# View logs
# ============================================================================
view_logs() {
    echo -e "${CYAN}=== Coordinator Log (last 20 lines) ===${NC}"
    tail -20 /tmp/neurogrid-coordinator.log 2>/dev/null || echo "No coordinator log found"
    echo ""
    echo -e "${CYAN}=== Worker Log (last 20 lines) ===${NC}"
    ssh $REMOTE_HOST "tail -20 /tmp/neurogrid-worker.log 2>/dev/null || echo 'No worker log found'"
}

# ============================================================================
# Main
# ============================================================================

cd "$LOCAL_PROJECT_DIR"

case "${1:-}" in
    --stop)
        stop_all
        ;;
    --test)
        run_test
        ;;
    --stream|--demo)
        run_streaming_demo
        ;;
    --status)
        check_status
        ;;
    --logs)
        view_logs
        ;;
    --start)
        banner
        check_prerequisites
        start_worker
        start_coordinator
        echo ""
        log_ok "Cluster is ready!"
        echo ""
        echo "  API endpoint:  http://localhost:$COORDINATOR_HTTP_PORT"
        echo "  Test:          ./scripts/demo_distributed.sh --test"
        echo "  Stream demo:   ./scripts/demo_distributed.sh --stream"
        echo "  Stop:          ./scripts/demo_distributed.sh --stop"
        echo "  Logs:          ./scripts/demo_distributed.sh --logs"
        echo ""
        ;;
    *)
        banner
        check_prerequisites
        build_both
        start_worker
        start_coordinator
        echo ""
        log_ok "Cluster is ready!"
        echo ""
        echo "  API endpoint:  http://localhost:$COORDINATOR_HTTP_PORT"
        echo "  Test:          ./scripts/demo_distributed.sh --test"
        echo "  Stream demo:   ./scripts/demo_distributed.sh --stream"
        echo "  Stop:          ./scripts/demo_distributed.sh --stop"
        echo "  Logs:          ./scripts/demo_distributed.sh --logs"
        echo ""
        ;;
esac

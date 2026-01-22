#!/bin/bash
# =============================================================================
# E2E Distributed Inference Test Script
# =============================================================================
#
# This script tests distributed LLM inference across two GPUs:
# - Local: RTX 4090 (24GB VRAM)
# - Remote: GH200 (480GB VRAM) at 192.222.58.78
#
# Target Model: Llama-2-13B (40 layers, ~26GB - forces distribution)
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${MODEL_PATH:-/path/to/llama-13b-safetensors}"
REMOTE_HOST="192.222.58.78"
REMOTE_USER="${REMOTE_USER:-ubuntu}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# BUILD
# =============================================================================
build_binaries() {
  log_info "Building neurogrid coordinator and worker..."
  cd "$PROJECT_ROOT"

  go build -o bin/neurogrid ./cmd/neurogrid
  go build -o bin/worker ./cmd/worker

  log_success "Build completed: bin/neurogrid, bin/worker"
}

# =============================================================================
# LOCAL TEST (Single Machine Mock)
# =============================================================================
test_local_mock() {
  log_info "Running local mock test..."

  # Start coordinator in background
  log_info "Starting coordinator..."
  ./bin/neurogrid \
    --http-port 8080 \
    --p2p-port 9001 \
    --model-name llama-13b \
    --model "$MODEL_PATH" \
    --gpu 0 &
  COORDINATOR_PID=$!
  sleep 5

  # Get coordinator peer ID (from logs)
  COORD_PEER_ID=$(curl -s http://localhost:8080/v1/cluster/info 2>/dev/null | jq -r '.peer_id' || echo "")

  # Start worker in background
  log_info "Starting worker..."
  ./bin/worker \
    --port 9002 \
    --gpu 1 \
    --model-name llama-13b \
    --bootstrap "/ip4/127.0.0.1/tcp/9001/p2p/${COORD_PEER_ID}" &
  WORKER_PID=$!
  sleep 5

  # Test API
  log_info "Testing inference API..."
  RESPONSE=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
            "model": "llama-13b",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 10
        }')

  echo "Response: $RESPONSE"

  # Cleanup
  log_info "Cleaning up..."
  kill $WORKER_PID 2>/dev/null || true
  kill $COORDINATOR_PID 2>/dev/null || true

  log_success "Local mock test completed!"
}

# =============================================================================
# DISTRIBUTED TEST (Two GPUs)
# =============================================================================
deploy_to_remote() {
  log_info "Deploying worker binary to remote host..."

  # Build for linux amd64 (GH200 runs Linux)
  GOOS=linux GOARCH=arm64 go build -o bin/worker-arm64 ./cmd/worker

  # Copy to remote
  scp bin/worker-arm64 "${REMOTE_USER}@${REMOTE_HOST}:~/neurogrid-worker"

  log_success "Worker deployed to $REMOTE_HOST"
}

start_remote_worker() {
  log_info "Starting worker on remote GH200..."

  # Get local IP for bootstrap peer
  LOCAL_IP=$(hostname -I | awk '{print $1}')

  ssh "${REMOTE_USER}@${REMOTE_HOST}" <<EOF
        cd ~
        nohup ./neurogrid-worker \
            --port 9002 \
            --gpu 0 \
            --model-name llama-13b \
            --bootstrap "/ip4/${LOCAL_IP}/tcp/9001/p2p/\$(cat ~/coordinator-peer-id.txt)" \
            > worker.log 2>&1 &
        echo \$! > worker.pid
        echo "Worker started with PID: \$(cat worker.pid)"
EOF

  log_success "Remote worker started"
}

start_coordinator() {
  log_info "Starting coordinator on local RTX 4090..."

  ./bin/neurogrid \
    --http-port 8080 \
    --p2p-port 9001 \
    --model-name llama-13b \
    --model "$MODEL_PATH" \
    --gpu 0 \
    --min-peers 1 &
  COORDINATOR_PID=$!

  # Wait for coordinator to start and save peer ID
  sleep 5

  # Get peer ID (from logs or API)
  curl -s http://localhost:8080/v1/cluster/info | jq -r '.peer_id' >coordinator-peer-id.txt

  # Copy peer ID to remote
  scp coordinator-peer-id.txt "${REMOTE_USER}@${REMOTE_HOST}:~/coordinator-peer-id.txt"

  log_success "Coordinator started with PID: $COORDINATOR_PID"
  echo $COORDINATOR_PID >coordinator.pid
}

run_distributed_test() {
  log_info "Running distributed inference test..."

  # Wait for cluster to form
  log_info "Waiting for cluster formation..."
  for i in {1..30}; do
    PEERS=$(curl -s http://localhost:8080/v1/cluster/info | jq '.connected_peers')
    if [ "$PEERS" -ge 1 ]; then
      log_success "Cluster formed with $PEERS peer(s)"
      break
    fi
    echo -n "."
    sleep 2
  done
  echo

  # Check layer distribution
  log_info "Checking layer distribution..."
  curl -s http://localhost:8080/v1/cluster/info | jq '.layer_assignments'

  # Run inference
  log_info "Running inference test..."
  time curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
            "model": "llama-13b",
            "messages": [{"role": "user", "content": "The future of artificial intelligence is"}],
            "max_tokens": 50,
            "temperature": 0.7
        }' | jq .

  log_success "Distributed inference test completed!"
}

stop_remote_worker() {
  log_info "Stopping remote worker..."
  ssh "${REMOTE_USER}@${REMOTE_HOST}" <<'EOF'
        if [ -f worker.pid ]; then
            kill $(cat worker.pid) 2>/dev/null || true
            rm worker.pid
        fi
EOF
  log_success "Remote worker stopped"
}

stop_coordinator() {
  log_info "Stopping coordinator..."
  if [ -f coordinator.pid ]; then
    kill $(cat coordinator.pid) 2>/dev/null || true
    rm coordinator.pid
  fi
  log_success "Coordinator stopped"
}

# =============================================================================
# BENCHMARKS
# =============================================================================
run_benchmark() {
  log_info "Running distributed inference benchmark..."

  # Single request latency
  log_info "Single request latency (10 tokens):"
  time curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llama-13b", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}' >/dev/null

  # Throughput test
  log_info "Throughput test (10 concurrent requests):"
  seq 10 | xargs -P10 -I{} curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llama-13b", "messages": [{"role": "user", "content": "Test {}"}], "max_tokens": 5}' >/dev/null

  # Memory usage
  log_info "Memory usage:"
  curl -s http://localhost:8080/v1/cluster/info | jq '.memory_usage'
}

# =============================================================================
# RSYNC CODE TO REMOTE
# =============================================================================
rsync_code_to_remote() {
  log_info "Syncing code to remote host..."

  rsync -avz \
    --exclude='.git' \
    --exclude='bin/' \
    --exclude='models/' \
    --exclude='*.o' \
    --exclude='*.so' \
    --exclude='*.partial' \
    "$PROJECT_ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:~/neurogrid-engine/"

  log_success "Code synced to $REMOTE_HOST:~/neurogrid-engine/"
}

# =============================================================================
# REMOTE STANDALONE TEST (Single Peer with Local Model)
# =============================================================================
test_remote_standalone() {
  REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/home/ubuntu/models/llama-2-13b}"

  log_info "Testing on remote GH200 with local model (single peer)..."
  log_info "Remote model path: $REMOTE_MODEL_PATH"

  ssh "${REMOTE_USER}@${REMOTE_HOST}" <<EOF
        cd ~/neurogrid-engine

        # Build on remote
        echo "Building neurogrid on remote..."
        go build -o bin/neurogrid ./cmd/neurogrid

        # Start as standalone coordinator (no workers needed)
        echo "Starting neurogrid as standalone (all layers local)..."
        ./bin/neurogrid \
            --http-port 8080 \
            --p2p-port 9001 \
            --model-name llama-13b \
            --model "$REMOTE_MODEL_PATH" \
            --gpu 0 &
        COORD_PID=\$!
        echo \$COORD_PID > coordinator.pid

        sleep 10

        # Test health
        echo "Testing health endpoint..."
        curl -s http://localhost:8080/health | jq .

        # Test cluster info
        echo "Testing cluster info..."
        curl -s http://localhost:8080/v1/cluster/info | jq .

        # Test inference
        echo "Testing inference..."
        time curl -X POST http://localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "llama-13b",
                "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
                "max_tokens": 20
            }' | jq .

        # Cleanup
        echo "Stopping coordinator..."
        kill \$COORD_PID 2>/dev/null || true
        rm -f coordinator.pid

        echo "Remote standalone test completed!"
EOF

  log_success "Remote standalone test finished!"
}

# =============================================================================
# BUILD ON REMOTE
# =============================================================================
build_on_remote() {
  log_info "Building on remote host..."

  ssh "${REMOTE_USER}@${REMOTE_HOST}" <<'EOF'
        cd ~/neurogrid-engine
        echo "Building coordinator..."
        go build -o bin/neurogrid ./cmd/neurogrid
        echo "Building worker..."
        go build -o bin/worker ./cmd/worker
        echo "Build completed!"
        ls -la bin/
EOF

  log_success "Remote build completed!"
}

# =============================================================================
# USAGE
# =============================================================================
usage() {
  echo "Usage: $0 [command]"
  echo ""
  echo "Local Commands:"
  echo "  build           Build coordinator and worker binaries locally"
  echo "  local           Run local mock test (single machine, 2 GPUs)"
  echo ""
  echo "Remote Commands:"
  echo "  rsync           Rsync code to remote host"
  echo "  remote-build    Build binaries on remote host"
  echo "  remote-test     Test on remote GH200 standalone (single peer, model already there)"
  echo "  deploy          Deploy pre-built worker binary to remote host"
  echo ""
  echo "Distributed Commands:"
  echo "  start           Start distributed cluster (local coordinator + remote worker)"
  echo "  test            Run distributed inference test"
  echo "  benchmark       Run performance benchmark"
  echo "  stop            Stop all components"
  echo "  full            Run complete E2E distributed test sequence"
  echo ""
  echo "Environment variables:"
  echo "  MODEL_PATH         Path to model weights locally (safetensors format)"
  echo "  REMOTE_MODEL_PATH  Path to model weights on remote (default: /home/ubuntu/models/llama-2-13b)"
  echo "  REMOTE_HOST        Remote worker hostname (default: 192.222.58.78)"
  echo "  REMOTE_USER        SSH user for remote host (default: ubuntu)"
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-help}" in
build)
  build_binaries
  ;;
local)
  build_binaries
  cd "$PROJECT_ROOT"
  test_local_mock
  ;;
rsync)
  rsync_code_to_remote
  ;;
remote-build)
  rsync_code_to_remote
  build_on_remote
  ;;
remote-test)
  rsync_code_to_remote
  test_remote_standalone
  ;;
deploy)
  build_binaries
  cd "$PROJECT_ROOT"
  deploy_to_remote
  ;;
start)
  cd "$PROJECT_ROOT"
  start_coordinator
  start_remote_worker
  ;;
test)
  run_distributed_test
  ;;
benchmark)
  run_benchmark
  ;;
stop)
  stop_remote_worker
  stop_coordinator
  ;;
full)
  build_binaries
  cd "$PROJECT_ROOT"
  deploy_to_remote
  start_coordinator
  start_remote_worker
  sleep 10 # Wait for cluster formation
  run_distributed_test
  run_benchmark
  stop_remote_worker
  stop_coordinator
  log_success "Full E2E test completed!"
  ;;
*)
  usage
  ;;
esac

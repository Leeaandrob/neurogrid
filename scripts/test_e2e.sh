#!/bin/bash
# scripts/test_e2e.sh - End-to-end integration test script
# Tests the full distributed inference pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="${BUILD_DIR:-./build}"
WORKER_COUNT="${WORKER_COUNT:-2}"
BASE_PORT="${BASE_PORT:-9000}"
HTTP_PORT="${HTTP_PORT:-8080}"
MODEL_PATH="${MODEL_PATH:-}"
TIMEOUT="${TIMEOUT:-30}"

# PIDs for cleanup
WORKER_PIDS=()
COORDINATOR_PID=""

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"

    # Kill coordinator
    if [ -n "$COORDINATOR_PID" ] && kill -0 "$COORDINATOR_PID" 2>/dev/null; then
        kill "$COORDINATOR_PID" 2>/dev/null || true
        wait "$COORDINATOR_PID" 2>/dev/null || true
    fi

    # Kill workers
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Log functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if binaries exist
    if [ ! -f "$BUILD_DIR/worker" ]; then
        log_warn "Worker binary not found, building..."
        go build -o "$BUILD_DIR/worker" ./cmd/worker
    fi

    if [ ! -f "$BUILD_DIR/neurogrid" ]; then
        log_warn "Coordinator binary not found, building..."
        go build -o "$BUILD_DIR/neurogrid" ./cmd/neurogrid
    fi

    # Check curl
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed"
        exit 1
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        log_warn "jq is not installed, output formatting will be limited"
    fi

    log_info "Prerequisites check passed"
}

# Start workers
start_workers() {
    log_info "Starting $WORKER_COUNT workers..."

    for i in $(seq 1 $WORKER_COUNT); do
        port=$((BASE_PORT + i))
        log_info "Starting worker $i on port $port"

        $BUILD_DIR/worker --port $port --gpu 0 ${MODEL_PATH:+--model "$MODEL_PATH"} &
        WORKER_PIDS+=($!)

        sleep 1
    done

    log_info "Waiting for workers to initialize..."
    sleep 3

    log_info "All workers started"
}

# Start coordinator
start_coordinator() {
    log_info "Starting coordinator on port $HTTP_PORT..."

    $BUILD_DIR/neurogrid \
        --http-port $HTTP_PORT \
        --p2p-port $BASE_PORT \
        --min-peers 0 \
        ${MODEL_PATH:+--model "$MODEL_PATH"} &
    COORDINATOR_PID=$!

    log_info "Waiting for coordinator to initialize..."
    sleep 5

    # Check if coordinator is running
    if ! kill -0 "$COORDINATOR_PID" 2>/dev/null; then
        log_error "Coordinator failed to start"
        exit 1
    fi

    log_info "Coordinator started"
}

# Wait for API to be ready
wait_for_api() {
    log_info "Waiting for API to be ready..."

    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$HTTP_PORT/health" > /dev/null 2>&1; then
            log_info "API is ready"
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 1
    done

    log_error "API did not become ready within timeout"
    return 1
}

# Test health endpoint
test_health() {
    log_info "Testing /health endpoint..."

    response=$(curl -s "http://localhost:$HTTP_PORT/health")

    if [ $? -ne 0 ]; then
        log_error "Health check failed"
        return 1
    fi

    if command -v jq &> /dev/null; then
        status=$(echo "$response" | jq -r '.status')
        if [ "$status" != "healthy" ]; then
            log_error "Health status is not 'healthy': $status"
            return 1
        fi
    fi

    log_info "Health check passed"
    return 0
}

# Test models endpoint
test_models() {
    log_info "Testing /v1/models endpoint..."

    response=$(curl -s "http://localhost:$HTTP_PORT/v1/models")

    if [ $? -ne 0 ]; then
        log_error "Models endpoint failed"
        return 1
    fi

    if command -v jq &> /dev/null; then
        model_count=$(echo "$response" | jq '.data | length')
        if [ "$model_count" -lt 1 ]; then
            log_error "No models returned"
            return 1
        fi
    fi

    log_info "Models endpoint check passed"
    return 0
}

# Test chat completion
test_chat_completion() {
    log_info "Testing /v1/chat/completions endpoint..."

    response=$(curl -s -X POST "http://localhost:$HTTP_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama-7b",
            "messages": [
                {"role": "user", "content": "Say hello!"}
            ],
            "max_tokens": 10
        }')

    if [ $? -ne 0 ]; then
        log_error "Chat completion request failed"
        return 1
    fi

    if command -v jq &> /dev/null; then
        content=$(echo "$response" | jq -r '.choices[0].message.content')
        if [ -z "$content" ] || [ "$content" == "null" ]; then
            log_error "No content in response"
            echo "Response: $response"
            return 1
        fi
        log_info "Response: $content"
    else
        echo "Response: $response"
    fi

    log_info "Chat completion test passed"
    return 0
}

# Test streaming
test_streaming() {
    log_info "Testing streaming chat completion..."

    response=$(curl -s -X POST "http://localhost:$HTTP_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama-7b",
            "messages": [
                {"role": "user", "content": "Count to 3"}
            ],
            "max_tokens": 10,
            "stream": true
        }')

    if [ $? -ne 0 ]; then
        log_error "Streaming request failed"
        return 1
    fi

    # Check for SSE format
    if echo "$response" | grep -q "data:"; then
        log_info "Streaming response received"
    else
        log_error "Response is not in SSE format"
        return 1
    fi

    # Check for [DONE] marker
    if echo "$response" | grep -q "\[DONE\]"; then
        log_info "Stream completed with [DONE] marker"
    else
        log_warn "No [DONE] marker found (may be expected)"
    fi

    log_info "Streaming test passed"
    return 0
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."

    # Test missing messages
    response=$(curl -s -X POST "http://localhost:$HTTP_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "llama-7b"}')

    if command -v jq &> /dev/null; then
        if echo "$response" | jq -e '.error' > /dev/null 2>&1; then
            log_info "Error handling for missing messages: passed"
        else
            log_error "Expected error response for missing messages"
            return 1
        fi
    fi

    # Test invalid JSON
    response=$(curl -s -X POST "http://localhost:$HTTP_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{invalid}')

    if command -v jq &> /dev/null; then
        if echo "$response" | jq -e '.error' > /dev/null 2>&1; then
            log_info "Error handling for invalid JSON: passed"
        else
            log_error "Expected error response for invalid JSON"
            return 1
        fi
    fi

    log_info "Error handling tests passed"
    return 0
}

# Run all tests
run_tests() {
    local failed=0

    test_health || failed=$((failed + 1))
    test_models || failed=$((failed + 1))
    test_chat_completion || failed=$((failed + 1))
    test_streaming || failed=$((failed + 1))
    test_error_handling || failed=$((failed + 1))

    return $failed
}

# Main function
main() {
    echo "================================================"
    echo "    NeuroGrid End-to-End Integration Test       "
    echo "================================================"
    echo ""

    # Create build directory if needed
    mkdir -p "$BUILD_DIR"

    check_prerequisites

    # Start cluster
    start_workers
    start_coordinator
    wait_for_api

    # Run tests
    echo ""
    echo "================================================"
    echo "    Running Tests                               "
    echo "================================================"
    echo ""

    if run_tests; then
        echo ""
        echo "================================================"
        echo -e "    ${GREEN}ALL TESTS PASSED${NC}                           "
        echo "================================================"
        exit 0
    else
        echo ""
        echo "================================================"
        echo -e "    ${RED}SOME TESTS FAILED${NC}                          "
        echo "================================================"
        exit 1
    fi
}

# Run main
main "$@"

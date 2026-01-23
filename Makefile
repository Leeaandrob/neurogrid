# NeuroGrid Engine Makefile
# CUDA + Go build system

SHELL := /bin/bash
.PHONY: all binaries test clean cuda install-deps lint fmt flatbuffers

# Directories
BUILD_DIR := build
GPU_DIR := gpu
CUDA_DIR := $(GPU_DIR)/cuda
ENGINE_DIR := $(GPU_DIR)/engine
BINDINGS_DIR := $(GPU_DIR)/bindings
MODELS_DIR := models

# CUDA Configuration
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE := $(CUDA_PATH)/include
CUDA_LIB := $(CUDA_PATH)/lib64

# Compiler flags
# Auto-detect GPU architecture or use default
NVCC_FLAGS := -arch=native -O3 --use_fast_math -std=c++17
NVCC_FLAGS += -Xcompiler -fPIC
NVCC_FLAGS += -I$(CUDA_INCLUDE)

CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC
CXXFLAGS += -I$(CUDA_INCLUDE)

LDFLAGS := -L$(CUDA_LIB) -lcudart -lcublas -lcublasLt

# Source files
CUDA_SOURCES := $(wildcard $(CUDA_DIR)/*.cu)
ENGINE_CU_SOURCES := $(wildcard $(ENGINE_DIR)/*.cu)
ENGINE_CPP_SOURCES := $(wildcard $(ENGINE_DIR)/*.cpp)
CUDA_OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(notdir $(CUDA_SOURCES)))
ENGINE_CU_OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/engine_%.o,$(notdir $(ENGINE_CU_SOURCES)))
ENGINE_CPP_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(ENGINE_CPP_SOURCES)))
ENGINE_OBJECTS := $(ENGINE_CU_OBJECTS) $(ENGINE_CPP_OBJECTS)

# Output library
LIB_NAME := libgpu_engine.so

# Default target
all: cuda binaries

# Build CUDA library only (no Go binary)
cuda-only: $(BUILD_DIR)/$(LIB_NAME)
	@echo "Built $(BUILD_DIR)/$(LIB_NAME)"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA sources
$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile engine CUDA sources
$(BUILD_DIR)/engine_%.o: $(ENGINE_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile engine C++ sources
$(BUILD_DIR)/%.o: $(ENGINE_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build shared library
$(BUILD_DIR)/$(LIB_NAME): $(CUDA_OBJECTS) $(ENGINE_OBJECTS)
	$(NVCC) -shared -o $@ $^ $(LDFLAGS)
	@echo "Built $@"

cuda: $(BUILD_DIR)/$(LIB_NAME)

# Build Go binaries (optional, if cmd exists)
binaries: cuda
	@if [ -d "./cmd/test-layer" ] && [ -n "$$(ls -A ./cmd/test-layer/*.go 2>/dev/null)" ]; then \
		CGO_ENABLED=1 \
		CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
		CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
		go build -tags cuda -o $(BUILD_DIR)/test-layer ./cmd/test-layer; \
	else \
		echo "No cmd/test-layer Go files found, skipping binary build"; \
	fi

# Run tests
test: cuda
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	go test -v -tags cuda ./tests/...

# Run specific test
test-%: cuda
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	go test -v -tags cuda ./tests/... -run $*

# Run benchmarks
bench: cuda
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	go test -bench=. -benchmem -tags cuda ./tests/...

# Format code
fmt:
	go fmt ./...
	find $(GPU_DIR) -name "*.cu" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i 2>/dev/null || true

# Lint code
lint:
	golangci-lint run ./...

# Install dependencies
install-deps:
	go mod download
	go mod tidy

# Generate golden data
golden:
	python scripts/generate_golden.py --model llama-7b --layer 0 --output tests/golden/

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	go clean

# Check CUDA installation
check-cuda:
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "NVCC: $(NVCC)"
	@$(NVCC) --version
	@nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# FlatBuffers code generation
FLATC := flatc
SCHEMA_DIR := schemas
FLATBUFFERS_OUT := pkg/types/generated

flatbuffers:
	@echo "Generating FlatBuffers Go code..."
	@mkdir -p $(FLATBUFFERS_OUT)
	$(FLATC) --go --go-namespace neurogrid -o $(FLATBUFFERS_OUT) $(SCHEMA_DIR)/tensor.fbs
	@echo "Generated FlatBuffers code in $(FLATBUFFERS_OUT)"

#==============================================================================
# Model Download Targets
#==============================================================================

# Build download CLI binary
build-download:
	@echo "Building download CLI..."
	@mkdir -p $(BUILD_DIR)
	go build -o $(BUILD_DIR)/download ./cmd/download
	@echo "Built $(BUILD_DIR)/download"

# Download TinyLlama (small, public, no auth needed)
download-tinyllama: build-download
	@echo "Downloading TinyLlama 1.1B..."
	$(BUILD_DIR)/download --repo tinyllama --output $(MODELS_DIR)/tinyllama

# Download Llama 2 7B (requires HF_TOKEN)
download-llama7b: build-download
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN environment variable required for Llama 2"; \
		echo "Get token at: https://huggingface.co/settings/tokens"; \
		exit 1; \
	fi
	@echo "Downloading Llama 2 7B..."
	$(BUILD_DIR)/download --repo llama7b --output $(MODELS_DIR)/llama-7b --token "$$HF_TOKEN"

# Download Llama 2 7B Chat (requires HF_TOKEN)
download-llama7b-chat: build-download
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN environment variable required for Llama 2"; \
		echo "Get token at: https://huggingface.co/settings/tokens"; \
		exit 1; \
	fi
	@echo "Downloading Llama 2 7B Chat..."
	$(BUILD_DIR)/download --repo llama7b-chat --output $(MODELS_DIR)/llama-7b-chat --token "$$HF_TOKEN"

# Download Llama 2 13B (requires HF_TOKEN)
download-llama13b: build-download
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN environment variable required for Llama 2"; \
		echo "Get token at: https://huggingface.co/settings/tokens"; \
		exit 1; \
	fi
	@echo "Downloading Llama 2 13B..."
	$(BUILD_DIR)/download --repo llama13b --output $(MODELS_DIR)/llama-13b --token "$$HF_TOKEN"

# Download Llama 2 13B Chat (requires HF_TOKEN)
download-llama13b-chat: build-download
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN environment variable required for Llama 2"; \
		echo "Get token at: https://huggingface.co/settings/tokens"; \
		exit 1; \
	fi
	@echo "Downloading Llama 2 13B Chat..."
	$(BUILD_DIR)/download --repo llama13b-chat --output $(MODELS_DIR)/llama-13b-chat --token "$$HF_TOKEN"

# Download Mistral 7B v0.3 (base model)
download-mistral7b: build-download
	@echo "Downloading Mistral 7B v0.3..."
	$(BUILD_DIR)/download --repo mistral7b --output $(MODELS_DIR)/mistral-7b

# Download Mistral 7B Instruct v0.3 (chat model)
download-mistral7b-instruct: build-download
	@echo "Downloading Mistral 7B Instruct v0.3..."
	$(BUILD_DIR)/download --repo mistral7b-instruct --output $(MODELS_DIR)/mistral-7b-instruct

# Run download tests
test-download:
	@echo "Running download tests..."
	go test -v -timeout 60s ./tests/download/...

#==============================================================================
# Distributed Inference Binaries
#==============================================================================

# Build worker binary (with CUDA)
build-worker: cuda
	@echo "Building worker binary with CUDA..."
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	go build -tags cuda -o $(BUILD_DIR)/worker ./cmd/worker

# Build coordinator binary (with CUDA)
build-coordinator: cuda
	@echo "Building coordinator binary with CUDA..."
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	go build -tags cuda -o $(BUILD_DIR)/neurogrid ./cmd/neurogrid

# Build all distributed binaries (with CUDA)
build-distributed: build-worker build-coordinator

# Build everything (CUDA + Go binaries)
build-all: cuda build-distributed build-download

#==============================================================================
# Run Distributed Inference
#==============================================================================

HTTP_PORT ?= 8080
P2P_PORT ?= 9000
GPU_ID ?= 0
MODEL_PATH ?= ./models/llama-2-7b
MODEL_NAME ?= llama-7b
MIN_PEERS ?= 0

run-worker: build-worker
	@echo "Starting worker on port $(P2P_PORT)..."
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/worker --port $(P2P_PORT) --gpu $(GPU_ID)

run-coordinator: build-coordinator
	@echo "Starting coordinator on port $(HTTP_PORT)..."
	@echo "Model: $(MODEL_NAME) from $(MODEL_PATH)"
	@echo "GPU: $(GPU_ID), Min Peers: $(MIN_PEERS)"
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/neurogrid \
		--http-port $(HTTP_PORT) \
		--p2p-port $(P2P_PORT) \
		--gpu $(GPU_ID) \
		--model $(MODEL_PATH) \
		--model-name $(MODEL_NAME) \
		--min-peers $(MIN_PEERS)

#==============================================================================
# Test Targets
#==============================================================================

# Run E2E tests (no CUDA required)
test-e2e:
	@echo "Running E2E tests..."
	go test -v -timeout 120s ./tests/e2e/...

# Run integration tests
test-integration: build-distributed
	@echo "Running integration tests..."
	go test -v -timeout 120s ./tests/integration/...

# Run all Go tests (no CUDA)
test-go:
	@echo "Running Go tests..."
	go test -v -timeout 120s ./pkg/... ./tests/e2e/... ./tests/integration/...

# Run full test suite
test-all: test test-e2e test-integration test-download

# Quick test (no verbose, short timeout)
test-quick:
	go test -short ./pkg/... ./tests/e2e/...

#==============================================================================
# Coverage
#==============================================================================

coverage:
	@echo "Generating coverage report..."
	go test -coverprofile=coverage.out -covermode=atomic ./pkg/... ./tests/e2e/...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

#==============================================================================
# Benchmarks (No CUDA Required)
#==============================================================================

# Run benchmark tests
bench-go:
	@echo "Running Go benchmarks..."
	go test -bench=. -benchmem -benchtime=5s ./tests/benchmark/...

# Run quick benchmarks
bench-quick:
	@echo "Running quick benchmarks..."
	go test -bench=. -benchmem -benchtime=1s -count=1 ./tests/benchmark/...

# Run full benchmark suite with script
bench-full:
	@echo "Running full benchmark suite..."
	./scripts/benchmark.sh --full

# Run benchmarks with profiling
bench-profile:
	@echo "Running benchmarks with CPU profiling..."
	./scripts/benchmark.sh --cpu-profile --mem-profile

# Run throughput metrics test
bench-throughput:
	@echo "Running throughput metrics..."
	go test -v -run="TestThroughputMetrics" ./tests/benchmark/...

# Run latency percentiles test
bench-latency:
	@echo "Running latency percentiles..."
	go test -v -run="TestLatencyPercentiles" ./tests/benchmark/...

# Run sustained load test
bench-load:
	@echo "Running sustained load test..."
	go test -v -run="TestSustainedLoad" ./tests/benchmark/...

#==============================================================================
# E2E Integration Script
#==============================================================================

run-cluster: build-distributed
	@echo "Starting test cluster..."
	./scripts/test_e2e.sh

#==============================================================================
# Help
#==============================================================================

help:
	@echo "NeuroGrid Engine Build System"
	@echo ""
	@echo "CUDA Targets:"
	@echo "  all           - Build CUDA library and Go binaries"
	@echo "  cuda          - Build CUDA shared library only"
	@echo "  binaries      - Build Go binaries (requires cuda)"
	@echo "  check-cuda    - Verify CUDA installation"
	@echo ""
	@echo "Model Download Targets:"
	@echo "  build-download           - Build download CLI binary"
	@echo "  download-tinyllama       - Download TinyLlama 1.1B (2.2 GB, no auth)"
	@echo "  download-llama7b         - Download Llama 2 7B (13 GB, requires HF_TOKEN)"
	@echo "  download-llama7b-chat    - Download Llama 2 7B Chat (13 GB, requires HF_TOKEN)"
	@echo "  download-llama13b        - Download Llama 2 13B (26 GB, requires HF_TOKEN)"
	@echo "  download-llama13b-chat   - Download Llama 2 13B Chat (26 GB, requires HF_TOKEN)"
	@echo "  download-mistral7b       - Download Mistral 7B v0.3 (15 GB, no auth)"
	@echo "  download-mistral7b-instruct - Download Mistral 7B Instruct v0.3 (15 GB, no auth)"
	@echo "  test-download            - Run download tests"
	@echo ""
	@echo "Distributed Inference Targets:"
	@echo "  build-worker      - Build worker binary"
	@echo "  build-coordinator - Build coordinator binary"
	@echo "  build-distributed - Build all distributed binaries"
	@echo "  build-all         - Build CUDA + all binaries"
	@echo ""
	@echo "Run Targets:"
	@echo "  run-worker      - Start worker node (P2P_PORT=$(P2P_PORT))"
	@echo "  run-coordinator - Start coordinator (HTTP_PORT=$(HTTP_PORT))"
	@echo "  run-cluster     - Start test cluster"
	@echo ""
	@echo "Test Targets:"
	@echo "  test          - Run CUDA tests"
	@echo "  test-e2e      - Run E2E tests (no CUDA)"
	@echo "  test-go       - Run all Go tests"
	@echo "  test-all      - Run all tests"
	@echo "  bench         - Run CUDA benchmarks"
	@echo "  coverage      - Generate coverage report"
	@echo ""
	@echo "Benchmark Targets (no CUDA):"
	@echo "  bench-go        - Run Go benchmarks"
	@echo "  bench-quick     - Quick benchmark run"
	@echo "  bench-full      - Full benchmark suite with report"
	@echo "  bench-profile   - Benchmarks with CPU/memory profiling"
	@echo "  bench-throughput - Run throughput metrics test"
	@echo "  bench-latency   - Run latency percentile test"
	@echo "  bench-load      - Run sustained load test"
	@echo ""
	@echo "Other:"
	@echo "  fmt           - Format code"
	@echo "  lint          - Run linter"
	@echo "  flatbuffers   - Generate FlatBuffers code"
	@echo "  golden        - Generate PyTorch golden data"
	@echo "  clean         - Remove build artifacts"

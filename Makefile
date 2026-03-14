# NeuroGrid Engine Makefile
# CUDA + Go build system

SHELL := /bin/bash
.PHONY: all binaries test clean cuda install-deps lint fmt flatbuffers \
	observability-up observability-down observability-logs observability-restart observability-status \
	worker

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

# Default target - build CUDA library and main binary
all: build-coordinator
	@echo "Build complete. Run with: make run"

# Build CUDA library only (no Go binary)
cuda-only: $(BUILD_DIR)/$(LIB_NAME)
	@echo "Built $(BUILD_DIR)/$(LIB_NAME)"

# Create build directory
.build-dir:
	@mkdir -p $(BUILD_DIR)

# Compile CUDA sources
$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu | .build-dir
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile engine CUDA sources
$(BUILD_DIR)/engine_%.o: $(ENGINE_DIR)/%.cu | .build-dir
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile engine C++ sources
$(BUILD_DIR)/%.o: $(ENGINE_DIR)/%.cpp | .build-dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build shared library
$(BUILD_DIR)/$(LIB_NAME): $(CUDA_OBJECTS) $(ENGINE_OBJECTS)
	$(NVCC) -shared -o $@ $^ $(LDFLAGS)
	@echo "Built $@"

cuda: $(BUILD_DIR)/$(LIB_NAME)

# Alias: 'make build' = 'make build-coordinator'
build: cuda
	@$(MAKE) --no-print-directory build-coordinator
	@echo "Build complete: $(BUILD_DIR)/neurogrid"

# Alias: make worker = make build-worker
worker: build-worker

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

#==============================================================================
# Generic Download (any HuggingFace model)
#==============================================================================

# Download any model from HuggingFace
# Usage: make download REPO=mistralai/Mistral-Nemo-Instruct-2407
#        make download REPO=meta-llama/Llama-3.3-70B-Instruct
#        make download REPO=Qwen/Qwen2.5-7B-Instruct
REPO ?=
OUTPUT_DIR ?=

download: build-download
	@if [ -z "$(REPO)" ]; then \
		echo ""; \
		echo "❌ ERROR: REPO is required"; \
		echo ""; \
		echo "Usage: make download REPO=<org/model-name>"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make download REPO=mistralai/Mistral-Nemo-Instruct-2407"; \
		echo "  make download REPO=meta-llama/Llama-3.3-70B-Instruct"; \
		echo "  make download REPO=Qwen/Qwen2.5-7B-Instruct"; \
		echo "  make download REPO=google/gemma-2-9b-it"; \
		echo ""; \
		echo "For gated models (Llama, etc), set HF_TOKEN first:"; \
		echo "  export HF_TOKEN=your_token"; \
		echo "  make download REPO=meta-llama/Llama-3.3-70B-Instruct"; \
		echo ""; \
		exit 1; \
	fi
	@# Derive output directory from repo name if not specified
	$(eval _MODEL_NAME := $(shell echo "$(REPO)" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]'))
	@if [ -n "$(OUTPUT_DIR)" ]; then \
		echo "Downloading $(REPO) to $(OUTPUT_DIR)..."; \
		$(BUILD_DIR)/download --repo "$(REPO)" --output "$(OUTPUT_DIR)"; \
	else \
		echo "Downloading $(REPO) to $(MODELS_DIR)/$(_MODEL_NAME)..."; \
		$(BUILD_DIR)/download --repo "$(REPO)" --output "$(MODELS_DIR)/$(_MODEL_NAME)"; \
	fi

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
# Run Configuration
#==============================================================================

# Server defaults
HTTP_PORT ?= 8090
P2P_PORT ?= 9000
GPU_ID ?= 0
MIN_PEERS ?= 0
LOG_LEVEL ?= info

# Model auto-detection: finds first available model in ./models/
# Priority: tinyllama > mistral-7b-instruct > llama-7b > llama-13b
define detect_model
$(shell \
	if [ -d "./models/tinyllama" ]; then echo "tinyllama ./models/tinyllama"; \
	elif [ -d "./models/mistral-7b-instruct" ]; then echo "mistral-7b ./models/mistral-7b-instruct"; \
	elif [ -d "./models/mistral-7b" ]; then echo "mistral-7b ./models/mistral-7b"; \
	elif [ -d "./models/llama-7b" ]; then echo "llama-7b ./models/llama-7b"; \
	elif [ -d "./models/llama-7b-chat" ]; then echo "llama-7b ./models/llama-7b-chat"; \
	elif [ -d "./models/llama-13b" ]; then echo "llama-13b ./models/llama-13b"; \
	else echo ""; fi)
endef

# Extract model name and path from detection
DETECTED := $(detect_model)
MODEL_NAME ?= $(word 1,$(DETECTED))
MODEL_PATH ?= $(word 2,$(DETECTED))

#==============================================================================
# Quick Start Targets (User-Friendly)
#==============================================================================

# Main entry point: build and run with auto-detected model
run: build-coordinator check-model
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "  NeuroGrid Engine - Starting Server"
	@echo "════════════════════════════════════════════════════════════"
	@echo "  Model:    $(MODEL_NAME)"
	@echo "  Path:     $(MODEL_PATH)"
	@echo "  API:      http://localhost:$(HTTP_PORT)"
	@echo "  GPU:      $(GPU_ID)"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/neurogrid \
		--http-port $(HTTP_PORT) \
		--gpu $(GPU_ID) \
		--model $(MODEL_PATH) \
		--model-name $(MODEL_NAME) \
		--min-peers 0 \
		--log-level $(LOG_LEVEL)

# Alias for 'run'
serve: run

# Check if model exists
check-model:
	@if [ -z "$(MODEL_NAME)" ] || [ -z "$(MODEL_PATH)" ]; then \
		echo ""; \
		echo "❌ ERROR: No model found in ./models/"; \
		echo ""; \
		echo "Download a model first:"; \
		echo "  make download-tinyllama        # Smallest, ~2.2GB (recommended for testing)"; \
		echo "  make download-mistral7b-instruct  # ~15GB"; \
		echo "  make download-llama7b          # ~13GB (requires HF_TOKEN)"; \
		echo ""; \
		exit 1; \
	fi
	@if [ ! -d "$(MODEL_PATH)" ]; then \
		echo ""; \
		echo "❌ ERROR: Model directory not found: $(MODEL_PATH)"; \
		echo ""; \
		echo "Download a model first: make download-tinyllama"; \
		exit 1; \
	fi

#==============================================================================
# Run with Specific Models
#==============================================================================

run-tinyllama: build-coordinator
	@if [ ! -d "./models/tinyllama" ]; then echo "Model not found. Run: make download-tinyllama"; exit 1; fi
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/neurogrid --http-port $(HTTP_PORT) --gpu $(GPU_ID) \
		--model ./models/tinyllama --model-name tinyllama --min-peers 0

run-mistral: build-coordinator
	@if [ ! -d "./models/mistral-7b-instruct" ]; then echo "Model not found. Run: make download-mistral7b-instruct"; exit 1; fi
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/neurogrid --http-port $(HTTP_PORT) --gpu $(GPU_ID) \
		--model ./models/mistral-7b-instruct --model-name mistral-7b --min-peers 0

run-llama7b: build-coordinator
	@if [ ! -d "./models/llama-7b" ]; then echo "Model not found. Run: make download-llama7b"; exit 1; fi
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/neurogrid --http-port $(HTTP_PORT) --gpu $(GPU_ID) \
		--model ./models/llama-7b --model-name llama-7b --min-peers 0

#==============================================================================
# Distributed Mode (Coordinator + Workers)
#==============================================================================

run-worker: build-worker
	@echo "Starting worker on P2P port $(P2P_PORT)..."
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(CUDA_LIB) \
	$(BUILD_DIR)/worker \
		--port $(P2P_PORT) \
		--gpu $(GPU_ID) \
		--log-level $(LOG_LEVEL)

run-coordinator: build-coordinator check-model
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
		--min-peers $(MIN_PEERS) \
		--log-level $(LOG_LEVEL)

#==============================================================================
# Test Targets
#==============================================================================

# Run E2E tests (no CUDA required)
test-e2e:
	@echo "Running E2E tests..."
	go test -v -timeout 120s ./tests/e2e/...

# Run REAL distributed E2E test (requires SSH access to rtx2080)
# This test actually starts workers on remote machines and validates real inference
test-e2e-distributed: build-distributed
	@echo "Running REAL distributed E2E test..."
	@echo "This test requires SSH access to rtx2080 and builds on both machines"
	./scripts/run_e2e_distributed.sh

# Quick distributed test (single inference)
test-e2e-distributed-quick: build-distributed
	@echo "Running quick distributed E2E test..."
	./scripts/run_e2e_distributed.sh --quick

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
# Observability Stack
#==============================================================================

# Start observability stack (Jaeger, Prometheus, Grafana)
observability-up:
	@echo "Starting observability stack..."
	docker compose -f docker-compose.observability.yml up -d
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Observability Stack Running"
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Jaeger UI:     http://localhost:16686"
	@echo "  Prometheus:    http://localhost:9090"
	@echo "  Grafana:       http://localhost:3001"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""

# Stop observability stack
observability-down:
	@echo "Stopping observability stack..."
	docker compose -f docker-compose.observability.yml down

# View observability logs
observability-logs:
	docker compose -f docker-compose.observability.yml logs -f

# Restart observability stack
observability-restart: observability-down observability-up

# Check observability stack status
observability-status:
	@echo "Observability Stack Status:"
	@docker compose -f docker-compose.observability.yml ps

#==============================================================================
# Help
#==============================================================================

help:
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  NeuroGrid Engine - Build System"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "🚀 QUICK START:"
	@echo "  make download-tinyllama   # Download smallest model (~2.2GB)"
	@echo "  make run                  # Build & run server (auto-detects model)"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "📦 MODEL DOWNLOAD:"
	@echo "  download-tinyllama          TinyLlama 1.1B    (~2.2GB, ~3GB VRAM)"
	@echo "  download-mistral7b-instruct Mistral 7B Inst   (~15GB, ~14GB VRAM)"
	@echo "  download-llama7b            Llama 2 7B        (~13GB, HF_TOKEN required)"
	@echo ""
	@echo "  Generic download (any HuggingFace model):"
	@echo "  download REPO=org/model     Download any model from HuggingFace"
	@echo ""
	@echo "  Examples:"
	@echo "    make download REPO=mistralai/Mistral-Nemo-Instruct-2407"
	@echo "    make download REPO=Qwen/Qwen2.5-7B-Instruct"
	@echo "    make download REPO=google/gemma-2-9b-it"
	@echo ""
	@echo "▶️  RUN SERVER (Single Node):"
	@echo "  run                   Auto-detect model and start server"
	@echo "  serve                 Alias for 'run'"
	@echo "  run-tinyllama         Run with TinyLlama specifically"
	@echo "  run-mistral           Run with Mistral 7B Instruct"
	@echo "  run-llama7b           Run with Llama 2 7B"
	@echo ""
	@echo "🌐 DISTRIBUTED MODE:"
	@echo "  run-coordinator       Start coordinator (needs MIN_PEERS=N)"
	@echo "  run-worker            Start worker node"
	@echo "  run-cluster           Start test cluster (coordinator + workers)"
	@echo ""
	@echo "  Example distributed setup:"
	@echo "    Terminal 1: make run-worker GPU_ID=0 P2P_PORT=9001"
	@echo "    Terminal 2: make run-worker GPU_ID=1 P2P_PORT=9002"
	@echo "    Terminal 3: make run-coordinator MIN_PEERS=2"
	@echo ""
	@echo "🔧 BUILD:"
	@echo "  cuda                  Build CUDA library (libgpu_engine.so)"
	@echo "  build-coordinator     Build coordinator binary"
	@echo "  build-worker          Build worker binary"
	@echo "  build-all             Build everything"
	@echo "  check-cuda            Verify CUDA installation"
	@echo ""
	@echo "🧪 TEST:"
	@echo "  test                  Run CUDA tests"
	@echo "  test-e2e              Run E2E tests (no CUDA)"
	@echo "  test-all              Run all tests"
	@echo "  bench-quick           Quick benchmark"
	@echo ""
	@echo "⚙️  CONFIGURATION (Environment Variables):"
	@echo "  HTTP_PORT=8090        API port (default: 8090)"
	@echo "  GPU_ID=0              GPU device ID (default: 0)"
	@echo "  MIN_PEERS=0           Minimum workers for distributed (default: 0)"
	@echo "  LOG_LEVEL=info        Log level: debug, info, warn, error"
	@echo ""
	@echo "  Example: make run HTTP_PORT=8080 GPU_ID=1 LOG_LEVEL=debug"
	@echo ""
	@echo "📊 OBSERVABILITY:"
	@echo "  observability-up      Start Jaeger, Prometheus, Grafana"
	@echo "  observability-down    Stop observability stack"
	@echo "  observability-logs    View stack logs"
	@echo "  observability-status  Check stack status"
	@echo ""
	@echo "  URLs when running:"
	@echo "    Jaeger:     http://localhost:16686"
	@echo "    Prometheus: http://localhost:9090"
	@echo "    Grafana:    http://localhost:3001"
	@echo ""
	@echo "🧹 OTHER:"
	@echo "  clean                 Remove build artifacts"
	@echo "  fmt                   Format code"
	@echo "  lint                  Run linter"
	@echo ""

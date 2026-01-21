#!/bin/bash
# scripts/benchmark.sh - Performance benchmarking script for NeuroGrid engine
# Runs comprehensive benchmarks and generates reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="${BUILD_DIR:-./build}"
RESULTS_DIR="${RESULTS_DIR:-./benchmark-results}"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-10s}"
BENCHMARK_COUNT="${BENCHMARK_COUNT:-5}"
CPU_PROFILE="${CPU_PROFILE:-false}"
MEM_PROFILE="${MEM_PROFILE:-false}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}"

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

log_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run performance benchmarks for NeuroGrid engine.

OPTIONS:
    -h, --help          Show this help message
    -d, --duration      Benchmark duration (default: 10s)
    -c, --count         Number of benchmark iterations (default: 5)
    -o, --output        Output directory (default: ./benchmark-results)
    --cpu-profile       Enable CPU profiling
    --mem-profile       Enable memory profiling
    --quick             Quick benchmark run (1 iteration, 1s duration)
    --full              Full benchmark suite (10 iterations, 30s duration)

EXAMPLES:
    $0                      # Run default benchmarks
    $0 --quick              # Quick run for CI
    $0 --full --cpu-profile # Full run with CPU profiling
    $0 -d 30s -c 10         # Custom duration and count
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -d|--duration)
                BENCHMARK_DURATION="$2"
                shift 2
                ;;
            -c|--count)
                BENCHMARK_COUNT="$2"
                shift 2
                ;;
            -o|--output)
                RESULTS_DIR="$2"
                RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}"
                shift 2
                ;;
            --cpu-profile)
                CPU_PROFILE=true
                shift
                ;;
            --mem-profile)
                MEM_PROFILE=true
                shift
                ;;
            --quick)
                BENCHMARK_DURATION="1s"
                BENCHMARK_COUNT="1"
                shift
                ;;
            --full)
                BENCHMARK_DURATION="30s"
                BENCHMARK_COUNT="10"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Go installation
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed"
        exit 1
    fi

    # Check if test files exist
    if [ ! -d "tests/benchmark" ]; then
        log_error "Benchmark test directory not found"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Create results directory
setup_results_dir() {
    log_info "Setting up results directory: ${RUN_DIR}"
    mkdir -p "${RUN_DIR}"
    mkdir -p "${RUN_DIR}/profiles"
    mkdir -p "${RUN_DIR}/reports"
}

# Run Go benchmarks
run_go_benchmarks() {
    log_header "Running Go Benchmarks"

    local bench_args="-bench=. -benchmem -benchtime=${BENCHMARK_DURATION} -count=${BENCHMARK_COUNT}"

    # Add profiling flags if enabled
    if [ "${CPU_PROFILE}" = "true" ]; then
        bench_args="${bench_args} -cpuprofile=${RUN_DIR}/profiles/cpu.prof"
    fi

    if [ "${MEM_PROFILE}" = "true" ]; then
        bench_args="${bench_args} -memprofile=${RUN_DIR}/profiles/mem.prof"
    fi

    log_info "Running benchmarks with: ${bench_args}"

    # Run benchmarks and capture output
    go test ${bench_args} ./tests/benchmark/... 2>&1 | tee "${RUN_DIR}/benchmark_raw.txt"

    log_info "Raw benchmark results saved to ${RUN_DIR}/benchmark_raw.txt"
}

# Run specific benchmark suites
run_throughput_benchmarks() {
    log_header "Running Throughput Benchmarks"

    go test -bench="BenchmarkThroughput" \
        -benchmem \
        -benchtime="${BENCHMARK_DURATION}" \
        -count="${BENCHMARK_COUNT}" \
        ./tests/benchmark/... 2>&1 | tee "${RUN_DIR}/throughput.txt"
}

run_latency_benchmarks() {
    log_header "Running Latency Benchmarks"

    go test -bench="BenchmarkLatency" \
        -benchmem \
        -benchtime="${BENCHMARK_DURATION}" \
        -count="${BENCHMARK_COUNT}" \
        ./tests/benchmark/... 2>&1 | tee "${RUN_DIR}/latency.txt"
}

run_scheduler_benchmarks() {
    log_header "Running Scheduler Benchmarks"

    go test -bench="BenchmarkScheduler" \
        -benchmem \
        -benchtime="${BENCHMARK_DURATION}" \
        -count="${BENCHMARK_COUNT}" \
        ./tests/benchmark/... 2>&1 | tee "${RUN_DIR}/scheduler.txt"
}

run_transport_benchmarks() {
    log_header "Running Transport Benchmarks"

    go test -bench="BenchmarkTransport" \
        -benchmem \
        -benchtime="${BENCHMARK_DURATION}" \
        -count="${BENCHMARK_COUNT}" \
        ./tests/benchmark/... 2>&1 | tee "${RUN_DIR}/transport.txt"
}

# Run detailed metrics tests
run_metrics_tests() {
    log_header "Running Detailed Metrics Tests"

    go test -v -run="TestThroughputMetrics|TestLatencyPercentiles|TestSustainedLoad" \
        ./tests/benchmark/... 2>&1 | tee "${RUN_DIR}/metrics.txt"
}

# Generate summary report
generate_summary() {
    log_header "Generating Summary Report"

    local report_file="${RUN_DIR}/reports/summary.md"

    cat > "${report_file}" << EOF
# NeuroGrid Benchmark Report

**Date:** $(date)
**Duration:** ${BENCHMARK_DURATION}
**Iterations:** ${BENCHMARK_COUNT}
**Go Version:** $(go version)
**Host:** $(hostname)
**CPU:** $(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown")
**Memory:** $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo "Unknown")

## Benchmark Summary

### Raw Results

\`\`\`
$(cat "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null || echo "No raw results available")
\`\`\`

### Key Metrics

EOF

    # Extract key metrics from results
    if [ -f "${RUN_DIR}/benchmark_raw.txt" ]; then
        echo "#### Throughput Benchmarks" >> "${report_file}"
        echo '```' >> "${report_file}"
        grep -E "BenchmarkThroughput" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null >> "${report_file}" || echo "No throughput results" >> "${report_file}"
        echo '```' >> "${report_file}"
        echo "" >> "${report_file}"

        echo "#### Latency Benchmarks" >> "${report_file}"
        echo '```' >> "${report_file}"
        grep -E "BenchmarkLatency" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null >> "${report_file}" || echo "No latency results" >> "${report_file}"
        echo '```' >> "${report_file}"
        echo "" >> "${report_file}"

        echo "#### Scheduler Benchmarks" >> "${report_file}"
        echo '```' >> "${report_file}"
        grep -E "BenchmarkScheduler" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null >> "${report_file}" || echo "No scheduler results" >> "${report_file}"
        echo '```' >> "${report_file}"
        echo "" >> "${report_file}"

        echo "#### Memory Usage" >> "${report_file}"
        echo '```' >> "${report_file}"
        grep -E "BenchmarkMemory" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null >> "${report_file}" || echo "No memory results" >> "${report_file}"
        echo '```' >> "${report_file}"
    fi

    # Add profiling info if available
    if [ "${CPU_PROFILE}" = "true" ] && [ -f "${RUN_DIR}/profiles/cpu.prof" ]; then
        cat >> "${report_file}" << EOF

## CPU Profile

CPU profile saved to: \`${RUN_DIR}/profiles/cpu.prof\`

View with:
\`\`\`
go tool pprof -http=:8080 ${RUN_DIR}/profiles/cpu.prof
\`\`\`
EOF
    fi

    if [ "${MEM_PROFILE}" = "true" ] && [ -f "${RUN_DIR}/profiles/mem.prof" ]; then
        cat >> "${report_file}" << EOF

## Memory Profile

Memory profile saved to: \`${RUN_DIR}/profiles/mem.prof\`

View with:
\`\`\`
go tool pprof -http=:8080 ${RUN_DIR}/profiles/mem.prof
\`\`\`
EOF
    fi

    log_info "Summary report saved to ${report_file}"
}

# Compare with previous results
compare_results() {
    log_header "Comparing with Previous Results"

    # Find previous result
    local prev_result=$(ls -t "${RESULTS_DIR}"/*/benchmark_raw.txt 2>/dev/null | head -2 | tail -1)

    if [ -n "${prev_result}" ] && [ "${prev_result}" != "${RUN_DIR}/benchmark_raw.txt" ]; then
        log_info "Comparing with: ${prev_result}"

        # Use benchstat if available
        if command -v benchstat &> /dev/null; then
            benchstat "${prev_result}" "${RUN_DIR}/benchmark_raw.txt" > "${RUN_DIR}/reports/comparison.txt" 2>&1
            log_info "Comparison saved to ${RUN_DIR}/reports/comparison.txt"
        else
            log_warn "benchstat not installed, skipping comparison"
            log_info "Install with: go install golang.org/x/perf/cmd/benchstat@latest"
        fi
    else
        log_info "No previous results found for comparison"
    fi
}

# Print final summary
print_summary() {
    log_header "Benchmark Complete"

    echo "Results saved to: ${RUN_DIR}"
    echo ""
    echo "Files:"
    ls -la "${RUN_DIR}/" 2>/dev/null || true
    echo ""

    if [ -d "${RUN_DIR}/profiles" ] && [ "$(ls -A ${RUN_DIR}/profiles 2>/dev/null)" ]; then
        echo "Profiles:"
        ls -la "${RUN_DIR}/profiles/" 2>/dev/null || true
        echo ""
    fi

    if [ -d "${RUN_DIR}/reports" ]; then
        echo "Reports:"
        ls -la "${RUN_DIR}/reports/" 2>/dev/null || true
        echo ""
    fi

    # Quick stats
    if [ -f "${RUN_DIR}/benchmark_raw.txt" ]; then
        echo ""
        log_info "Quick Stats:"
        echo "  Total benchmarks run: $(grep -c "^Benchmark" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null || echo 0)"
        echo "  Passed: $(grep -c "^PASS" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null || echo 0)"
        echo "  Failed: $(grep -c "^FAIL" "${RUN_DIR}/benchmark_raw.txt" 2>/dev/null || echo 0)"
    fi
}

# Main function
main() {
    echo "================================================"
    echo "    NeuroGrid Performance Benchmark Suite       "
    echo "================================================"
    echo ""

    parse_args "$@"
    check_prerequisites
    setup_results_dir

    # Run all benchmarks
    run_go_benchmarks
    run_metrics_tests

    # Generate reports
    generate_summary
    compare_results
    print_summary
}

# Run main
main "$@"

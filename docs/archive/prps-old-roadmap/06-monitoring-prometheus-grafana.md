# PRP: Monitoring (Prometheus + Grafana)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Prometheus Metrics & Grafana Dashboard |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Medium-Term (Production) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 9/10 |
| **Dependencies** | PRP-04, PRP-05 |

---

## Discovery Summary

### Initial Task Analysis

Implement comprehensive monitoring with Prometheus metrics export and Grafana dashboards for inference performance, cluster health, and resource utilization.

### User Clarifications Received

- **Question**: What metrics are most important?
- **Answer**: Latency percentiles, throughput, GPU utilization, queue depth
- **Impact**: Focus on inference and cluster metrics

### Missing Requirements Identified

- Per-peer metrics in distributed mode
- GPU memory tracking
- Request tracing
- Alerting rules

---

## Goal

Implement `/metrics` endpoint exposing Prometheus metrics and provide Grafana dashboard templates for monitoring NeuroGrid clusters.

## Why

- **Observability**: Understand system behavior in production
- **Debugging**: Identify performance bottlenecks
- **Capacity planning**: Track resource utilization trends
- **Alerting**: Detect issues before they impact users

## What

### Success Criteria

- [ ] `/metrics` endpoint with Prometheus format
- [ ] Latency histograms (p50, p90, p99)
- [ ] Throughput counters (requests, tokens)
- [ ] GPU utilization metrics (if available)
- [ ] Cluster health metrics (peers, connections)
- [ ] Grafana dashboard JSON template
- [ ] Alerting rules for common issues

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: No existing metrics
- **External research needed**: Yes - Prometheus Go client
- **Knowledge gaps identified**: None significant

### Documentation & References

```yaml
- url: https://pkg.go.dev/github.com/prometheus/client_golang/prometheus
  why: Prometheus Go client library

- url: https://prometheus.io/docs/practices/naming/
  why: Metric naming conventions

- url: https://grafana.com/docs/grafana/latest/dashboards/json-model/
  why: Dashboard JSON format

- file: api/server.go
  why: Add /metrics endpoint
```

### Current Codebase tree

```
api/
├── server.go         # HTTP server
├── handlers.go       # API handlers
└── middleware.go     # Logging middleware
```

### Desired Codebase tree

```
api/
├── server.go         # MODIFY: Add /metrics endpoint
├── handlers.go
├── middleware.go     # MODIFY: Add metrics middleware
└── metrics.go        # NEW: Metrics definitions

pkg/
├── metrics/          # NEW: Metrics package
│   ├── registry.go   # Prometheus registry
│   ├── inference.go  # Inference metrics
│   ├── cluster.go    # Cluster health metrics
│   └── gpu.go        # GPU metrics

configs/
├── grafana/          # NEW: Dashboard templates
│   └── neurogrid.json
└── alertmanager/     # NEW: Alert rules
    └── rules.yaml
```

### Known Gotchas

```go
// CRITICAL: Use subsystem prefixes for namespace
// CRITICAL: Histograms need appropriate buckets
// CRITICAL: Labels have cardinality cost
// CRITICAL: GPU metrics need nvidia-smi or NVML
```

---

## Implementation Blueprint

### Data Models

```go
// pkg/metrics/registry.go

var (
    // Inference metrics
    RequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Namespace: "neurogrid",
            Subsystem: "inference",
            Name:      "requests_total",
            Help:      "Total number of inference requests",
        },
        []string{"model", "status"},
    )

    RequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Namespace: "neurogrid",
            Subsystem: "inference",
            Name:      "request_duration_seconds",
            Help:      "Inference request duration in seconds",
            Buckets:   []float64{0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
        },
        []string{"model"},
    )

    TimeToFirstToken = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Namespace: "neurogrid",
            Subsystem: "inference",
            Name:      "time_to_first_token_seconds",
            Help:      "Time to first token in seconds",
            Buckets:   []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5},
        },
        []string{"model"},
    )

    TokensGenerated = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Namespace: "neurogrid",
            Subsystem: "inference",
            Name:      "tokens_generated_total",
            Help:      "Total tokens generated",
        },
        []string{"model"},
    )

    TokensPerSecond = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Namespace: "neurogrid",
            Subsystem: "inference",
            Name:      "tokens_per_second",
            Help:      "Current token generation rate",
        },
        []string{"model"},
    )

    // Batch metrics
    BatchSize = promauto.NewHistogram(
        prometheus.HistogramOpts{
            Namespace: "neurogrid",
            Subsystem: "batch",
            Name:      "size",
            Help:      "Current batch size",
            Buckets:   []float64{1, 2, 4, 8, 16, 32},
        },
    )

    QueueDepth = promauto.NewGauge(
        prometheus.GaugeOpts{
            Namespace: "neurogrid",
            Subsystem: "batch",
            Name:      "queue_depth",
            Help:      "Number of requests waiting in queue",
        },
    )

    // Cluster metrics
    PeersConnected = promauto.NewGauge(
        prometheus.GaugeOpts{
            Namespace: "neurogrid",
            Subsystem: "cluster",
            Name:      "peers_connected",
            Help:      "Number of connected peers",
        },
    )

    PeerLatency = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Namespace: "neurogrid",
            Subsystem: "cluster",
            Name:      "peer_latency_seconds",
            Help:      "Latency to peer in seconds",
            Buckets:   []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1},
        },
        []string{"peer_id"},
    )

    // GPU metrics
    GPUMemoryUsed = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Namespace: "neurogrid",
            Subsystem: "gpu",
            Name:      "memory_used_bytes",
            Help:      "GPU memory used in bytes",
        },
        []string{"device"},
    )

    GPUMemoryTotal = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Namespace: "neurogrid",
            Subsystem: "gpu",
            Name:      "memory_total_bytes",
            Help:      "GPU memory total in bytes",
        },
        []string{"device"},
    )

    GPUUtilization = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Namespace: "neurogrid",
            Subsystem: "gpu",
            Name:      "utilization_percent",
            Help:      "GPU utilization percentage",
        },
        []string{"device"},
    )
)
```

### Task List

```yaml
Task 1: Add Prometheus dependency
  MODIFY go.mod:
    - ADD: github.com/prometheus/client_golang v1.19.0
  RUN: go mod tidy

Task 2: Create metrics registry
  CREATE pkg/metrics/registry.go:
    - Define all metric variables
    - Helper functions for recording

Task 3: Create inference metrics
  CREATE pkg/metrics/inference.go:
    - RecordRequest function
    - RecordTokens function
    - RecordLatency function

Task 4: Create cluster metrics
  CREATE pkg/metrics/cluster.go:
    - RecordPeerConnect/Disconnect
    - RecordPeerLatency

Task 5: Create GPU metrics collector
  CREATE pkg/metrics/gpu.go:
    - GPUCollector implementing prometheus.Collector
    - NVML or nvidia-smi based collection

Task 6: Add metrics endpoint
  MODIFY api/server.go:
    - Add /metrics handler
    - Use promhttp.Handler()

Task 7: Add metrics middleware
  MODIFY api/middleware.go:
    - Wrap handlers with timing
    - Record request metrics

Task 8: Instrument inference engine
  MODIFY pkg/inference/engine.go:
    - Record request start/end
    - Record token counts
    - Record batch sizes

Task 9: Create Grafana dashboard
  CREATE configs/grafana/neurogrid.json:
    - Inference performance panels
    - Cluster health panels
    - GPU utilization panels

Task 10: Create alerting rules
  CREATE configs/alertmanager/rules.yaml:
    - High latency alert
    - Queue depth alert
    - GPU memory alert
    - Peer disconnect alert

Task 11: Add tests
  CREATE tests/metrics/metrics_test.go:
    - Test metric recording
    - Test /metrics endpoint format
```

### Per-Task Pseudocode

```go
// Task 3: Inference metrics helpers
func RecordRequest(model string, duration time.Duration, status string, tokens int) {
    RequestsTotal.WithLabelValues(model, status).Inc()
    RequestDuration.WithLabelValues(model).Observe(duration.Seconds())
    TokensGenerated.WithLabelValues(model).Add(float64(tokens))
}

func RecordTimeToFirstToken(model string, duration time.Duration) {
    TimeToFirstToken.WithLabelValues(model).Observe(duration.Seconds())
}

func RecordBatchMetrics(batchSize int, queueDepth int) {
    BatchSize.Observe(float64(batchSize))
    QueueDepth.Set(float64(queueDepth))
}

// Task 5: GPU metrics collector
type GPUCollector struct {
    memUsed  *prometheus.Desc
    memTotal *prometheus.Desc
    util     *prometheus.Desc
}

func NewGPUCollector() *GPUCollector {
    return &GPUCollector{
        memUsed: prometheus.NewDesc(
            "neurogrid_gpu_memory_used_bytes",
            "GPU memory used",
            []string{"device"}, nil,
        ),
        // ... other descriptors
    }
}

func (c *GPUCollector) Collect(ch chan<- prometheus.Metric) {
    // Query nvidia-smi or NVML
    gpus := queryGPUs()
    for _, gpu := range gpus {
        ch <- prometheus.MustNewConstMetric(
            c.memUsed,
            prometheus.GaugeValue,
            float64(gpu.MemUsed),
            gpu.DeviceID,
        )
        // ... other metrics
    }
}

func queryGPUs() []GPUInfo {
    // Option 1: nvidia-smi
    cmd := exec.Command("nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits")
    output, _ := cmd.Output()
    // Parse CSV output

    // Option 2: NVML bindings (more efficient)
    // Use github.com/NVIDIA/go-nvml
}

// Task 6: Metrics endpoint
func (s *Server) setupRoutes() {
    mux := http.NewServeMux()

    // Existing routes
    mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
    mux.HandleFunc("/v1/models", s.handleListModels)
    mux.HandleFunc("/health", s.handleHealth)

    // Metrics endpoint
    mux.Handle("/metrics", promhttp.Handler())

    s.handler = mux
}

// Task 7: Metrics middleware
func metricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        // Wrap response writer to capture status
        wrapped := &statusResponseWriter{ResponseWriter: w, status: 200}

        next.ServeHTTP(wrapped, r)

        // Record metrics
        duration := time.Since(start)
        HTTPRequestDuration.WithLabelValues(
            r.Method,
            r.URL.Path,
            strconv.Itoa(wrapped.status),
        ).Observe(duration.Seconds())
    })
}
```

### Grafana Dashboard JSON (Task 9)

```json
{
  "title": "NeuroGrid Inference Engine",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(neurogrid_inference_requests_total[5m])",
          "legendFormat": "{{model}} - {{status}}"
        }
      ]
    },
    {
      "title": "Latency Percentiles",
      "type": "graph",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(neurogrid_inference_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p50"
        },
        {
          "expr": "histogram_quantile(0.90, rate(neurogrid_inference_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p90"
        },
        {
          "expr": "histogram_quantile(0.99, rate(neurogrid_inference_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p99"
        }
      ]
    },
    {
      "title": "Tokens/Second",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(neurogrid_inference_tokens_per_second)"
        }
      ]
    },
    {
      "title": "GPU Memory",
      "type": "gauge",
      "targets": [
        {
          "expr": "neurogrid_gpu_memory_used_bytes / neurogrid_gpu_memory_total_bytes * 100",
          "legendFormat": "GPU {{device}}"
        }
      ]
    },
    {
      "title": "Batch Size Distribution",
      "type": "heatmap",
      "targets": [
        {
          "expr": "rate(neurogrid_batch_size_bucket[5m])"
        }
      ]
    },
    {
      "title": "Queue Depth",
      "type": "graph",
      "targets": [
        {
          "expr": "neurogrid_batch_queue_depth"
        }
      ]
    },
    {
      "title": "Connected Peers",
      "type": "stat",
      "targets": [
        {
          "expr": "neurogrid_cluster_peers_connected"
        }
      ]
    }
  ]
}
```

### Alert Rules (Task 10)

```yaml
groups:
  - name: neurogrid
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(neurogrid_inference_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "p99 latency is {{ $value }}s"

      - alert: QueueBacklog
        expr: neurogrid_batch_queue_depth > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Request queue backlog"
          description: "Queue depth is {{ $value }}"

      - alert: GPUMemoryHigh
        expr: neurogrid_gpu_memory_used_bytes / neurogrid_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory critical"
          description: "GPU {{ $labels.device }} memory at {{ $value | humanizePercentage }}"

      - alert: PeerDisconnected
        expr: changes(neurogrid_cluster_peers_connected[5m]) < 0
        labels:
          severity: warning
        annotations:
          summary: "Peer disconnected"
```

### Integration Points

```yaml
API:
  - modify: api/server.go
  - add: /metrics endpoint with promhttp.Handler()

MIDDLEWARE:
  - modify: api/middleware.go
  - add: Metrics recording middleware

INFERENCE:
  - modify: pkg/inference/engine.go
  - add: Metric recording calls

CONFIG:
  - add: configs/grafana/neurogrid.json
  - add: configs/alertmanager/rules.yaml
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./pkg/metrics/...
go vet ./pkg/metrics/...
golangci-lint run ./pkg/metrics/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/metrics/...

# Expected: All tests pass
```

### Level 3: Integration Test

```bash
# Start server
./build/neurogrid &

# Verify metrics endpoint
curl http://localhost:8080/metrics | grep neurogrid

# Expected: Prometheus format metrics
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./pkg/metrics/...`
- [ ] No linting errors: `golangci-lint run ./pkg/metrics/...`
- [ ] /metrics endpoint returns valid Prometheus format
- [ ] Latency histograms have appropriate buckets
- [ ] GPU metrics collected (if available)
- [ ] Grafana dashboard imports successfully
- [ ] Alert rules are valid

---

## Anti-Patterns to Avoid

- ❌ Don't use high-cardinality labels (user IDs, request IDs)
- ❌ Don't collect metrics too frequently (GPU polling)
- ❌ Don't forget to initialize metrics
- ❌ Don't expose sensitive data in labels
- ❌ Don't use summary instead of histogram

---

## Dependencies

```go
// go.mod additions
require (
    github.com/prometheus/client_golang v1.19.0
)
```

---

**PRP Confidence Score: 9/10**

**Rationale**:
- +3: Well-documented Prometheus client
- +2: Standard metrics patterns
- +2: Clear integration points
- +2: Grafana dashboard is JSON template
- -1: GPU metrics require nvidia-smi or NVML

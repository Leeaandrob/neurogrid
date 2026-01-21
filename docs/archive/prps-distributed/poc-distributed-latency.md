# POC: Distributed Inference Latency Validation

## Document Information

| Field | Value |
|-------|-------|
| **POC Name** | Distributed Latency Benchmark |
| **Type** | Technical Validation |
| **Duration** | 3-5 days |
| **Risk Mitigation** | Validates core assumptions before full implementation |

## Executive Summary

### Problem Statement

Before investing 8-10 weeks in the full distributed inference implementation, we need empirical evidence that:

1. **libp2p can transfer large tensors efficiently** - 16MB activations per layer
2. **Serialization overhead is acceptable** - Not a hidden bottleneck
3. **Pipeline latency is viable** - 5-hop inference completes in reasonable time
4. **The architecture makes sense** - Not building on flawed assumptions

### Success Criteria

```yaml
# Test sizes based on realistic workloads:
# - 8KB:   Single token generation (batch=1, seq=1)
# - 16MB:  Prefill typical (batch=1, seq=2048, Llama 7B)
# - 64MB:  Batch inference (batch=4, seq=2048, Llama 7B)
# - 256MB: Large batch (batch=8, seq=2048, Llama 70B)
# - 512MB: Stress test (batch=32, seq=2048, Llama 7B)

must_pass:
  # Single token (8KB) - This is the steady-state generation
  - "Single token hop latency < 5ms localhost"
  - "Single token pipeline (5 hops) < 30ms localhost"

  # Prefill (16MB) - Initial prompt processing
  - "Prefill hop latency (16MB) < 50ms localhost"
  - "Prefill hop latency (16MB) < 100ms LAN"
  - "Prefill pipeline (5 hops) < 500ms localhost"

  # Batch inference (64MB)
  - "Batch hop latency (64MB) < 200ms localhost"
  - "Batch pipeline (5 hops) < 1.5s localhost"

  # General
  - "Serialization overhead < 5ms per tensor (any size)"
  - "Sustained throughput > 50MB/s per connection"

nice_to_have:
  - "Large batch (256MB) pipeline < 3s localhost"
  - "Throughput > 100MB/s on Gigabit LAN"
  - "Throughput > 500MB/s on 10GbE"
```

### Go/No-Go Decision

| Result | Action |
|--------|--------|
| **All must_pass criteria met** | Proceed with full PRP implementation |
| **Localhost passes, LAN fails** | Investigate network config, consider RDMA/InfiniBand |
| **Serialization is bottleneck** | Use raw binary, skip FlatBuffers |
| **Pipeline latency too high** | Reconsider architecture (tensor parallel instead?) |
| **Fundamental issues** | Pivot to single-machine only or different approach |

---

## Technical Design

### Directory Structure

```
neurogrid-engine/
├── poc/                          # POC code (separate from main)
│   ├── cmd/
│   │   ├── coordinator/          # Starts pipeline, measures latency
│   │   │   └── main.go
│   │   └── worker/               # Receives, processes, forwards
│   │       └── main.go
│   ├── pkg/
│   │   ├── protocol/             # Tensor transfer protocol
│   │   │   ├── protocol.go       # Protocol definition
│   │   │   └── handler.go        # Stream handlers
│   │   ├── serialization/        # Serialization comparison
│   │   │   ├── flatbuf.go        # FlatBuffers
│   │   │   ├── raw.go            # Raw binary
│   │   │   └── proto.go          # Protobuf (comparison)
│   │   ├── metrics/              # Latency collection
│   │   │   └── collector.go
│   │   └── mock/                 # Mock layer processing
│   │       └── layer.go
│   ├── schemas/
│   │   ├── tensor.fbs            # FlatBuffers schema
│   │   └── tensor.proto          # Protobuf schema
│   ├── scripts/
│   │   ├── run_localhost.sh      # Run all 5 stages locally
│   │   ├── run_lan.sh            # Run across machines
│   │   └── analyze_results.py    # Generate report
│   └── Makefile
└── docs/
    └── poc-results/              # Results will be saved here
```

### Component 1: libp2p Host Setup

**File**: `poc/pkg/protocol/protocol.go`

```go
package protocol

import (
    "context"
    "fmt"
    "time"

    "github.com/libp2p/go-libp2p"
    "github.com/libp2p/go-libp2p/core/host"
    "github.com/libp2p/go-libp2p/core/peer"
    "github.com/libp2p/go-libp2p/p2p/discovery/mdns"
)

const (
    TensorProtocol = "/neurogrid/poc/tensor/1.0.0"
    PingProtocol   = "/neurogrid/poc/ping/1.0.0"
    ServiceTag     = "neurogrid-poc"
)

type Host struct {
    host.Host
    peerChan chan peer.AddrInfo
}

func NewHost(ctx context.Context, port int) (*Host, error) {
    h, err := libp2p.New(
        libp2p.ListenAddrStrings(
            fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", port),
            fmt.Sprintf("/ip4/0.0.0.0/udp/%d/quic-v1", port),
        ),
    )
    if err != nil {
        return nil, err
    }

    host := &Host{
        Host:     h,
        peerChan: make(chan peer.AddrInfo, 10),
    }

    // Setup mDNS discovery
    mdnsService := mdns.NewMdnsService(h, ServiceTag, host)
    if err := mdnsService.Start(); err != nil {
        return nil, err
    }

    fmt.Printf("Host started: %s\n", h.ID())
    for _, addr := range h.Addrs() {
        fmt.Printf("  Listening on: %s/p2p/%s\n", addr, h.ID())
    }

    return host, nil
}

func (h *Host) HandlePeerFound(pi peer.AddrInfo) {
    fmt.Printf("Discovered peer: %s\n", pi.ID)
    h.peerChan <- pi
}

func (h *Host) WaitForPeer(ctx context.Context, timeout time.Duration) (peer.AddrInfo, error) {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    select {
    case pi := <-h.peerChan:
        // Connect to peer
        if err := h.Connect(ctx, pi); err != nil {
            return peer.AddrInfo{}, err
        }
        return pi, nil
    case <-ctx.Done():
        return peer.AddrInfo{}, ctx.Err()
    }
}
```

### Component 2: Serialization Methods

**File**: `poc/pkg/serialization/raw.go`

```go
package serialization

import (
    "encoding/binary"
    "fmt"
)

// TensorHeader: [8 bytes size][4 bytes dtype][4 bytes ndim][ndim * 8 bytes shape]
const HeaderBaseSize = 16

type RawSerializer struct{}

type TensorData struct {
    Shape []int64
    Dtype uint32
    Data  []byte
}

func (s *RawSerializer) Serialize(t *TensorData) ([]byte, error) {
    headerSize := HeaderBaseSize + len(t.Shape)*8
    totalSize := headerSize + len(t.Data)

    buf := make([]byte, totalSize)

    // Write size
    binary.BigEndian.PutUint64(buf[0:8], uint64(len(t.Data)))
    // Write dtype
    binary.BigEndian.PutUint32(buf[8:12], t.Dtype)
    // Write ndim
    binary.BigEndian.PutUint32(buf[12:16], uint32(len(t.Shape)))
    // Write shape
    offset := 16
    for _, dim := range t.Shape {
        binary.BigEndian.PutUint64(buf[offset:offset+8], uint64(dim))
        offset += 8
    }
    // Write data
    copy(buf[offset:], t.Data)

    return buf, nil
}

func (s *RawSerializer) Deserialize(buf []byte) (*TensorData, error) {
    if len(buf) < HeaderBaseSize {
        return nil, fmt.Errorf("buffer too small")
    }

    dataSize := binary.BigEndian.Uint64(buf[0:8])
    dtype := binary.BigEndian.Uint32(buf[8:12])
    ndim := binary.BigEndian.Uint32(buf[12:16])

    shape := make([]int64, ndim)
    offset := 16
    for i := uint32(0); i < ndim; i++ {
        shape[i] = int64(binary.BigEndian.Uint64(buf[offset : offset+8]))
        offset += 8
    }

    data := buf[offset : offset+int(dataSize)]

    return &TensorData{
        Shape: shape,
        Dtype: dtype,
        Data:  data,
    }, nil
}

func (s *RawSerializer) Name() string {
    return "raw"
}
```

**File**: `poc/pkg/serialization/benchmark.go`

```go
package serialization

import (
    "fmt"
    "time"
)

type Serializer interface {
    Serialize(*TensorData) ([]byte, error)
    Deserialize([]byte) (*TensorData, error)
    Name() string
}

type BenchmarkResult struct {
    SerializerName    string
    DataSize          int
    SerializeTime     time.Duration
    DeserializeTime   time.Duration
    SerializedSize    int
    CompressionRatio  float64
}

func BenchmarkSerializer(s Serializer, tensor *TensorData, iterations int) (*BenchmarkResult, error) {
    var totalSerialize, totalDeserialize time.Duration
    var serializedSize int

    for i := 0; i < iterations; i++ {
        // Benchmark serialize
        start := time.Now()
        buf, err := s.Serialize(tensor)
        totalSerialize += time.Since(start)
        if err != nil {
            return nil, err
        }
        serializedSize = len(buf)

        // Benchmark deserialize
        start = time.Now()
        _, err = s.Deserialize(buf)
        totalDeserialize += time.Since(start)
        if err != nil {
            return nil, err
        }
    }

    return &BenchmarkResult{
        SerializerName:   s.Name(),
        DataSize:         len(tensor.Data),
        SerializeTime:    totalSerialize / time.Duration(iterations),
        DeserializeTime:  totalDeserialize / time.Duration(iterations),
        SerializedSize:   serializedSize,
        CompressionRatio: float64(len(tensor.Data)) / float64(serializedSize),
    }, nil
}

func RunSerializationBenchmark(sizes []int) {
    serializers := []Serializer{
        &RawSerializer{},
        // Add FlatBuffers and Protobuf here
    }

    fmt.Println("=== Serialization Benchmark ===")
    fmt.Printf("%-12s | %-10s | %-12s | %-12s | %-12s | %-10s\n",
               "Serializer", "Size", "Serialize", "Deserialize", "Output Size", "Ratio")
    fmt.Println(strings.Repeat("-", 80))

    for _, size := range sizes {
        // Create test tensor (simulating FP16 activations)
        tensor := &TensorData{
            Shape: []int64{1, 4096},  // Llama hidden size
            Dtype: 1,                  // FP16
            Data:  make([]byte, size),
        }
        // Fill with random-ish data
        for i := range tensor.Data {
            tensor.Data[i] = byte(i % 256)
        }

        for _, s := range serializers {
            result, err := BenchmarkSerializer(s, tensor, 100)
            if err != nil {
                fmt.Printf("Error with %s: %v\n", s.Name(), err)
                continue
            }

            fmt.Printf("%-12s | %-10s | %-12s | %-12s | %-12s | %-10.2f\n",
                       result.SerializerName,
                       formatBytes(result.DataSize),
                       result.SerializeTime,
                       result.DeserializeTime,
                       formatBytes(result.SerializedSize),
                       result.CompressionRatio)
        }
    }
}
```

### Component 3: Tensor Transfer Protocol

**File**: `poc/pkg/protocol/handler.go`

```go
package protocol

import (
    "bufio"
    "context"
    "encoding/binary"
    "fmt"
    "io"
    "time"

    "github.com/libp2p/go-libp2p/core/network"
    "github.com/libp2p/go-libp2p/core/peer"
    "github.com/neurogrid/engine/poc/pkg/metrics"
    "github.com/neurogrid/engine/poc/pkg/serialization"
)

const BufferSize = 4 * 1024 * 1024 // 4MB

type TransferHandler struct {
    host       *Host
    serializer serialization.Serializer
    metrics    *metrics.Collector
    onReceive  func(*serialization.TensorData, *TransferMetrics) (*serialization.TensorData, error)
}

type TransferMetrics struct {
    HopID           int
    SendTime        time.Time
    ReceiveTime     time.Time
    ProcessTime     time.Duration
    SerializeTime   time.Duration
    DeserializeTime time.Duration
    BytesTransferred int
}

func NewTransferHandler(h *Host, s serialization.Serializer, m *metrics.Collector) *TransferHandler {
    return &TransferHandler{
        host:       h,
        serializer: s,
        metrics:    m,
    }
}

func (h *TransferHandler) SetReceiveHandler(fn func(*serialization.TensorData, *TransferMetrics) (*serialization.TensorData, error)) {
    h.onReceive = fn
}

func (h *TransferHandler) RegisterProtocol() {
    h.host.SetStreamHandler(TensorProtocol, h.handleIncoming)
}

func (h *TransferHandler) handleIncoming(s network.Stream) {
    defer s.Close()

    metrics := &TransferMetrics{
        ReceiveTime: time.Now(),
    }

    reader := bufio.NewReaderSize(s, BufferSize)

    // Read header: [4 bytes hopID][8 bytes timestamp][4 bytes dataLen]
    header := make([]byte, 16)
    if _, err := io.ReadFull(reader, header); err != nil {
        fmt.Printf("Error reading header: %v\n", err)
        return
    }

    metrics.HopID = int(binary.BigEndian.Uint32(header[0:4]))
    metrics.SendTime = time.Unix(0, int64(binary.BigEndian.Uint64(header[4:12])))
    dataLen := binary.BigEndian.Uint32(header[12:16])
    metrics.BytesTransferred = int(dataLen)

    // Read tensor data
    data := make([]byte, dataLen)
    if _, err := io.ReadFull(reader, data); err != nil {
        fmt.Printf("Error reading data: %v\n", err)
        return
    }

    // Deserialize
    deserStart := time.Now()
    tensor, err := h.serializer.Deserialize(data)
    metrics.DeserializeTime = time.Since(deserStart)
    if err != nil {
        fmt.Printf("Error deserializing: %v\n", err)
        return
    }

    // Process (mock layer forward)
    if h.onReceive != nil {
        processStart := time.Now()
        _, err = h.onReceive(tensor, metrics)
        metrics.ProcessTime = time.Since(processStart)
        if err != nil {
            fmt.Printf("Error processing: %v\n", err)
            return
        }
    }

    // Record metrics
    h.metrics.Record(metrics)
}

func (h *TransferHandler) SendTensor(ctx context.Context, peerID peer.ID, hopID int, tensor *serialization.TensorData) (*TransferMetrics, error) {
    metrics := &TransferMetrics{
        HopID:    hopID,
        SendTime: time.Now(),
    }

    // Serialize
    serStart := time.Now()
    data, err := h.serializer.Serialize(tensor)
    metrics.SerializeTime = time.Since(serStart)
    if err != nil {
        return nil, err
    }
    metrics.BytesTransferred = len(data)

    // Open stream
    stream, err := h.host.NewStream(ctx, peerID, TensorProtocol)
    if err != nil {
        return nil, fmt.Errorf("failed to open stream: %w", err)
    }
    defer stream.Close()

    stream.SetDeadline(time.Now().Add(30 * time.Second))
    writer := bufio.NewWriterSize(stream, BufferSize)

    // Write header
    header := make([]byte, 16)
    binary.BigEndian.PutUint32(header[0:4], uint32(hopID))
    binary.BigEndian.PutUint64(header[4:12], uint64(metrics.SendTime.UnixNano()))
    binary.BigEndian.PutUint32(header[12:16], uint32(len(data)))

    if _, err := writer.Write(header); err != nil {
        return nil, err
    }
    if _, err := writer.Write(data); err != nil {
        return nil, err
    }
    if err := writer.Flush(); err != nil {
        return nil, err
    }

    return metrics, nil
}
```

### Component 4: Metrics Collector

**File**: `poc/pkg/metrics/collector.go`

```go
package metrics

import (
    "encoding/json"
    "fmt"
    "os"
    "sort"
    "sync"
    "time"
)

type HopMetrics struct {
    HopID             int           `json:"hop_id"`
    TransferLatency   time.Duration `json:"transfer_latency_ns"`
    SerializeTime     time.Duration `json:"serialize_time_ns"`
    DeserializeTime   time.Duration `json:"deserialize_time_ns"`
    ProcessTime       time.Duration `json:"process_time_ns"`
    BytesTransferred  int           `json:"bytes_transferred"`
    ThroughputMBps    float64       `json:"throughput_mbps"`
}

type PipelineRun struct {
    RunID           int           `json:"run_id"`
    TotalLatency    time.Duration `json:"total_latency_ns"`
    Hops            []HopMetrics  `json:"hops"`
    StartTime       time.Time     `json:"start_time"`
    EndTime         time.Time     `json:"end_time"`
}

type Collector struct {
    mu       sync.Mutex
    runs     []PipelineRun
    current  *PipelineRun
    hopData  []HopMetrics
}

func NewCollector() *Collector {
    return &Collector{
        runs: make([]PipelineRun, 0),
    }
}

func (c *Collector) StartRun(runID int) {
    c.mu.Lock()
    defer c.mu.Unlock()

    c.current = &PipelineRun{
        RunID:     runID,
        StartTime: time.Now(),
        Hops:      make([]HopMetrics, 0),
    }
    c.hopData = make([]HopMetrics, 0)
}

func (c *Collector) RecordHop(hopID int, transferLatency, serializeTime, deserializeTime, processTime time.Duration, bytes int) {
    c.mu.Lock()
    defer c.mu.Unlock()

    throughput := float64(bytes) / (float64(transferLatency.Nanoseconds()) / 1e9) / 1024 / 1024

    c.hopData = append(c.hopData, HopMetrics{
        HopID:            hopID,
        TransferLatency:  transferLatency,
        SerializeTime:    serializeTime,
        DeserializeTime:  deserializeTime,
        ProcessTime:      processTime,
        BytesTransferred: bytes,
        ThroughputMBps:   throughput,
    })
}

func (c *Collector) EndRun() {
    c.mu.Lock()
    defer c.mu.Unlock()

    c.current.EndTime = time.Now()
    c.current.TotalLatency = c.current.EndTime.Sub(c.current.StartTime)
    c.current.Hops = c.hopData
    c.runs = append(c.runs, *c.current)
    c.current = nil
}

func (c *Collector) GenerateReport() string {
    c.mu.Lock()
    defer c.mu.Unlock()

    if len(c.runs) == 0 {
        return "No data collected"
    }

    // Calculate statistics
    var totalLatencies []time.Duration
    hopLatencies := make(map[int][]time.Duration)
    var totalThroughput float64

    for _, run := range c.runs {
        totalLatencies = append(totalLatencies, run.TotalLatency)
        for _, hop := range run.Hops {
            hopLatencies[hop.HopID] = append(hopLatencies[hop.HopID], hop.TransferLatency)
            totalThroughput += hop.ThroughputMBps
        }
    }

    report := "=== POC Latency Benchmark Report ===\n\n"
    report += fmt.Sprintf("Total Runs: %d\n\n", len(c.runs))

    // Pipeline latency stats
    report += "--- Pipeline Total Latency ---\n"
    report += fmt.Sprintf("  P50: %v\n", percentile(totalLatencies, 50))
    report += fmt.Sprintf("  P95: %v\n", percentile(totalLatencies, 95))
    report += fmt.Sprintf("  P99: %v\n", percentile(totalLatencies, 99))
    report += fmt.Sprintf("  Min: %v\n", min(totalLatencies))
    report += fmt.Sprintf("  Max: %v\n", max(totalLatencies))
    report += "\n"

    // Per-hop latency stats
    report += "--- Per-Hop Latency ---\n"
    for hopID := 0; hopID < 5; hopID++ {
        if lats, ok := hopLatencies[hopID]; ok {
            report += fmt.Sprintf("  Hop %d:\n", hopID)
            report += fmt.Sprintf("    P50: %v\n", percentile(lats, 50))
            report += fmt.Sprintf("    P95: %v\n", percentile(lats, 95))
            report += fmt.Sprintf("    Avg: %v\n", avg(lats))
        }
    }

    // Throughput
    avgThroughput := totalThroughput / float64(len(c.runs)*5)
    report += fmt.Sprintf("\n--- Throughput ---\n")
    report += fmt.Sprintf("  Average: %.2f MB/s per hop\n", avgThroughput)

    // Pass/Fail assessment
    report += "\n--- Validation Results ---\n"
    p50Pipeline := percentile(totalLatencies, 50)

    if p50Pipeline < 500*time.Millisecond {
        report += "  ✅ Pipeline latency < 500ms: PASS\n"
    } else {
        report += "  ❌ Pipeline latency < 500ms: FAIL\n"
    }

    if avgThroughput > 50 {
        report += "  ✅ Throughput > 50MB/s: PASS\n"
    } else {
        report += "  ❌ Throughput > 50MB/s: FAIL\n"
    }

    return report
}

func (c *Collector) SaveJSON(filepath string) error {
    c.mu.Lock()
    defer c.mu.Unlock()

    data, err := json.MarshalIndent(c.runs, "", "  ")
    if err != nil {
        return err
    }
    return os.WriteFile(filepath, data, 0644)
}

// Helper functions
func percentile(durations []time.Duration, p int) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    sorted := make([]time.Duration, len(durations))
    copy(sorted, durations)
    sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
    idx := (p * len(sorted)) / 100
    if idx >= len(sorted) {
        idx = len(sorted) - 1
    }
    return sorted[idx]
}

func avg(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    var total time.Duration
    for _, d := range durations {
        total += d
    }
    return total / time.Duration(len(durations))
}

func min(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    m := durations[0]
    for _, d := range durations[1:] {
        if d < m {
            m = d
        }
    }
    return m
}

func max(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    m := durations[0]
    for _, d := range durations[1:] {
        if d > m {
            m = d
        }
    }
    return m
}
```

### Component 5: Coordinator (Test Runner)

**File**: `poc/cmd/coordinator/main.go`

```go
package main

import (
    "context"
    "flag"
    "fmt"
    "os"
    "time"

    "github.com/neurogrid/engine/poc/pkg/metrics"
    "github.com/neurogrid/engine/poc/pkg/protocol"
    "github.com/neurogrid/engine/poc/pkg/serialization"
)

var (
    port       = flag.Int("port", 9000, "Listen port")
    runs       = flag.Int("runs", 100, "Number of pipeline runs")
    tensorSize = flag.Int("size", 16*1024*1024, "Tensor size in bytes (default 16MB)")
    outputFile = flag.String("output", "results.json", "Output file for results")
)

func main() {
    flag.Parse()

    ctx := context.Background()

    // Create host
    host, err := protocol.NewHost(ctx, *port)
    if err != nil {
        fmt.Printf("Failed to create host: %v\n", err)
        os.Exit(1)
    }
    defer host.Close()

    // Wait for all 4 worker peers
    fmt.Println("Waiting for 4 worker peers...")
    peers := make([]peer.ID, 0, 4)
    for i := 0; i < 4; i++ {
        pi, err := host.WaitForPeer(ctx, 60*time.Second)
        if err != nil {
            fmt.Printf("Failed to find peer %d: %v\n", i, err)
            os.Exit(1)
        }
        peers = append(peers, pi.ID)
        fmt.Printf("Connected to peer %d: %s\n", i, pi.ID)
    }

    // Setup
    collector := metrics.NewCollector()
    serializer := &serialization.RawSerializer{}
    handler := protocol.NewTransferHandler(host, serializer, collector)

    // Create test tensor
    tensor := &serialization.TensorData{
        Shape: []int64{1, int64(*tensorSize / 2)}, // FP16 = 2 bytes per element
        Dtype: 1,
        Data:  make([]byte, *tensorSize),
    }
    // Fill with pattern
    for i := range tensor.Data {
        tensor.Data[i] = byte(i % 256)
    }

    fmt.Printf("\nStarting benchmark: %d runs, %d MB tensor\n", *runs, *tensorSize/1024/1024)
    fmt.Println("Pipeline: Coordinator -> Peer0 -> Peer1 -> Peer2 -> Peer3 -> (done)")

    // Run benchmark
    for run := 0; run < *runs; run++ {
        collector.StartRun(run)

        currentTensor := tensor
        for hop, peerID := range peers {
            transferMetrics, err := handler.SendTensor(ctx, peerID, hop, currentTensor)
            if err != nil {
                fmt.Printf("Run %d, Hop %d failed: %v\n", run, hop, err)
                break
            }

            // Record metrics
            // In real scenario, we'd wait for response from final peer
            // For POC, we measure send latency
            collector.RecordHop(
                hop,
                time.Since(transferMetrics.SendTime),
                transferMetrics.SerializeTime,
                0, // Deserialize measured on worker
                0, // Process measured on worker
                transferMetrics.BytesTransferred,
            )
        }

        collector.EndRun()

        if (run+1) % 10 == 0 {
            fmt.Printf("Completed %d/%d runs\n", run+1, *runs)
        }
    }

    // Generate report
    report := collector.GenerateReport()
    fmt.Println("\n" + report)

    // Save results
    if err := collector.SaveJSON(*outputFile); err != nil {
        fmt.Printf("Failed to save results: %v\n", err)
    } else {
        fmt.Printf("\nResults saved to %s\n", *outputFile)
    }
}
```

### Component 6: Worker

**File**: `poc/cmd/worker/main.go`

```go
package main

import (
    "context"
    "flag"
    "fmt"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/neurogrid/engine/poc/pkg/metrics"
    "github.com/neurogrid/engine/poc/pkg/protocol"
    "github.com/neurogrid/engine/poc/pkg/serialization"
)

var (
    port      = flag.Int("port", 9001, "Listen port")
    workerID  = flag.Int("id", 0, "Worker ID (0-3)")
    nextPort  = flag.Int("next", 0, "Next worker port (0 = this is last worker)")
    mockDelay = flag.Duration("delay", 1*time.Millisecond, "Mock processing delay")
)

func main() {
    flag.Parse()

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Create host
    host, err := protocol.NewHost(ctx, *port)
    if err != nil {
        fmt.Printf("Failed to create host: %v\n", err)
        os.Exit(1)
    }
    defer host.Close()

    fmt.Printf("Worker %d started on port %d\n", *workerID, *port)

    // Setup
    collector := metrics.NewCollector()
    serializer := &serialization.RawSerializer{}
    handler := protocol.NewTransferHandler(host, serializer, collector)

    // Find next peer if not last worker
    var nextPeer peer.ID
    if *nextPort > 0 {
        fmt.Printf("Waiting for next worker (port %d)...\n", *nextPort)
        pi, err := host.WaitForPeer(ctx, 60*time.Second)
        if err != nil {
            fmt.Printf("Failed to find next peer: %v\n", err)
            os.Exit(1)
        }
        nextPeer = pi.ID
        fmt.Printf("Connected to next worker: %s\n", nextPeer)
    }

    // Set receive handler
    handler.SetReceiveHandler(func(tensor *serialization.TensorData, m *protocol.TransferMetrics) (*serialization.TensorData, error) {
        fmt.Printf("Worker %d received hop %d: %d bytes\n", *workerID, m.HopID, m.BytesTransferred)

        // Mock layer processing
        time.Sleep(*mockDelay)

        // Forward to next worker if not last
        if nextPeer != "" {
            _, err := handler.SendTensor(ctx, nextPeer, m.HopID+1, tensor)
            if err != nil {
                return nil, err
            }
        } else {
            fmt.Printf("Worker %d: Pipeline complete for hop %d\n", *workerID, m.HopID)
        }

        return tensor, nil
    })

    handler.RegisterProtocol()

    // Wait for shutdown
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    <-sigCh

    fmt.Println("\nShutting down...")
}
```

---

## Test Matrix

### Tensor Sizes to Test

| ID | Size | Workload | Formula | Notes |
|----|------|----------|---------|-------|
| T1 | 8 KB | Single token | 1 × 1 × 4096 × 2 | Steady-state generation |
| T2 | 4 MB | Short prefill | 1 × 512 × 4096 × 2 | Quick prompts |
| T3 | 16 MB | Typical prefill | 1 × 2048 × 4096 × 2 | **Primary benchmark** |
| T4 | 32 MB | Long prefill | 1 × 4096 × 4096 × 2 | Max context 7B |
| T5 | 64 MB | Batch small | 4 × 2048 × 4096 × 2 | Multi-request |
| T6 | 128 MB | Batch medium | 8 × 2048 × 4096 × 2 | High throughput |
| T7 | 256 MB | Batch large | 8 × 2048 × 8192 × 2 | Llama 70B batch |
| T8 | 512 MB | Stress test | 32 × 2048 × 4096 × 2 | Limits testing |

### Network Configurations

| ID | Config | Description | Expected Bandwidth |
|----|--------|-------------|-------------------|
| N1 | Localhost | Same machine, loopback | ~10 GB/s |
| N2 | LAN 1GbE | Gigabit Ethernet | ~100 MB/s |
| N3 | LAN 10GbE | 10 Gigabit Ethernet | ~1 GB/s |
| N4 | WiFi | Wireless (best effort) | ~50 MB/s |

### Test Combinations

```
Primary Tests (must run):
├── T1 × N1 (single token localhost)     → Baseline latency
├── T3 × N1 (16MB localhost)             → Prefill localhost
├── T3 × N2 (16MB LAN 1GbE)              → Prefill over network
├── T5 × N1 (64MB localhost)             → Batch localhost
└── T5 × N2 (64MB LAN 1GbE)              → Batch over network

Extended Tests (nice to have):
├── T1 × N2 (single token LAN)           → Network overhead floor
├── T6 × N1 (128MB localhost)            → Scaling behavior
├── T7 × N1 (256MB localhost)            → Large batch
├── T8 × N1 (512MB localhost)            → Stress test
└── T3 × N3 (16MB 10GbE)                 → Fast network comparison
```

### Expected Results by Size

```
Theoretical minimums (localhost, no serialization):
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Size     │ Memory Copy  │ 5-hop Min    │ Realistic*   │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 8 KB     │ ~0.01ms      │ ~0.05ms      │ 1-5ms        │
│ 16 MB    │ ~2ms         │ ~10ms        │ 30-100ms     │
│ 64 MB    │ ~8ms         │ ~40ms        │ 100-300ms    │
│ 256 MB   │ ~32ms        │ ~160ms       │ 400-800ms    │
│ 512 MB   │ ~64ms        │ ~320ms       │ 800-1500ms   │
└──────────┴──────────────┴──────────────┴──────────────┘
* Realistic includes: serialization, Go runtime, libp2p overhead

Theoretical minimums (1GbE LAN):
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Size     │ Wire Time    │ 5-hop Min    │ Realistic    │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 8 KB     │ ~0.06ms      │ ~0.3ms       │ 2-10ms       │
│ 16 MB    │ ~130ms       │ ~650ms       │ 800-1200ms   │
│ 64 MB    │ ~520ms       │ ~2600ms      │ 3-4s         │
│ 256 MB   │ ~2080ms      │ ~10400ms     │ 12-15s       │
└──────────┴──────────────┴──────────────┴──────────────┘
Note: 1GbE is clearly a bottleneck for large tensors!
```

### Implications for Architecture

Based on theoretical analysis:

1. **Single token generation (8KB)**: Network is NOT a bottleneck
   - Even on 1GbE, 5-hop < 50ms is achievable
   - This is 99% of inference time (autoregressive generation)

2. **Prefill (16MB)**: Network becomes significant on 1GbE
   - Localhost: ~100ms total → Acceptable
   - 1GbE LAN: ~1s total → Acceptable for first token
   - Consider: Chunked transfer? Compression?

3. **Large batch (64MB+)**: 1GbE is limiting
   - May need 10GbE for production batch inference
   - Or accept longer prefill latency

**Key Insight**: Focus POC validation on:
- T1 (8KB) for steady-state performance
- T3 (16MB) for prefill acceptability
- T5 (64MB) for batch ceiling

---

## Test Scenarios

### Scenario 1: Localhost (All 5 processes same machine)

```bash
#!/bin/bash
# scripts/run_localhost.sh

echo "Starting POC Localhost Test"
echo "==========================="

# Start workers (background)
./bin/worker -id=3 -port=9004 -next=0 &
sleep 1
./bin/worker -id=2 -port=9003 -next=9004 &
sleep 1
./bin/worker -id=1 -port=9002 -next=9003 &
sleep 1
./bin/worker -id=0 -port=9001 -next=9002 &
sleep 2

# Run coordinator
./bin/coordinator -port=9000 -runs=100 -size=16777216 -output=results_localhost.json

# Cleanup
pkill -f "bin/worker"

echo "Done! Results in results_localhost.json"
```

### Scenario 2: LAN (5 processes across machines)

```bash
#!/bin/bash
# scripts/run_lan.sh

# Run on each machine:
# Machine 1 (coordinator): ./bin/coordinator -port=9000 -runs=100
# Machine 2 (worker 0):    ./bin/worker -id=0 -port=9001 -next=9002
# Machine 3 (worker 1):    ./bin/worker -id=1 -port=9002 -next=9003
# Machine 4 (worker 2):    ./bin/worker -id=2 -port=9003 -next=9004
# Machine 5 (worker 3):    ./bin/worker -id=3 -port=9004 -next=0

echo "LAN test requires manual execution on 5 machines"
echo "See comments in this script for commands"
```

### Scenario 3: Serialization Comparison

```bash
#!/bin/bash
# scripts/benchmark_serialization.sh

./bin/serialization-bench -sizes="1048576,4194304,16777216,67108864"
# 1MB, 4MB, 16MB, 64MB
```

---

## Expected Results Format

```json
{
  "test_config": {
    "tensor_size_bytes": 16777216,
    "num_hops": 5,
    "num_runs": 100,
    "serializer": "raw"
  },
  "pipeline_latency": {
    "p50_ms": 45.2,
    "p95_ms": 62.1,
    "p99_ms": 78.4,
    "min_ms": 38.1,
    "max_ms": 95.2
  },
  "per_hop_latency": {
    "hop_0": {"p50_ms": 8.5, "p95_ms": 12.3},
    "hop_1": {"p50_ms": 8.7, "p95_ms": 12.8},
    "hop_2": {"p50_ms": 8.9, "p95_ms": 13.1},
    "hop_3": {"p50_ms": 9.1, "p95_ms": 13.5},
    "hop_4": {"p50_ms": 9.0, "p95_ms": 13.2}
  },
  "throughput": {
    "avg_mbps": 187.5,
    "min_mbps": 156.2,
    "max_mbps": 210.8
  },
  "serialization": {
    "serialize_avg_us": 320,
    "deserialize_avg_us": 180
  },
  "validation": {
    "pipeline_under_500ms": true,
    "throughput_over_50mbps": true,
    "serialization_under_5ms": true
  },
  "recommendation": "PROCEED"
}
```

---

## Decision Matrix

| Result | Pipeline P50 | Throughput | Action |
|--------|-------------|------------|--------|
| **GREEN** | < 200ms | > 100MB/s | Proceed with full PRP |
| **YELLOW** | 200-500ms | 50-100MB/s | Proceed with optimizations noted |
| **ORANGE** | 500ms-1s | 25-50MB/s | Investigate bottlenecks, consider tensor parallel |
| **RED** | > 1s | < 25MB/s | Architecture pivot needed |

---

## Timeline

| Day | Deliverable |
|-----|-------------|
| 1 | Project setup, libp2p host, raw serialization |
| 2 | Transfer protocol, metrics collector |
| 3 | Coordinator and worker implementation |
| 4 | Localhost testing, FlatBuffers/Protobuf comparison |
| 5 | LAN testing, report generation, documentation |

---

## Files to Create

```
poc/
├── cmd/
│   ├── coordinator/main.go
│   ├── worker/main.go
│   └── serialization-bench/main.go
├── pkg/
│   ├── protocol/
│   │   ├── protocol.go
│   │   └── handler.go
│   ├── serialization/
│   │   ├── serializer.go
│   │   ├── raw.go
│   │   ├── flatbuf.go
│   │   └── benchmark.go
│   ├── metrics/
│   │   └── collector.go
│   └── mock/
│       └── layer.go
├── scripts/
│   ├── run_localhost.sh
│   ├── run_lan.sh
│   └── analyze_results.py
├── Makefile
├── go.mod
└── README.md
```

---

## Appendix: Quick Start Commands

```bash
# Build POC
cd poc && make build

# Run serialization benchmark
make bench-serialization

# Run localhost test (all 5 processes)
make test-localhost

# Run with specific tensor size
./bin/coordinator -size=67108864 -runs=50  # 64MB tensor

# Analyze results
python scripts/analyze_results.py results_localhost.json
```

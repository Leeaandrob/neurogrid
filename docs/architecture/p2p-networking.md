# P2P Networking Layer - Architecture Document

## Overview

The P2P Networking Layer enables distributed inference across multiple machines using libp2p. It provides peer discovery (local via mDNS, remote via DHT) and a custom tensor transfer protocol for activation exchange, weight distribution, and model configuration transfer.

## Component Diagram

```
+-----------------------------------------------------------------------+
|                     NeuroGrid P2P Layer                                |
+-----------------------------------------------------------------------+
|                                                                        |
|  +------------------------------------------------------------------+  |
|  |                        libp2p Host                                |  |
|  |  +------------+  +-------------+  +------------------------+      |  |
|  |  | TCP/QUIC   |  | Hole Punch  |  | NAT Port Mapping       |      |  |
|  |  | Transports |  | + Relay     |  |                        |      |  |
|  |  +------------+  +-------------+  +------------------------+      |  |
|  +------------------------------------------------------------------+  |
|                               |                                        |
|               +---------------+---------------+                        |
|               v               v               v                        |
|  +-----------------+ +-----------------+ +---------------------+       |
|  |    Discovery    | |    Discovery    | |     Protocol        |       |
|  |      mDNS       | |      DHT        | |   Tensor/1.0.0      |       |
|  |  (Local LAN)    | |  (Kademlia)     | |   (All Messages)    |       |
|  +--------+--------+ +--------+--------+ +---------+-----------+       |
|           |                   |                     |                  |
|           +-------------------+---------------------+                  |
|                               v                                        |
|                    +---------------------+                             |
|                    |    Peer Channel     |                             |
|                    |  (Discovered Peers) |                             |
|                    +---------------------+                             |
|                                                                        |
+------------------------------------------------------------------------+
```

## Host Configuration

The libp2p host is configured with:

| Feature | Purpose |
|---------|---------|
| TCP Transport | Reliable connections (port N) |
| QUIC Transport | Fast UDP-based connections (port N) |
| Hole Punching | NAT traversal for direct connections |
| Relay | Fallback when direct connection fails |
| NAT Port Map | UPnP/NAT-PMP for automatic port forwarding |

## Discovery Mechanisms

### mDNS (Local Discovery)

- **Protocol**: Multicast DNS with service tag `_neurogrid._tcp`
- **Scope**: Local network only
- **Latency**: Near-instant discovery
- **Use case**: Multi-GPU servers on same LAN

```go
discovery := p2p.NewDiscovery(host)
discovery.SetupMDNS()

// Receive discovered peers
peer := <-discovery.PeerChan()
host.Connect(ctx, peer)
```

### DHT (Remote Discovery)

- **Protocol**: Kademlia DHT in auto-server mode
- **Scope**: Global (via bootstrap nodes)
- **Latency**: Seconds to minutes for initial bootstrap
- **Use case**: Geographically distributed inference

```go
discovery := p2p.NewDiscovery(host)
discovery.SetupDHT(ctx)

// Access DHT for advanced operations
dht := discovery.DHT()
```

## Tensor Protocol

### Protocol ID

```
/neurogrid/tensor/1.0.0
```

### Message Types

The protocol supports multiple message types for different operations:

| Type | Value | Purpose | Format |
|------|-------|---------|--------|
| MsgTypeActivation | 0x01 | Forward activation to remote peer | Extended (25B) |
| MsgTypeResponse | 0x02 | Return computed hidden state | Extended (25B) |
| MsgTypeWeights | 0x03 | Transfer layer weights (chunked) | Extended (25B) |
| MsgTypeWeightsAck | 0x04 | Acknowledge weights received | Extended (25B) |
| MsgTypeLayerStatus | 0x05 | Worker reports loaded layers | Extended (25B) |
| MsgTypeLayerRequest | 0x06 | Coordinator requests layers | Extended (25B) |
| **MsgTypeModelConfig** | **0x07** | **Transfer model config to stateless worker** | **Extended (25B)** |
| MsgTypeTracedActivation | 0x11 | Activation with trace context | Traced (50B) |
| MsgTypeTracedResponse | 0x12 | Response with trace context | Traced (50B) |

### Header Formats

#### Legacy Header (16 bytes)

```
+----------+----------+----------+-----------------+
| LayerID  |  SeqID   | DataLen  |    Data         |
| (4 bytes)| (8 bytes)| (4 bytes)|  (variable)     |
+----------+----------+----------+-----------------+
```

#### Extended Header (25 bytes)

```
+--------+----------+----------+------------+----------+-----------------+
| MsgType| LayerID  |  SeqID   | RequestID  | DataLen  |    Data         |
| (1 byte)| (4 bytes)| (8 bytes)| (8 bytes) | (4 bytes)|  (variable)     |
+--------+----------+----------+------------+----------+-----------------+
```

#### Traced Header (50 bytes)

```
+------------------+----------+----------+-------------+
| Extended Header  | TraceID  |  SpanID  | TraceFlags  |
| (25 bytes)       | (16 bytes)| (8 bytes)| (1 byte)   |
+------------------+----------+----------+-------------+
```

### Field Descriptions

| Field | Size | Description |
|-------|------|-------------|
| MsgType | 1 byte | Message type (0x01-0x12) |
| LayerID | 4 bytes | Source/destination layer in the model |
| SeqID | 8 bytes | Sequence number for ordering |
| RequestID | 8 bytes | Request ID for response matching |
| DataLen | 4 bytes | Length of tensor data in bytes |
| Data | Variable | Payload (tensor bytes, JSON config, etc.) |

### API Reference

```go
// Create protocol handler
protocol := p2p.NewProtocol(host)

// Register receive callbacks
protocol.OnActivationReceived(func(msg *p2p.TensorMessage) {
    // Process activation tensor
})

protocol.OnWeightsReceived(func(layerID, chunkIndex, totalChunks int, data []byte) {
    // Process weight chunk
})

protocol.OnModelConfigReceived(func(config []byte, from peer.ID) {
    // Process model configuration (JSON)
    llamaConfig, modelName, _ := inference.DeserializeConfig(config)
})

// Send messages
err := protocol.SendActivation(ctx, peerID, layerID, seqID, requestID, data)
err := protocol.SendWeights(ctx, peerID, layerID, weightData)
err := protocol.SendModelConfig(ctx, peerID, configJSON)
```

## Stateless Worker Support

### Overview

Stateless workers start without local model files (`--model` flag omitted) and receive both configuration and weights from the coordinator over P2P.

### Worker Modes

| Mode | Startup | Config Source | Weight Source |
|------|---------|---------------|---------------|
| Stateful | `--model /path` | Local model files | Local SafeTensors |
| Stateless | No `--model` | Coordinator (0x07) | Coordinator (0x03) |

### Stateless Worker Flow

```
Coordinator                           Worker (Stateless)
    |                                    |
    |<-- Connect (mDNS/DHT) ------------|
    |                                    |
    |--- MsgTypeLayerRequest (empty) -->|
    |<-- MsgTypeLayerStatus ([]) --------|
    |                                    |
    |    (Worker has no local layers)    |
    |                                    |
    |--- MsgTypeModelConfig (JSON) ---->|  handleModelConfig()
    |    (wait 100ms)                    |  sets worker.modelConfig
    |                                    |
    |--- MsgTypeWeights (chunked) ----->|  handleWeights()
    |                                    |  uses modelConfig
    |                                    |  uploads to GPU
    |<-- MsgTypeWeightsAck -------------|
    |                                    |
    |    Worker ready for inference      |
```

### Config Tracking

The coordinator tracks which peers have received config to prevent duplicate sends:

```go
type DistributedInferenceCoordinator struct {
    // ...
    configSent map[string]bool // Track which peers received config
    modelName  string          // Model name for config serialization
}
```

### Worker Handler Registration

```go
// In Worker.Start()
w.protocol.OnModelConfigReceived(w.handleModelConfig)

// handleModelConfig sets modelConfig for GPU operations
func (w *Worker) handleModelConfig(data []byte, from peer.ID) {
    config, modelName, err := inference.DeserializeConfig(data)
    if err != nil {
        log.Printf("Error deserializing model config: %v", err)
        return
    }
    w.modelConfig = config
    log.Printf("Received model config: %s", modelName)
}
```

### Weight Handling with Config

The `handleWeights` function uses `modelConfig` for GPU upload:

```go
func (w *Worker) handleWeights(layerID int, chunkIndex int, totalChunks int, data []byte) {
    // ... chunk accumulation ...

    if w.modelConfig != nil {
        gpuWeights, err := bindings.CreateLayerWeightsFromHost(
            weights.QWeight, weights.KWeight, /* ... */,
            w.modelConfig, // Required for GPU allocation
        )
        w.gpuWeights[layerID] = gpuWeights
    }
}
```

## Model Config Transfer

For stateless workers (no local model files), the coordinator sends model configuration before weights:

```
Coordinator                           Worker
    |                                    |
    |--- MsgTypeLayerRequest (empty) --> |
    |<-- MsgTypeLayerStatus ([]) --------|
    |                                    |
    |--- MsgTypeModelConfig (JSON) ----> |  ~450 bytes
    |    (wait 100ms)                    |
    |--- MsgTypeWeights (chunked) -----> |  ~14GB for Mistral 7B
    |<-- MsgTypeWeightsAck --------------|
    |                                    |
```

### Config Payload (JSON)

```json
{
  "model_name": "mistral-7b-instruct",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_layers": 32,
  "num_heads": 32,
  "num_kv_heads": 8,
  "head_dim": 128,
  "vocab_size": 32000,
  "max_seq_len": 4096,
  "rms_norm_eps": 1e-6
}
```

## Data Flow

```
+-------------------------------------------------------------+
|                   Tensor Transfer Flow                       |
+-------------------------------------------------------------+
|                                                              |
|  Node A (Sender)                 Node B (Receiver)          |
|  -----------------              ------------------          |
|                                                              |
|  +-----------------+           +-----------------+          |
|  | SendActivation()|           | OnActivation    |          |
|  | - layerID: 5    |           | callback        |          |
|  | - seqID: 100    |           +--------^--------+          |
|  | - data: [...]   |                    |                    |
|  +--------+--------+                    |                    |
|           |                             |                    |
|           v                             |                    |
|  +-----------------+                    |                    |
|  | NewStream()     |                    |                    |
|  | to peerID       |                    |                    |
|  +--------+--------+                    |                    |
|           |                             |                    |
|           v                             |                    |
|  +-----------------+           +--------+--------+          |
|  | Write Header    |---------->| Read Header     |          |
|  | (25 bytes)      |  network  | DecodeHeader()  |          |
|  +--------+--------+           +--------+--------+          |
|           |                             |                    |
|           v                             v                    |
|  +-----------------+           +-----------------+          |
|  | Write Data      |---------->| Read Data       |          |
|  | (dataLen bytes) |  network  | io.ReadFull()   |          |
|  +-----------------+           +--------+--------+          |
|                                         |                    |
|                                         v                    |
|                                +-----------------+          |
|                                | TensorMessage   |          |
|                                | - MsgType: 0x01 |          |
|                                | - LayerID: 5    |          |
|                                | - SeqID: 100    |          |
|                                | - Data: [...]   |          |
|                                | - From: nodeA   |          |
|                                +-----------------+          |
|                                                              |
+--------------------------------------------------------------+
```

## Thread Safety

| Component | Synchronization | Notes |
|-----------|-----------------|-------|
| Discovery | sync.Mutex | Protects mDNS and DHT instances |
| Protocol | sync.RWMutex | Protects all handler callbacks |
| PeerChan | Buffered channel | Non-blocking sends (100 capacity) |
| pendingResponses | sync.RWMutex | Request-response correlation |
| pendingWeightsAck | sync.RWMutex | Weight transfer acknowledgments |
| pendingLayerStatus | sync.RWMutex | Layer status queries |
| configSent | sync.RWMutex | Coordinator config tracking |
| chunkBuffers | sync.Mutex | Worker chunk accumulation |

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| Failed to open stream | Peer unreachable | Retry with backoff, check connectivity |
| Failed to write header | Stream closed | Retry connection |
| Failed to write data | Network error | Retry or failover to another peer |
| ErrResponseTimeout | Response not received | Retry request or check peer health |
| Config deserialization | Malformed JSON | Log error, worker stays in waiting state |
| Weight upload without config | Config not received | Log error, skip GPU upload |

## Files

| File | Description |
|------|-------------|
| `p2p/host.go` | libp2p host creation and utilities |
| `p2p/discovery.go` | mDNS and DHT discovery |
| `p2p/protocol.go` | Tensor transfer protocol (all message types) |
| `pkg/inference/config_transfer.go` | Model config serialization |
| `pkg/inference/coordinator.go` | Config sending, configSent tracking |
| `cmd/worker/main.go` | Worker handlers, handleModelConfig |
| `tests/e2e/p2p_test.go` | E2E test suite |

## Integration with Transport Layer

The P2P networking layer integrates with the Transport layer through `P2PTransport`:

```go
// P2PTransport implements Transport interface
type P2PTransport struct {
    protocol *p2p.Protocol
    peerID   peer.ID
}

func (t *P2PTransport) SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error {
    return t.protocol.SendActivation(ctx, t.peerID, layerID, seqID, requestID, data)
}

func (t *P2PTransport) SendModelConfig(ctx context.Context, config []byte) error {
    return t.protocol.SendModelConfig(ctx, t.peerID, config)
}
```

## Related Documentation

- [ADR-006: Config Transfer Protocol](decisions/ADR-006-config-transfer-protocol.md)
- [C4 Component Diagram](diagrams/c4-component-distributed-inference.md)
- [Sequence Diagram: Config Transfer](diagrams/sequence-config-transfer.md)
- [Sequence Diagram: Worker Modes](diagrams/sequence-worker-modes.md)
- [Data Flow: Config Transfer](diagrams/data-flow-config-transfer.md)

## Related Tasks

- TASK-009: libp2p Host Setup
- TASK-010: mDNS Local Discovery
- TASK-011: DHT Remote Discovery
- TASK-012: Tensor Protocol
- TASK-013: P2P Transport Implementation
- PRP: Hybrid Distributed Inference (config transfer)

---

*Updated 2025-01-24 - Added stateless worker support (Phase 3-5 implementation)*

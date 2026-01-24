# ADR-006: Config Transfer Protocol for Stateless Workers

## Status

Accepted

## Date

2025-01-23

## Context

The NeuroGrid distributed inference system required support for stateless workers - workers that start without local model files and receive configuration and weights from the coordinator. The existing P2P protocol only supported weight transfer (MsgTypeWeights), which assumed workers already knew the model configuration from local files.

### Problem Statement

When a worker starts without a `--model` flag:
1. The worker has no `LlamaConfig` to configure GPU memory allocation
2. Weight chunks cannot be uploaded to GPU without knowing layer dimensions
3. The `handleWeights()` function failed with "GPU weights not available"

### Requirements

1. Transfer model configuration from coordinator to stateless workers
2. Configuration must arrive BEFORE weight chunks
3. Maintain backward compatibility with workers that have local models
4. Configuration payload must be human-readable for debugging
5. Single-message transfer (config is small, < 1KB)

### Decision Drivers

- Protocol already supports multiple message types (0x01-0x06)
- Existing weight transfer uses chunked binary for large payloads
- Model config is small (~500 bytes) and benefits from debuggability
- Workers need config to allocate GPU memory before receiving weights

## Decision

We implement a dedicated message type `MsgTypeModelConfig` (0x07) with JSON serialization for model configuration transfer. The coordinator sends this message before any weight chunks to stateless workers.

### Architecture

```
Coordinator                           Worker (stateless)
    |                                      |
    |--- SendModelConfig (0x07) --------> |
    |                                      | handleModelConfig()
    |                                      | sets worker.modelConfig
    |                                      |
    |--- SendWeights (0x03) ------------> |
    |                                      | handleWeights()
    |                                      | uploads to GPU (needs config)
```

### Key Components

1. **TransferableConfig** (`pkg/inference/config_transfer.go`)
   - JSON-serializable struct mirroring `types.LlamaConfig`
   - Includes `model_name` for identification
   - Size < 1KB (fits in single P2P message)

2. **MsgTypeModelConfig** (`p2p/protocol.go`)
   - Message type constant: `0x07`
   - Handler: `OnModelConfigReceived(handler ModelConfigHandler)`
   - Sender: `SendModelConfig(ctx, peerID, configData)`

3. **Serialization Functions**
   - `SerializeConfig(config *LlamaConfig, modelName string) ([]byte, error)`
   - `DeserializeConfig(data []byte) (*LlamaConfig, string, error)`

### Wire Format

Config is sent as a single extended message (not chunked):

```
[MsgType=0x07][LayerID=0][SeqID=0][RequestID=0][DataLen][JSON payload]
```

Example JSON payload (~450 bytes):
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

## Alternatives Considered

### Alternative 1: Binary Serialization (Protocol Buffers)

**Pros**:
- Smaller payload (~200 bytes vs ~450 bytes JSON)
- Faster serialization/deserialization
- Schema versioning

**Cons**:
- Requires protobuf dependency
- Binary not human-readable for debugging
- Overkill for < 1KB payload

**Why rejected**: JSON is simpler, debuggable, and size difference is negligible for this use case.

### Alternative 2: Embed Config in First Weight Chunk

**Pros**:
- No new message type needed
- Atomic config+weights transfer

**Cons**:
- Breaks existing weight chunk format
- Complicates chunk parsing logic
- Config mixed with binary data

**Why rejected**: Violates separation of concerns; config and weights are distinct concepts.

### Alternative 3: Request-Response Pattern

**Pros**:
- Worker explicitly requests config when ready
- More control over timing

**Cons**:
- Additional round-trip latency
- More complex state machine
- Worker must know coordinator ID

**Why rejected**: Coordinator already knows when to send config (before weights). Push model is simpler.

## Consequences

### Positive

- Stateless workers can join cluster without local model files
- Clean separation between config and weight transfer
- JSON format enables easy debugging and logging
- Minimal protocol changes (one new message type)
- Backward compatible - workers with local models unaffected

### Negative

- JSON slightly larger than binary (~2x)
- Coordinator must track which peers received config
- Brief timing dependency (config must arrive before weights)

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Config arrives after weights | Low | High | Coordinator waits 100ms after config before sending weights |
| JSON parsing errors | Low | Medium | Explicit error types (ErrNullConfig, ErrEmptyData) |
| Config schema mismatch | Low | Medium | All fields explicit, no optional fields |

## Implementation Notes

### Phase 1-2: Protocol Layer + Serialization

**Files Created:**

- `pkg/inference/config_transfer.go` (107 lines)
  - TransferableConfig struct
  - FromLlamaConfig/ToLlamaConfig conversion
  - SerializeConfig/DeserializeConfig functions
  - Custom error types (ErrNilConfig, ErrEmptyModelName, ErrEmptyData, ErrNullConfig)

**Files Modified:**

- `p2p/protocol.go`
  - Added `MsgTypeModelConfig = 0x07` constant
  - Added `ModelConfigHandler` type
  - Added `modelConfigHandler` field to Protocol struct
  - Added `OnModelConfigReceived()` method
  - Added `SendModelConfig()` method
  - Added case in `handleExtendedMessage()` for MsgTypeModelConfig

### Phase 3-5: Worker + Coordinator Integration

**Files Modified:**

- `pkg/inference/coordinator.go`
  - Added `configSent map[string]bool` field - tracks which peers received config
  - Added `modelName string` field - for config serialization
  - Added `sendConfigToPeer(peerID peer.ID) bool` function - extracted config sending logic
  - Added `shortPeerID(peerID string) string` helper - truncates peer IDs for logging
  - Modified `distributeWeightsToPeer()` - calls `sendConfigToPeer()` before weights
  - Modified `NewDistributedInferenceCoordinator()` - initializes configSent map and modelName

- `cmd/worker/main.go`
  - Added `handleModelConfig(data []byte, from peer.ID)` handler
  - Registered `OnModelConfigReceived` in `Start()` after other handlers
  - Uses `inference.DeserializeConfig()` to convert JSON to LlamaConfig
  - Sets `worker.modelConfig` for use by `handleWeights()`

### Tests

- `pkg/inference/config_transfer_test.go`
  - TestSerializeConfig_RoundTrip
  - TestDeserializeConfig_AllFields
  - TestSerializeConfig_NilConfig
  - TestDeserializeConfig_EmptyData

### Usage Example

```go
// Coordinator sending config (in sendConfigToPeer)
configData, err := inference.SerializeConfig(dic.config, dic.modelName)
if err != nil {
    return false
}
err = dic.protocol.SendModelConfig(ctx, peerID, configData)

// Worker receiving config (handleModelConfig)
func (w *Worker) handleModelConfig(data []byte, from peer.ID) {
    config, modelName, err := inference.DeserializeConfig(data)
    if err != nil {
        log.Printf("Error deserializing model config: %v", err)
        return
    }
    w.modelConfig = config
    log.Printf("Received model config: %s (%d layers, hidden=%d)",
        modelName, config.NumLayers, config.HiddenSize)
}
```

## References

- [PRP: Hybrid Distributed Inference System](../../prps/distributed-inference-hybrid.md)
- [P2P Networking Architecture](../p2p-networking.md)
- [ADR-005: Go-CUDA Weight Bridge](ADR-005-go-cuda-weight-bridge.md)
- [Sequence Diagram: Config Transfer](../diagrams/sequence-config-transfer.md)
- [Sequence Diagram: Worker Modes](../diagrams/sequence-worker-modes.md)
- [C4 Component Diagram](../diagrams/c4-component-distributed-inference.md)

---

*Updated 2025-01-24 - Added Phase 3-5 implementation notes*

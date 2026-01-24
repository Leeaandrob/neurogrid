# Sequence Diagram: Config Transfer for Stateless Workers

## Overview

This diagram shows the complete flow when a stateless worker (without local model files) joins the distributed inference cluster. The coordinator detects the worker lacks configuration and sends it before weight transfer.

## Primary Flow: Stateless Worker Initialization

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker (Stateless)
    participant P as P2P Protocol
    participant C as Coordinator
    participant D as Weight Distributor

    Note over W: Worker starts without --model flag
    W->>W: Initialize P2P host
    W->>P: OnModelConfigReceived(handler)
    W->>P: OnWeightsReceived(handler)
    W->>P: OnLayerRequestReceived(handler)

    Note over W,C: Discovery & Connection
    W-->>C: mDNS/DHT discovery
    C->>W: Connect (libp2p)

    Note over C: Coordinator queries worker status
    C->>P: RequestLayerStatus(workerID)
    P->>W: MsgTypeLayerRequest (empty)
    W->>P: SendLayerStatus([])
    P->>C: layers = [] (no local layers)

    Note over C: Coordinator detects stateless worker
    C->>C: Check configSent[workerID]

    alt Worker needs config
        C->>D: SerializeConfig(config, modelName)
        D-->>C: JSON bytes (~450 bytes)
        C->>P: SendModelConfig(workerID, data)
        P->>W: MsgTypeModelConfig (0x07)
        W->>W: DeserializeConfig(data)
        W->>W: worker.modelConfig = config
        Note over W: Ready to receive weights
        C->>C: configSent[workerID] = true
        C->>C: Sleep(100ms)
    end

    Note over C,W: Weight transfer begins
    loop For each assigned layer
        C->>D: GetLayerWeights(layerID)
        D-->>C: weight bytes
        C->>P: SendWeights(workerID, layerID, data)
        P->>W: MsgTypeWeights (0x03, chunked)
        W->>W: handleWeights()
        W->>W: Upload to GPU (uses modelConfig)
        W->>P: SendWeightsAck(coordinatorID, layerID)
        P->>C: MsgTypeWeightsAck (0x04)
    end

    Note over W: Worker ready for inference
    W->>W: Set isReady = true
```

## Worker Mode Comparison

```mermaid
sequenceDiagram
    autonumber
    participant CW as Worker (Cached/Stateful)
    participant SW as Worker (Stateless)
    participant C as Coordinator

    Note over CW,SW: Startup Phase

    rect rgb(200, 230, 200)
        Note over CW: Stateful Mode (--model flag)
        CW->>CW: loadLocalWeights()
        CW->>CW: Parse config from model
        CW->>CW: Upload to GPU
        CW->>C: SendLayerStatus([0,1,2...])
        Note over CW: Ready immediately
    end

    rect rgb(200, 200, 230)
        Note over SW: Stateless Mode (no --model)
        SW->>SW: Initialize empty
        SW->>C: SendLayerStatus([])
        C->>SW: MsgTypeModelConfig
        SW->>SW: Set modelConfig
        C->>SW: MsgTypeWeights (chunked)
        SW->>SW: handleWeights() with config
        SW->>SW: Upload to GPU
        Note over SW: Ready after transfer
    end
```

## Weight Handling with Config

```mermaid
sequenceDiagram
    autonumber
    participant C as Coordinator
    participant W as Worker
    participant GPU as GPU Memory

    Note over W: handleWeights() flow

    C->>W: Weight chunk 1/N
    W->>W: Store in chunkBuffer[layerID]

    C->>W: Weight chunk 2/N
    W->>W: Append to chunkBuffer

    C->>W: Weight chunk N/N
    W->>W: Concatenate all chunks

    alt modelConfig is set
        W->>W: DeserializeLayerWeights()
        W->>GPU: CreateLayerWeightsFromHost()
        W->>GPU: Store in gpuWeights[layerID]
        W->>W: Initialize KV cache
        W->>W: weightsReady = true
    else modelConfig is nil
        W->>W: Log "GPU weights not available"
        Note over W: Cannot upload without config
    end

    W->>C: SendWeightsAck(layerID)
```

## Steps

| Step | Actor | Action | Description |
|------|-------|--------|-------------|
| 1 | Worker | Initialize P2P host | Create libp2p node without model files |
| 2-4 | Worker | Register handlers | Set up callbacks for config, weights, layer requests |
| 5-6 | Both | Discovery | Peer discovery via mDNS or DHT |
| 7-10 | Coordinator | Query layer status | Check what layers worker has locally |
| 11-17 | Coordinator | Send config | If worker has no layers, send model config |
| 18-24 | Both | Weight transfer | Send layer weights with chunking |
| 25 | Worker | Ready state | Worker can now process inference requests |

## Error Scenarios

| Scenario | Trigger | Response |
|----------|---------|----------|
| Config deserialization fails | Malformed JSON | Worker logs error, stays in waiting state |
| Config arrives after weights | Race condition | Weights dropped, logged as "GPU weights not available" |
| Worker disconnect during transfer | Network failure | Coordinator retries on reconnect |
| Timeout waiting for layer status | Slow worker | Coordinator uses default (assume stateless) |

## Key Design Decisions

1. **Config before weights**: The 100ms sleep after config ensures worker processes config before weight chunks arrive.

2. **Empty layer request triggers status**: Coordinator sends empty layer request to get worker's current status.

3. **configSent tracking**: Prevents duplicate config sends if worker reconnects.

4. **Chunk buffering**: Worker accumulates all chunks before attempting deserialization and GPU upload.

## Files Involved

| File | Role |
|------|------|
| `p2p/protocol.go` | Message routing and handlers |
| `pkg/inference/config_transfer.go` | JSON serialization/deserialization |
| `pkg/inference/coordinator.go` | Config sending logic, configSent tracking |
| `cmd/worker/main.go` | Config receiving, handleModelConfig, handleWeights |

## Related Documentation

- [ADR-006: Config Transfer Protocol](../decisions/ADR-006-config-transfer-protocol.md)
- [Data Flow: Config Transfer](data-flow-config-transfer.md)
- [Worker Architecture](sequence-worker-modes.md)

---

*Updated 2025-01-24 - Added Phase 3-5 implementation details*

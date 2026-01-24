# Sequence Diagram: Worker Operation Modes

## Overview

NeuroGrid workers can operate in two modes: stateful (with local model files) or stateless (receiving config and weights from coordinator). This document describes both modes and their initialization sequences.

## Worker Mode Decision Flow

```mermaid
flowchart TD
    A[Worker Start] --> B{--model flag?}
    B -->|Yes| C[Stateful Mode]
    B -->|No| D[Stateless Mode]

    C --> C1[loadLocalWeights]
    C1 --> C2[Parse config from model]
    C2 --> C3[Upload to GPU]
    C3 --> C4[Report loaded layers]
    C4 --> C5[Ready for inference]

    D --> D1[Initialize empty]
    D1 --> D2[Wait for config]
    D2 --> D3[handleModelConfig]
    D3 --> D4[Wait for weights]
    D4 --> D5[handleWeights + GPU upload]
    D5 --> C5

    style C fill:#d4edda
    style D fill:#cce5ff
```

## Stateful Mode (Local Model)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as Command Line
    participant W as Worker
    participant FS as File System
    participant GPU as GPU Memory
    participant C as Coordinator

    CLI->>W: Start with --model /path/to/model
    W->>W: NewWorker(config)
    W->>W: Start()

    Note over W,FS: Load local weights
    W->>FS: model.NewWeightLoader(path)
    FS-->>W: loader

    loop For each layer
        W->>FS: LoadLayerWeights(layerID)
        FS-->>W: LayerWeights
        W->>GPU: CreateLayerWeightsFromHost()
        GPU-->>W: gpuWeights[layerID]
        W->>W: Initialize KV cache
    end

    W->>W: weightsReady = true

    Note over W,C: Connect to coordinator
    W->>C: mDNS/DHT discovery
    C->>W: Connect

    W->>C: SendLayerStatus([0,1,2,...])
    C->>C: Skip weight transfer (layers cached)

    Note over W: Ready for inference
```

## Stateless Mode (Remote Config + Weights)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as Command Line
    participant W as Worker
    participant P as P2P Protocol
    participant GPU as GPU Memory
    participant C as Coordinator

    CLI->>W: Start without --model
    W->>W: NewWorker(config)
    W->>W: Start()

    Note over W: Register P2P handlers
    W->>P: OnModelConfigReceived(handleModelConfig)
    W->>P: OnWeightsReceived(handleWeights)
    W->>P: OnLayerRequestReceived(handleLayerRequest)

    W->>W: Log "Waiting for config from coordinator..."

    Note over W,C: Connect to coordinator
    W->>C: mDNS/DHT discovery
    C->>W: Connect

    W->>C: SendLayerStatus([])
    Note over C: Detects stateless worker

    rect rgb(255, 243, 224)
        Note over C,W: Config Transfer (Phase 3-5)
        C->>C: SerializeConfig(config, modelName)
        C->>W: MsgTypeModelConfig (0x07)
        W->>W: handleModelConfig()
        W->>W: DeserializeConfig(data)
        W->>W: modelConfig = config
        W->>W: Log "Received model config"
    end

    C->>C: Sleep(100ms)

    rect rgb(224, 247, 250)
        Note over C,W: Weight Transfer
        loop For each assigned layer
            C->>W: MsgTypeWeights (chunked)
            W->>W: handleWeights()
            W->>W: Buffer chunks
            W->>W: DeserializeLayerWeights()
            W->>GPU: CreateLayerWeightsFromHost()
            W->>W: gpuWeights[layerID] = weights
            W->>W: Initialize KV cache
            W->>C: SendWeightsAck(layerID)
        end
    end

    W->>W: weightsReady = true
    Note over W: Ready for inference
```

## Handler Registration

```mermaid
sequenceDiagram
    participant W as Worker
    participant P as Protocol

    Note over W,P: Handler registration in Start()

    W->>P: OnActivationReceived(handleActivation)
    Note right of P: Process inference requests

    W->>P: OnWeightsReceived(handleWeights)
    Note right of P: Receive layer weights from coordinator

    W->>P: OnLayerRequestReceived(handleLayerRequest)
    Note right of P: Respond to layer status queries

    W->>P: OnModelConfigReceived(handleModelConfig)
    Note right of P: Receive config for stateless mode
```

## Key Components

### Worker State Fields

| Field | Type | Purpose |
|-------|------|---------|
| `modelConfig` | `*types.LlamaConfig` | Model configuration (set by local load or received) |
| `layerWeights` | `map[int]*LayerWeights` | CPU layer weights |
| `gpuWeights` | `map[int]*bindings.LayerWeights` | GPU layer weights |
| `gpuKVCaches` | `map[int]*bindings.KVCache` | KV cache per layer |
| `weightsReady` | `bool` | Flag indicating ready for inference |
| `chunkBuffers` | `map[int]*chunkBuffer` | Accumulates weight chunks |
| `startLayerID` | `int` | First layer this worker handles |
| `endLayerID` | `int` | Last layer this worker handles |

### Handler Functions

| Handler | Message Type | Purpose |
|---------|-------------|---------|
| `handleActivation` | `MsgTypeActivation` | Process layer forward pass |
| `handleWeights` | `MsgTypeWeights` | Receive and upload weights to GPU |
| `handleLayerRequest` | `MsgTypeLayerRequest` | Report loaded layers to coordinator |
| `handleModelConfig` | `MsgTypeModelConfig` | Set model config for stateless mode |

## Chunk Buffer Flow

```mermaid
flowchart TD
    A[Receive Weight Chunk] --> B{Buffer exists?}
    B -->|No| C[Create chunkBuffer]
    B -->|Yes| D[Get existing buffer]
    C --> E[Store chunk at index]
    D --> E
    E --> F{All chunks received?}
    F -->|No| G[Wait for more]
    F -->|Yes| H[Concatenate chunks]
    H --> I[DeserializeLayerWeights]
    I --> J{modelConfig set?}
    J -->|Yes| K[CreateLayerWeightsFromHost]
    J -->|No| L[Log error - cannot upload]
    K --> M[Store in gpuWeights]
    M --> N[Initialize KV cache]
    N --> O[Delete buffer]
    O --> P[SendWeightsAck]
```

## Mode Comparison Table

| Aspect | Stateful Mode | Stateless Mode |
|--------|---------------|----------------|
| Startup | `--model /path` | No `--model` flag |
| Config source | Local model files | Coordinator (MsgTypeModelConfig) |
| Weight source | Local SafeTensors | Coordinator (MsgTypeWeights) |
| GPU upload | During startup | After receiving config + weights |
| Ready time | Fast (local I/O) | Slower (network transfer) |
| Disk usage | High (model files) | Low (no local model) |
| Use case | Primary nodes | Lightweight workers |

## Error Handling

| Error | Handler | Recovery |
|-------|---------|----------|
| Config deserialization failure | `handleModelConfig` | Log error, stay waiting |
| Weight deserialization failure | `handleWeights` | Log error, skip layer |
| GPU upload failure | `handleWeights` | Log error, continue (fail at execution) |
| No modelConfig when uploading | `handleWeights` | Skip GPU upload, log warning |

## Files Involved

| File | Components |
|------|------------|
| `cmd/worker/main.go` | Worker struct, handlers, Start() |
| `pkg/inference/config_transfer.go` | DeserializeConfig |
| `pkg/model/weights.go` | DeserializeLayerWeights |
| `gpu/bindings/layer.go` | CreateLayerWeightsFromHost |

---

*Created 2025-01-24 - Documents Phase 3-5 worker modes implementation*

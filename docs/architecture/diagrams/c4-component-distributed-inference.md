# C4 Component Diagram: Hybrid Distributed Inference System

## Overview

This diagram shows the internal components of the NeuroGrid distributed inference system, focusing on the hybrid worker model that supports both stateful (local model) and stateless (config transfer) operation modes.

## C4 Context

```mermaid
C4Context
    title System Context - NeuroGrid Distributed Inference

    Person(user, "API Client", "Sends inference requests")

    System(neurogrid, "NeuroGrid Engine", "Distributed LLM inference across GPUs")

    System_Ext(gpu1, "RTX 4090", "Primary GPU - 22 layers")
    System_Ext(gpu2, "RTX 2080 Ti", "Secondary GPU - 10 layers")

    Rel(user, neurogrid, "HTTP POST /v1/chat/completions", "JSON")
    Rel(neurogrid, gpu1, "CUDA kernels", "PCIe")
    Rel(neurogrid, gpu2, "CUDA kernels", "PCIe")
```

## C4 Container

```mermaid
C4Container
    title Container Diagram - NeuroGrid Nodes

    Person(client, "HTTP Client", "curl / SDK")

    Container_Boundary(coord_node, "Coordinator Node") {
        Container(api, "HTTP API", "Go net/http", "OpenAI-compatible endpoints")
        Container(coordinator, "Coordinator", "Go", "Layer distribution, weight management")
        Container(p2p_coord, "P2P Protocol", "libp2p", "Peer discovery, message routing")
        ContainerDb(weights, "Model Weights", "SafeTensors", "Mistral 7B (~14GB)")
    }

    Container_Boundary(worker_node, "Worker Node") {
        Container(worker, "Worker", "Go", "Layer execution, GPU management")
        Container(p2p_work, "P2P Protocol", "libp2p", "Message handling")
        Container(cuda, "CUDA Runtime", "C++/CUDA", "GPU kernels")
    }

    Rel(client, api, "POST /v1/chat/completions", "HTTPS")
    Rel(api, coordinator, "Forward request")
    Rel(coordinator, weights, "Load layer weights")
    Rel(coordinator, p2p_coord, "SendModelConfig, SendWeights")
    Rel(p2p_coord, p2p_work, "MsgType 0x01-0x07", "TCP/QUIC")
    Rel(p2p_work, worker, "Deliver messages")
    Rel(worker, cuda, "Execute layers", "CGO")
```

## C4 Component - Config Transfer

```mermaid
C4Component
    title Component Diagram - Config Transfer Subsystem

    Container_Boundary(inference_pkg, "pkg/inference") {
        Component(config_transfer, "config_transfer.go", "Go", "TransferableConfig serialization")
        Component(coordinator_comp, "coordinator.go", "Go", "Weight distribution, config tracking")
        Component(weight_dist, "weight_distributor.go", "Go", "Layer weight serialization")
    }

    Container_Boundary(p2p_pkg, "p2p") {
        Component(protocol, "protocol.go", "Go", "Message types, handlers, wire format")
    }

    Container_Boundary(types_pkg, "pkg/types") {
        Component(config_types, "config.go", "Go", "LlamaConfig struct")
    }

    Container_Boundary(worker_cmd, "cmd/worker") {
        Component(worker_main, "main.go", "Go", "Worker process, message handlers")
    }

    Rel(coordinator_comp, config_transfer, "SerializeConfig()")
    Rel(coordinator_comp, weight_dist, "SerializeLayerWeights()")
    Rel(coordinator_comp, protocol, "SendModelConfig(), SendWeights()")
    Rel(config_transfer, config_types, "FromLlamaConfig()")
    Rel(protocol, worker_main, "OnModelConfigReceived callback")
    Rel(worker_main, config_transfer, "DeserializeConfig()")
    Rel(config_transfer, config_types, "ToLlamaConfig()")
```

## C4 Component - Worker Internal

```mermaid
C4Component
    title Component Diagram - Worker Internal Components

    Container_Boundary(worker_cmd, "cmd/worker") {
        Component(worker_main, "main.go", "Go", "Worker entry point")
        Component(handler_activation, "handleActivation", "Go", "Process layer forward pass")
        Component(handler_weights, "handleWeights", "Go", "Receive + upload weights")
        Component(handler_config, "handleModelConfig", "Go", "Receive model config")
        Component(handler_layer, "handleLayerRequest", "Go", "Report layer status")
    }

    Container_Boundary(gpu_pkg, "gpu/bindings") {
        Component(layer_weights, "LayerWeights", "Go/CGO", "GPU weight buffers")
        Component(kv_cache, "KVCache", "Go/CGO", "GPU KV cache")
        Component(layer_forward, "LayerForward", "Go/CGO", "Execute layer")
    }

    Container_Boundary(inference_pkg, "pkg/inference") {
        Component(config_transfer, "config_transfer.go", "Go", "Config deserialization")
    }

    Rel(worker_main, handler_activation, "calls")
    Rel(worker_main, handler_weights, "calls")
    Rel(worker_main, handler_config, "calls")
    Rel(worker_main, handler_layer, "calls")
    Rel(handler_config, config_transfer, "DeserializeConfig()")
    Rel(handler_weights, layer_weights, "CreateLayerWeightsFromHost()")
    Rel(handler_weights, kv_cache, "NewKVCache()")
    Rel(handler_activation, layer_forward, "LayerForward()")
```

## Components

### Config Transfer Components

| Component | File | Responsibility |
|-----------|------|----------------|
| TransferableConfig | `pkg/inference/config_transfer.go` | JSON-serializable model config struct |
| SerializeConfig | `pkg/inference/config_transfer.go` | LlamaConfig -> JSON bytes |
| DeserializeConfig | `pkg/inference/config_transfer.go` | JSON bytes -> LlamaConfig |
| FromLlamaConfig | `pkg/inference/config_transfer.go` | Type conversion helper |
| ToLlamaConfig | `pkg/inference/config_transfer.go` | Type conversion helper |

### Protocol Components

| Component | File | Responsibility |
|-----------|------|----------------|
| MsgTypeModelConfig | `p2p/protocol.go` | Message type constant (0x07) |
| ModelConfigHandler | `p2p/protocol.go` | Callback type for config messages |
| OnModelConfigReceived | `p2p/protocol.go` | Register config handler |
| SendModelConfig | `p2p/protocol.go` | Send config to peer |
| handleExtendedMessage | `p2p/protocol.go` | Route incoming messages |

### Coordinator Components

| Component | File | Responsibility |
|-----------|------|----------------|
| distributeWeightsToPeer | `pkg/inference/coordinator.go` | Orchestrate config + weight transfer |
| sendConfigToPeer | `pkg/inference/coordinator.go` | Send config before weights |
| configSent | `pkg/inference/coordinator.go` | Track which peers received config |
| modelName | `pkg/inference/coordinator.go` | Model name for serialization |
| shortPeerID | `pkg/inference/coordinator.go` | Helper for log formatting |

### Worker Components

| Component | File | Responsibility |
|-----------|------|----------------|
| handleModelConfig | `cmd/worker/main.go` | Process received config |
| handleWeights | `cmd/worker/main.go` | Receive weights, upload to GPU |
| handleActivation | `cmd/worker/main.go` | Execute layer forward pass |
| handleLayerRequest | `cmd/worker/main.go` | Report loaded layers |
| modelConfig | `cmd/worker/main.go` | Stored LlamaConfig for GPU ops |
| chunkBuffers | `cmd/worker/main.go` | Accumulate weight chunks |

## Protocol Message Types

```mermaid
flowchart LR
    subgraph Legacy["Legacy (16B header)"]
        L1[Tensor Data]
    end

    subgraph Extended["Extended (25B header)"]
        E1["0x01 Activation"]
        E2["0x02 Response"]
        E3["0x03 Weights"]
        E4["0x04 WeightsAck"]
        E5["0x05 LayerStatus"]
        E6["0x06 LayerRequest"]
        E7["0x07 ModelConfig"]
    end

    subgraph Traced["Traced (50B header)"]
        T1["0x11 TracedActivation"]
        T2["0x12 TracedResponse"]
    end
```

## Hybrid Worker Modes

```mermaid
flowchart TB
    subgraph StatefulMode["Stateful Mode (--model flag)"]
        S1[Worker Start] --> S2[Load Local Model]
        S2 --> S3[Parse Config]
        S3 --> S4[Report LayerStatus]
        S4 --> S5[Ready for Inference]
    end

    subgraph StatelessMode["Stateless Mode (no --model)"]
        L1[Worker Start] --> L2[Wait for Config]
        L2 --> L3[Receive MsgTypeModelConfig]
        L3 --> L4[handleModelConfig - DeserializeConfig]
        L4 --> L5[Receive MsgTypeWeights]
        L5 --> L6[handleWeights - Upload to GPU]
        L6 --> L7[Ready for Inference]
    end
```

## Coordinator Config Flow

```mermaid
flowchart TD
    A[Peer Connected] --> B{Has Layer Assignments?}
    B -->|No| Z[Skip]
    B -->|Yes| C[Setup Remote Executor]
    C --> D[requestAndDistributeWeights]
    D --> E[Wait 500ms for status]
    E --> F{Received Layer Status?}
    F -->|No| G[RequestLayerStatus]
    F -->|Yes| H[Got loadedLayers]
    G --> H
    H --> I[distributeWeightsToPeer]
    I --> J{configSent?}
    J -->|Yes| K[Skip config]
    J -->|No| L[sendConfigToPeer]
    L --> M[SerializeConfig]
    M --> N[SendModelConfig 0x07]
    N --> O[configSent = true]
    O --> P[Sleep 100ms]
    P --> K
    K --> Q[Filter missing weights]
    Q --> R[DistributeLayersToPerformer]
```

## Dependencies

```mermaid
graph TD
    config_transfer --> types_config[pkg/types/config.go]
    config_transfer --> encoding_json[encoding/json]
    protocol --> libp2p[github.com/libp2p/go-libp2p]
    coordinator --> config_transfer
    coordinator --> protocol
    worker --> config_transfer
    worker --> protocol
    worker --> cuda_bindings[gpu/bindings]
    worker --> model_weights[pkg/model/weights.go]
```

## Worker Chunk Buffer Flow

```mermaid
flowchart TD
    A[Receive Chunk] --> B{chunkBuffers has layer?}
    B -->|No| C[Create chunkBuffer]
    B -->|Yes| D[Get buffer]
    C --> D
    D --> E[Store chunk at index]
    E --> F{All chunks received?}
    F -->|No| G[Return, wait for more]
    F -->|Yes| H[Concatenate chunks]
    H --> I[DeserializeLayerWeights]
    I --> J{modelConfig != nil?}
    J -->|Yes| K[CreateLayerWeightsFromHost]
    J -->|No| L[Log warning, skip GPU]
    K --> M[Store in gpuWeights]
    M --> N[Initialize KV cache]
    N --> O[Delete chunkBuffer]
    O --> P[weightsReady = true]
    P --> Q[SendWeightsAck]
```

## Related Documentation

- [ADR-006: Config Transfer Protocol](../decisions/ADR-006-config-transfer-protocol.md)
- [Sequence Diagram: Config Transfer](sequence-config-transfer.md)
- [Sequence Diagram: Worker Modes](sequence-worker-modes.md)
- [Data Flow: Config Transfer](data-flow-config-transfer.md)
- [P2P Networking Architecture](../p2p-networking.md)

---

*Updated 2025-01-24 - Added Phase 3-5 implementation details (coordinator sendConfigToPeer, worker handleModelConfig)*

# Data Flow Diagram: Activation Transfer with Pinned Memory

## Overview

This document describes the data flow for activation transfers in distributed LLM inference using CUDA pinned memory buffers. The optimization eliminates per-transfer allocations and enables DMA transfers for improved latency.

## High-Level Data Flow

```mermaid
flowchart LR
    subgraph Coordinator["Coordinator Node"]
        A[Token Input] --> B[Embedding Lookup]
        B --> C[Layer Forward]
        C --> D[Buffer Pool Get]
        D --> E[Serialize Activation]
    end

    subgraph Network["libp2p Network"]
        E --> F[TCP Stream]
    end

    subgraph Worker["Worker Node"]
        F --> G[Buffer Pool Get]
        G --> H[Deserialize]
        H --> I[GPU Transfer]
        I --> J[Layer Forward]
        J --> K[Serialize Result]
        K --> L[Buffer Pool Put]
    end

    subgraph Response["Response Path"]
        L --> M[TCP Stream]
        M --> N[Buffer Pool Put]
        N --> O[Continue Pipeline]
    end
```

## Detailed Activation Transfer Flow

### 1. Outbound Transfer (Coordinator to Worker)

```mermaid
sequenceDiagram
    autonumber
    participant Engine as Inference Engine
    participant Pool as BufferPool
    participant Transport as P2PTransport
    participant Network as libp2p Stream
    participant Worker as Worker Node

    Engine->>Pool: Get(activationSize)
    Pool-->>Engine: pinnedBuffer[]

    Engine->>Engine: Copy activation to buffer

    Engine->>Transport: SendActivation(layerID, seqID, pinnedBuffer)
    Transport->>Network: Write header + data

    Network-->>Worker: TCP delivery

    Note over Engine,Pool: Buffer held during transfer
    Engine->>Pool: Put(pinnedBuffer)
    Note over Pool: Buffer returned for reuse
```

### 2. Inbound Transfer (Worker Receive)

```mermaid
sequenceDiagram
    autonumber
    participant Network as libp2p Stream
    participant Transport as P2PTransport
    participant Pool as BufferPool
    participant Worker as Worker Handler
    participant GPU as GPU Device

    Network->>Transport: Incoming stream

    Transport->>Pool: Get(dataLen)
    Pool-->>Transport: pinnedBuffer[]

    Transport->>Transport: io.ReadFull(stream, pinnedBuffer)

    Transport->>Worker: handleActivation(pinnedBuffer)

    Worker->>GPU: CopyToDeviceRaw(gpuPtr, pinnedBuffer)
    Note over Worker,GPU: DMA transfer (pinned memory)

    Worker->>Pool: Put(pinnedBuffer)

    Worker->>GPU: LayerForward(...)
```

## Buffer Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Preallocated: Pool Creation

    Preallocated --> InPool: buffers channel

    InPool --> InUse: pool.Get()
    InUse --> InPool: pool.Put()

    InUse --> Overflow: Pool exhausted
    Overflow --> Freed: pool.Put() when full

    InPool --> Freed: pool.Close()
    Freed --> [*]

    note right of Preallocated
        32 buffers * 16KB = 512KB
        allocated at startup
    end note

    note right of InUse
        Buffer tracked in bufMap
        for Put() lookup
    end note

    note right of Overflow
        Fallback allocation when
        pool exhausted
    end note
```

## Memory Flow Comparison

### Before Optimization (Regular Memory)

```mermaid
flowchart TB
    subgraph "Per-Transfer Allocation"
        A1[make[]byte, dataLen] --> B1[CPU copies data]
        B1 --> C1[cudaMemcpy H2D]
        C1 --> D1[CPU involved in copy]
        D1 --> E1[GPU receives]
        E1 --> F1[GC collects buffer]
    end

    style A1 fill:#f96
    style F1 fill:#f96
```

### After Optimization (Pinned Memory)

```mermaid
flowchart TB
    subgraph "Pooled Pinned Memory"
        A2[pool.Get] --> B2[Reuse existing buffer]
        B2 --> C2[cudaMemcpy H2D]
        C2 --> D2[DMA transfer]
        D2 --> E2[GPU receives]
        E2 --> F2[pool.Put]
    end

    style A2 fill:#9f6
    style F2 fill:#9f6
```

## Data Transformation Steps

| Step | Input | Process | Output | Memory Type |
|------|-------|---------|--------|-------------|
| 1 | Token IDs | Embedding lookup | Hidden state FP16 | GPU Memory |
| 2 | Hidden state | Layer forward | Activation FP16 | GPU Memory |
| 3 | Activation | GPU to Host copy | Byte slice | Pinned Memory |
| 4 | Byte slice | Network transfer | TCP packets | Pinned Memory |
| 5 | TCP packets | Reassembly | Byte slice | Pinned Memory |
| 6 | Byte slice | Host to GPU copy | Activation FP16 | GPU Memory |
| 7 | Activation | Layer forward | Hidden state FP16 | GPU Memory |

## Transfer Size Analysis

For Llama models over Gigabit Ethernet:

| Model | Hidden Size | Transfer Size (FP16) | Theoretical Time |
|-------|-------------|---------------------|------------------|
| Llama 7B | 4096 | 8 KB | ~64 us |
| Llama 13B | 5120 | 10 KB | ~80 us |
| Llama 30B | 6656 | 13 KB | ~104 us |
| Llama 65B | 8192 | 16 KB | ~128 us |

Note: Actual latency includes protocol overhead and GPU transfer time.

## Pinned Memory DMA Benefits

```mermaid
flowchart LR
    subgraph "Regular Memory Transfer"
        R1[Host Memory] --> R2[CPU Buffer]
        R2 --> R3[PCIe Controller]
        R3 --> R4[GPU Memory]
    end

    subgraph "Pinned Memory Transfer"
        P1[Pinned Memory] --> P2[PCIe Controller]
        P2 --> P3[GPU Memory]
    end

    style R2 fill:#f96
    style P1 fill:#9f6
```

**Key Difference**: Pinned memory bypasses CPU buffer, enabling Direct Memory Access.

## Error Handling Flow

```mermaid
flowchart TB
    A[Get Buffer] --> B{Pool Available?}
    B -->|Yes| C[Return Pooled Buffer]
    B -->|No| D[Allocate Fallback]
    D --> E{Alloc Success?}
    E -->|Yes| F[Return New Buffer]
    E -->|No| G[Return nil]

    C --> H[Use Buffer]
    F --> H

    H --> I[Put Buffer]
    I --> J{Pool Full?}
    J -->|No| K[Return to Pool]
    J -->|Yes| L[Free Buffer]
```

## Metrics and Observability

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| buffer_pool_available | Gauge | Buffers available in pool |
| buffer_pool_capacity | Gauge | Total pool capacity |
| buffer_pool_misses | Counter | Fallback allocations |
| transfer_latency_ms | Histogram | End-to-end transfer time |
| dma_transfer_bytes | Counter | Bytes transferred via DMA |

### Monitoring Points

```mermaid
flowchart TB
    subgraph "Metrics Collection"
        M1[Pool Stats] --> P[Prometheus]
        M2[Transfer Latency] --> P
        M3[Memory Usage] --> P
    end

    P --> G[Grafana Dashboard]
```

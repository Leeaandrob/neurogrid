# C4 Diagram: CUDA Transport Optimization

## Overview

This document describes the architecture of the CUDA Transport Optimization feature using C4 model diagrams. The feature implements CUDA pinned memory buffer pools to optimize activation transfers in distributed LLM inference.

## C4 Context Diagram

Shows the NeuroGrid system in context with external users and systems.

```mermaid
C4Context
    title System Context Diagram - NeuroGrid Distributed Inference

    Person(user, "API Client", "Application or user sending inference requests")
    Person(admin, "Administrator", "System operator monitoring clusters")

    System(neurogrid, "NeuroGrid Engine", "Distributed LLM inference engine with optimized GPU transfers")

    System_Ext(prometheus, "Prometheus", "Metrics collection")
    System_Ext(models, "Model Storage", "HuggingFace Hub or local safetensors")

    Rel(user, neurogrid, "POST /v1/chat/completions", "HTTPS/HTTP")
    Rel(admin, neurogrid, "GET /metrics, /health", "HTTP")
    Rel(neurogrid, prometheus, "Metrics export", "HTTP")
    Rel(neurogrid, models, "Download weights", "HTTPS")
```

## C4 Container Diagram

Shows the major containers within the NeuroGrid system.

```mermaid
C4Container
    title Container Diagram - NeuroGrid Distributed Inference

    Person(user, "API Client", "Sends inference requests")

    Container_Boundary(neurogrid, "NeuroGrid Cluster") {
        Container(coordinator, "Coordinator", "Go + CUDA", "Orchestrates inference, manages workers, handles API")
        Container(worker1, "Worker Node 1", "Go + CUDA", "Executes transformer layers on GPU 1")
        Container(worker2, "Worker Node 2", "Go + CUDA", "Executes transformer layers on GPU 2")

        ContainerDb(kvstore, "KV Cache", "GPU Memory", "Attention key-value cache per layer")
        Container(p2p, "libp2p Network", "libp2p", "Peer discovery and activation transfer")
    }

    System_Ext(gpu1, "RTX 4090", "24GB VRAM, Primary GPU")
    System_Ext(gpu2, "RTX 2080 Ti", "11GB VRAM, Secondary GPU")

    Rel(user, coordinator, "Chat Completions API", "HTTP/REST")
    Rel(coordinator, worker1, "Activations", "libp2p/TCP")
    Rel(coordinator, worker2, "Activations", "libp2p/TCP")
    Rel(worker1, worker2, "Activations", "libp2p/TCP")
    Rel(worker1, gpu1, "CUDA ops", "PCIe")
    Rel(worker2, gpu2, "CUDA ops", "PCIe")
    Rel(coordinator, kvstore, "KV update", "GPU Memory")
```

## C4 Component Diagram - Buffer Pool System

Detailed view of the buffer pool components within a worker node.

```mermaid
C4Component
    title Component Diagram - CUDA Transport Buffer Pool

    Container_Boundary(transport, "Transport Layer (pkg/transport)") {
        Component(interface, "BufferPool Interface", "Go Interface", "Get(size) / Put(buf) / Close()")
        Component(pinned_pool, "PinnedTransportBufferPool", "Go + CUDA", "CUDA pinned memory wrapper for transport")
        Component(default_pool, "DefaultBufferPool", "Go", "sync.Pool fallback for non-CUDA")
        Component(p2p_transport, "P2PTransport", "Go + libp2p", "libp2p-based peer communication")
    }

    Container_Boundary(bindings, "GPU Bindings (gpu/bindings)") {
        Component(gpu_bindings, "CUDA CGO Bindings", "CGO", "AllocPinnedMemory, FreePinnedMemory")
        Component(pinned_buffer, "PinnedBuffer", "Go struct", "Ptr, Size, AsSlice()")
        Component(buffer_pool, "PinnedBufferPool", "Go", "Channel-based lock-free pool")
    }

    Container_Boundary(cuda, "CUDA Runtime") {
        Component(cuda_host, "cudaHostAlloc", "CUDA API", "Page-locked memory allocation")
        Component(cuda_free, "cudaFreeHost", "CUDA API", "Page-locked memory deallocation")
    }

    Rel(p2p_transport, interface, "Uses")
    Rel(interface, pinned_pool, "Implements")
    Rel(interface, default_pool, "Implements")
    Rel(pinned_pool, buffer_pool, "Wraps")
    Rel(buffer_pool, pinned_buffer, "Manages")
    Rel(buffer_pool, gpu_bindings, "Calls")
    Rel(gpu_bindings, cuda_host, "CGO call")
    Rel(gpu_bindings, cuda_free, "CGO call")
```

## C4 Component Diagram - Phase 2 Integration

Shows buffer pool integration points across all components (Phase 2 implementation).

```mermaid
C4Component
    title Component Diagram - Buffer Pool Integration (Phase 2)

    Container_Boundary(worker_container, "Worker (cmd/worker)") {
        Component(worker, "Worker", "Go", "handleActivation, executeLayer, initPinnedPool")
        Component(adapter, "pinnedPoolAdapter", "Go", "Bridges bindings.PinnedBufferPool to p2p.BufferPool")
    }

    Container_Boundary(protocol_container, "P2P Protocol (p2p)") {
        Component(protocol, "Protocol", "Go", "Tensor transfer protocol handler")
        Component(p2p_buffer_interface, "BufferPool", "Interface", "p2p package interface (avoids circular imports)")
    }

    Container_Boundary(transport_container, "Transport Layer (pkg/transport)") {
        Component(p2p_transport, "P2PTransport", "Go + libp2p", "WithBufferPool option, SetBufferPool/GetBufferPool")
        Component(transport_interface, "BufferPool", "Interface", "transport.BufferPool interface")
    }

    Container_Boundary(inference_container, "Inference (pkg/inference)") {
        Component(cuda_executor, "CUDALayerExecutor", "Go + CUDA", "Preallocated inputGPU/outputGPU buffers")
    }

    Container_Boundary(bindings_container, "GPU Bindings (gpu/bindings)") {
        Component(pinned_pool, "PinnedBufferPool", "Go", "Core pinned memory pool")
    }

    Rel(worker, adapter, "Creates")
    Rel(adapter, pinned_pool, "Wraps")
    Rel(worker, protocol, "Configures with SetBufferPool")
    Rel(protocol, p2p_buffer_interface, "Uses")
    Rel(adapter, p2p_buffer_interface, "Implements")
    Rel(p2p_transport, transport_interface, "Uses")
    Rel(cuda_executor, pinned_pool, "Independent (GPU buffers)")
```

## Component Details

### BufferPool Interface

```go
type BufferPool interface {
    Get(size int) []byte  // Returns buffer of at least size bytes
    Put(buf []byte)       // Returns buffer to pool for reuse
    Close() error         // Releases all pooled resources
}
```

### PinnedTransportBufferPool

Wraps `bindings.PinnedBufferPool` for transport layer use:

| Field | Type | Purpose |
|-------|------|---------|
| pool | *bindings.PinnedBufferPool | Underlying CUDA pinned pool |
| bufSize | int | Size of each buffer |
| bufMap | map[uintptr]*PinnedBuffer | Maps slice ptr to PinnedBuffer |
| closed | bool | Closed state flag |

### PinnedBufferPool (bindings)

Channel-based lock-free buffer management:

| Field | Type | Purpose |
|-------|------|---------|
| buffers | chan *PinnedBuffer | Lock-free buffer channel |
| bufSize | uint64 | Buffer size (16KB default) |
| count | int | Pool capacity (32 default) |

### pinnedPoolAdapter (Worker)

Bridges bindings and p2p package interfaces:

| Method | Behavior | Notes |
|--------|----------|-------|
| Get(size) | Returns buffer from pool or allocates fallback | Falls back to make([]byte) |
| Put(buf) | No-op (cannot track slice->PinnedBuffer mapping) | Memory managed by pool/GC |
| Close() | Closes underlying PinnedBufferPool | Frees CUDA pinned memory |

### CUDA Bindings

| Function | CUDA API | Purpose |
|----------|----------|---------|
| AllocPinnedMemory | cudaHostAlloc | Allocate page-locked memory |
| FreePinnedMemory | cudaFreeHost | Free page-locked memory |
| IsPinnedMemory | cudaHostGetFlags | Check if memory is pinned |

## Integration Points (Phase 2)

```mermaid
flowchart TB
    subgraph "Application Layer"
        API["/v1/chat/completions"]
        Engine["Inference Engine"]
    end

    subgraph "Worker Layer"
        Worker["Worker"]
        Adapter["pinnedPoolAdapter"]
        Executor["CUDALayerExecutor"]
    end

    subgraph "Protocol Layer"
        Protocol["p2p.Protocol"]
        P2PInterface["p2p.BufferPool"]
    end

    subgraph "Transport Layer"
        Transport["P2PTransport"]
        TInterface["transport.BufferPool"]
    end

    subgraph "GPU Bindings"
        Bindings["gpu/bindings"]
        PinnedPool["PinnedBufferPool"]
    end

    subgraph "CUDA Runtime"
        CUDA["cudaHostAlloc/cudaFreeHost"]
    end

    API --> Engine
    Engine --> Transport
    Transport --> TInterface
    TInterface --> PinnedPool

    Worker --> Adapter
    Adapter --> P2PInterface
    P2PInterface --> Protocol
    Protocol --> PinnedPool

    Worker --> Executor
    Executor --> Bindings
    Bindings --> CUDA

    PinnedPool --> Bindings
    Bindings --> CUDA
```

## Thread Safety

All components are thread-safe:

| Component | Mechanism |
|-----------|-----------|
| PinnedBufferPool | Channel-based (lock-free) |
| PinnedTransportBufferPool | sync.RWMutex for bufMap |
| DefaultBufferPool | sync.Pool (lock-free) |
| Protocol.bufferPool | sync.RWMutex |
| P2PTransport.bufferPool | sync.Mutex (via SetBufferPool) |
| CUDALayerExecutor | sync.RWMutex |

## Configuration

| Constant | Value | Purpose |
|----------|-------|---------|
| DefaultPinnedBufferSize | 16KB | Covers Llama 7B (8KB), 13B (10KB) |
| DefaultPinnedPoolCount | 32 | Concurrent request handling |
| DefaultTransportBufferSize | 16KB | Transport layer default |
| DefaultTransportPoolSize | 32 | Transport pool capacity |

## Buffer Flow Summary

```mermaid
sequenceDiagram
    participant W as Worker
    participant A as pinnedPoolAdapter
    participant PR as Protocol
    participant PP as PinnedBufferPool
    participant CUDA as CUDA Runtime

    Note over W,CUDA: Initialization (Start)
    W->>PP: NewPinnedBufferPoolWithDefaults()
    PP->>CUDA: cudaHostAlloc x 32
    W->>A: Create adapter
    W->>PR: SetBufferPool(adapter)

    Note over W,CUDA: Message Handling
    PR->>A: Get(dataLen)
    A->>PP: Get()
    PP-->>A: PinnedBuffer
    A-->>PR: []byte slice
    PR->>PR: io.ReadFull(stream, buffer)
    PR->>PR: Copy data, Put(buffer)
    PR->>A: Put(buffer)
    Note right of A: No-op (cannot track)

    Note over W,CUDA: Shutdown
    W->>PP: Close()
    PP->>CUDA: cudaFreeHost x 32
```

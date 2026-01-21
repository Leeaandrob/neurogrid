# Task Breakdown: Distributed LLM Inference Engine E2E

## Document Information

| Field | Value |
|-------|-------|
| **Source PRP** | `docs/prps/distributed-inference-e2e.md` |
| **Created** | 2025-01-20 |
| **Total Tasks** | 32 |
| **Estimated Duration** | 8-10 weeks |
| **Complexity** | High |

---

## PRP Analysis Summary

### Feature Scope
Deliver a fully functional distributed LLM inference engine that:
- Runs Llama 7B/13B models across multiple GPUs (local and remote via libp2p)
- Exposes an OpenAI-compatible `/v1/chat/completions` API endpoint
- Uses layer parallelism for distribution across 5 GPUs (1 local + 4 remote)

### Key Technical Requirements
1. **Multi-GPU Infrastructure**: Extend existing single-GPU CUDA memory management to multi-device
2. **Transport Layer**: Unified abstraction for local CUDA P2P and remote libp2p transfers
3. **libp2p Integration**: Peer discovery (mDNS + DHT), tensor protocol, host management
4. **Pipeline Scheduler**: VRAM-aware layer assignment across heterogeneous GPUs
5. **Inference Engine**: Distributed forward pass, KV cache management, token sampling
6. **Model Loading**: Weight loading, sentencepiece tokenization
7. **HTTP API**: OpenAI-compatible chat completions endpoint

### Validation Requirements
- First token latency < 2s
- Token generation > 5 tokens/sec (batch=1)
- GPU utilization > 50% on each peer
- Query "Qual e a capital da Franca?" returns "Paris"

---

## Task Complexity Assessment

### Overall Complexity: HIGH

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Technical Depth | Complex | CUDA multi-device, libp2p networking, distributed systems |
| Integration Points | 8 | CUDA, CGO, libp2p, FlatBuffers, HTTP, sentencepiece |
| Codebase Familiarity | Medium | Phase 1 complete, patterns established |
| Risk Level | Medium-High | Network reliability, GPU memory management |

### Technical Challenges
1. **P2P Detection and Fallback**: Not all GPU pairs support direct P2P access
2. **Distributed KV Cache**: State management across network partitions
3. **Heterogeneous VRAM**: Dynamic layer assignment based on available memory
4. **Network Latency**: Activation transfer overhead impacts end-to-end latency
5. **Error Recovery**: Graceful handling of peer disconnects mid-inference

---

## Phase Organization

### Phase 1: Multi-GPU Infrastructure (M1)
**Objective**: Enable CUDA operations across multiple local GPUs
**Duration**: 1-1.5 weeks
**Deliverables**: Multi-device memory manager, P2P transfer utilities, CGO bindings

### Phase 2: Transport Layer (M2)
**Objective**: Create unified transport abstraction for local and remote transfers
**Duration**: 1 week
**Deliverables**: Transport interface, CUDA transport, transport router

### Phase 3: libp2p Integration (M3)
**Objective**: Enable peer-to-peer networking with discovery
**Duration**: 1.5 weeks
**Deliverables**: libp2p host, mDNS/DHT discovery, tensor protocol, P2P transport

### Phase 4: Pipeline Scheduler (M4)
**Objective**: Intelligent layer distribution across heterogeneous GPUs
**Duration**: 1 week
**Deliverables**: VRAM tracker, layer assignment algorithm, prefetch coordinator

### Phase 5: Inference Engine (M5)
**Objective**: Distributed forward pass with KV cache
**Duration**: 1.5 weeks
**Deliverables**: Distributed engine, KV cache manager, token sampler

### Phase 6: Model Loading & Tokenization (M6)
**Objective**: Load model weights and process text
**Duration**: 1 week
**Deliverables**: Weight loader, sentencepiece tokenizer, distributed model setup

### Phase 7: HTTP API (M7)
**Objective**: OpenAI-compatible REST API
**Duration**: 0.5 weeks
**Deliverables**: HTTP server, chat completions handler, streaming support

### Phase 8: Integration & Testing (M8)
**Objective**: End-to-end validation and performance optimization
**Duration**: 1 week
**Deliverables**: Integration tests, performance benchmarks, documentation

---

## Detailed Task Breakdown

---

### Milestone 1: Multi-GPU Infrastructure

---

#### TASK-001: Multi-Device Context Manager

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | None |
| **Milestone** | M1: Multi-GPU Infrastructure |

**Description**:
Implement `DeviceContext` and `MultiDeviceManager` structures in CUDA to manage multiple GPU devices. Each device context maintains its own compute stream, transfer stream, and P2P access matrix.

**Files to Create/Modify**:
- `gpu/cuda/memory.cu` (modify - add multi-device structures)
- `gpu/cuda/memory.h` (modify - add new APIs)

**Implementation Details**:
```c
// Structures to add to memory.cu
typedef struct {
    int device_id;
    size_t total_memory;
    size_t used_memory;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    bool peer_access[8];  // P2P access matrix
} DeviceContext;

typedef struct {
    DeviceContext* devices;
    int num_devices;
    void* staging_buffer;      // Pinned CPU memory
    size_t staging_buffer_size;
} MultiDeviceManager;
```

**Code Patterns to Follow**:
- Reference existing `CUDA_CHECK` macro at `gpu/cuda/memory.cu:14-21`
- Follow existing device initialization pattern at `gpu/cuda/memory.cu:30-54`

**Acceptance Criteria**:

*Scenario 1: Initialize multi-device manager*
```gherkin
Given a system with 2+ CUDA-capable GPUs
When cuda_multi_init is called with device IDs [0, 1]
Then both device contexts are created successfully
And each context has dedicated compute and transfer streams
And P2P access matrix is populated for all device pairs
```

*Scenario 2: Handle single GPU gracefully*
```gherkin
Given a system with 1 CUDA-capable GPU
When cuda_multi_init is called with device ID [0]
Then single device context is created
And P2P matrix shows no peer access available
```

**Checklist**:
- [ ] `DeviceContext` struct defined with all required fields
- [ ] `MultiDeviceManager` struct defined
- [ ] `cuda_multi_init()` implemented and tested
- [ ] `cuda_multi_shutdown()` properly releases all resources
- [ ] `cuda_get_device_context()` returns correct context by device ID
- [ ] Compute and transfer streams created per device
- [ ] Unit test passes with multiple GPUs

---

#### TASK-002: P2P Access Detection and Enablement

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | TASK-001 |
| **Milestone** | M1: Multi-GPU Infrastructure |

**Description**:
Implement P2P (peer-to-peer) access detection between GPU pairs using `cudaDeviceCanAccessPeer` and enable direct memory access using `cudaDeviceEnablePeerAccess`. Populate the P2P access matrix in each `DeviceContext`.

**Files to Create/Modify**:
- `gpu/cuda/memory.cu` (modify)
- `gpu/cuda/memory.h` (modify)

**Implementation Details**:
```c
// Add to cuda_multi_init
for (int i = 0; i < num_devices; i++) {
    for (int j = 0; j < num_devices; j++) {
        if (i != j) {
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]);
            manager->devices[i].peer_access[j] = can_access;
            if (can_access) {
                cudaSetDevice(device_ids[i]);
                cudaDeviceEnablePeerAccess(device_ids[j], 0);
            }
        }
    }
}
```

**Acceptance Criteria**:

*Scenario 1: P2P available between GPU pair*
```gherkin
Given GPUs 0 and 1 support P2P access
When P2P detection runs during initialization
Then devices[0].peer_access[1] is true
And cudaDeviceEnablePeerAccess is called for GPU 0 -> 1
```

*Scenario 2: P2P unavailable (PCIe topology)*
```gherkin
Given GPUs 0 and 2 do not support P2P access
When P2P detection runs during initialization
Then devices[0].peer_access[2] is false
And warning is logged about staged copy fallback
```

**Checklist**:
- [ ] P2P detection loop implemented
- [ ] P2P access enabled for supported pairs
- [ ] P2P matrix correctly populated
- [ ] Warning logged for non-P2P pairs
- [ ] Unit test validates P2P matrix accuracy

---

#### TASK-003: Cross-Device Memory Copy

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-001, TASK-002 |
| **Milestone** | M1: Multi-GPU Infrastructure |

**Description**:
Implement `cuda_cross_device_copy` that transfers data between GPUs. Use direct P2P copy when available, otherwise fall back to staged copy through pinned CPU memory.

**Files to Create/Modify**:
- `gpu/cuda/memory.cu` (modify)
- `gpu/cuda/memory.h` (modify)

**Implementation Details**:
```c
int cuda_cross_device_copy(void* dst, int dst_device,
                           void* src, int src_device,
                           size_t size, cudaStream_t stream) {
    if (dst_device == src_device) {
        // Same device - use regular copy
        return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }

    DeviceContext* src_ctx = &g_manager->devices[src_device];
    if (src_ctx->peer_access[dst_device]) {
        // Direct P2P copy
        cudaMemcpyPeerAsync(dst, dst_device, src, src_device, size, stream);
    } else {
        // Staged copy through pinned memory
        cudaMemcpyAsync(g_manager->staging_buffer, src, size,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(dst, g_manager->staging_buffer, size,
                        cudaMemcpyHostToDevice, stream);
    }
    return 0;
}
```

**Acceptance Criteria**:

*Scenario 1: P2P copy between supported GPUs*
```gherkin
Given GPUs 0 and 1 have P2P access enabled
When cuda_cross_device_copy is called from GPU 0 to GPU 1
Then cudaMemcpyPeerAsync is used
And data transfer completes within 1ms for 16MB
```

*Scenario 2: Staged copy fallback*
```gherkin
Given GPUs 0 and 2 do not have P2P access
When cuda_cross_device_copy is called from GPU 0 to GPU 2
Then staged copy through pinned memory is used
And data transfer completes correctly (slower than P2P)
```

**Checklist**:
- [ ] P2P path implemented using cudaMemcpyPeerAsync
- [ ] Staged copy path implemented
- [ ] Pinned staging buffer allocated in multi_init
- [ ] Stream parameter respected for async operations
- [ ] Benchmark shows P2P path is faster than staged
- [ ] Unit test validates data integrity

---

#### TASK-004: Multi-Device Allocation Functions

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | TASK-001 |
| **Milestone** | M1: Multi-GPU Infrastructure |

**Description**:
Implement `cuda_alloc_on_device` and `cuda_free_on_device` functions that explicitly allocate/free memory on a specific GPU device.

**Files to Create/Modify**:
- `gpu/cuda/memory.cu` (modify)
- `gpu/cuda/memory.h` (modify)

**Implementation Details**:
```c
int cuda_alloc_on_device(void** ptr, size_t size, int device_id) {
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);

    cudaError_t err = cudaMalloc(ptr, size);

    cudaSetDevice(current_device);  // Restore

    if (err != cudaSuccess) return -1;

    // Update memory tracking
    g_manager->devices[device_id].used_memory += size;
    return 0;
}

int cuda_free_on_device(void* ptr, int device_id) {
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);

    cudaFree(ptr);

    cudaSetDevice(current_device);
    return 0;
}
```

**Acceptance Criteria**:

*Scenario 1: Allocate on specific device*
```gherkin
Given multi-device manager is initialized with GPUs [0, 1]
When cuda_alloc_on_device is called for GPU 1 with size 1MB
Then memory is allocated on GPU 1
And devices[1].used_memory increases by 1MB
And current device context is restored after allocation
```

**Checklist**:
- [ ] `cuda_alloc_on_device` implemented
- [ ] `cuda_free_on_device` implemented
- [ ] Device context switching is clean (restore original)
- [ ] Memory tracking updated correctly
- [ ] Unit test allocates on multiple devices

---

#### TASK-005: Go Bindings for Multi-GPU Operations

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-001, TASK-002, TASK-003, TASK-004 |
| **Milestone** | M1: Multi-GPU Infrastructure |

**Description**:
Add CGO bindings in Go for all new multi-GPU CUDA functions. Follow existing patterns in `gpu/bindings/gpu.go`.

**Files to Create/Modify**:
- `gpu/bindings/gpu.go` (modify)
- `gpu/bindings/gpu.h` (modify)

**Implementation Details**:
```go
// Add to gpu.go

// InitMultiGPU initializes multiple GPU devices.
func InitMultiGPU(deviceIDs []int) error {
    if len(deviceIDs) == 0 {
        return errors.New("no devices specified")
    }
    result := C.cuda_multi_init(
        (*C.int)(unsafe.Pointer(&deviceIDs[0])),
        C.int(len(deviceIDs)),
    )
    if result != 0 {
        return fmt.Errorf("multi-GPU init failed: %d", result)
    }
    return nil
}

// CrossDeviceCopy copies data between GPUs.
func CrossDeviceCopy(dst unsafe.Pointer, dstDevice int,
                      src unsafe.Pointer, srcDevice int,
                      size uint64) error {
    result := C.cuda_cross_device_copy(
        dst, C.int(dstDevice),
        src, C.int(srcDevice),
        C.size_t(size), nil,
    )
    if result != 0 {
        return fmt.Errorf("cross-device copy failed: %d", result)
    }
    return nil
}
```

**Code Patterns to Follow**:
- Reference existing CGO patterns at `gpu/bindings/gpu.go:44-50` for error handling
- Follow existing function signature style (return error, use fmt.Errorf)

**Acceptance Criteria**:

*Scenario 1: Go code can initialize multi-GPU*
```gherkin
Given CUDA library is built with multi-GPU support
When Go calls InitMultiGPU([]int{0, 1})
Then both GPUs are initialized
And no error is returned
```

*Scenario 2: Go code can perform cross-device copy*
```gherkin
Given tensors allocated on GPU 0 and GPU 1
When Go calls CrossDeviceCopy from GPU 0 to GPU 1
Then data is correctly transferred
And verification shows identical data on both devices
```

**Checklist**:
- [ ] `InitMultiGPU` binding implemented
- [ ] `ShutdownMultiGPU` binding implemented
- [ ] `CrossDeviceCopy` binding implemented
- [ ] `AllocOnDevice` binding implemented
- [ ] `FreeOnDevice` binding implemented
- [ ] `GetDeviceContext` binding implemented
- [ ] gpu.h header updated with new function declarations
- [ ] Integration test passes in Go

---

### Milestone 2: Transport Layer

---

#### TASK-006: Transport Interface Definition

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | None |
| **Milestone** | M2: Transport Layer |

**Description**:
Define the `Transport` interface and `PeerDescriptor` struct that will be implemented by both local CUDA transport and remote P2P transport.

**Files to Create**:
- `pkg/transport/transport.go` (create)

**Implementation Details**:
```go
package transport

import (
    "context"
    "time"
)

// PeerDescriptor describes a compute peer
type PeerDescriptor struct {
    ID          string
    Address     string
    IsLocal     bool
    DeviceID    int      // GPU device ID (local peers)
    Layers      []int    // Assigned layer IDs
    TotalVRAM   uint64
    AvailVRAM   uint64
    Latency     time.Duration
}

// Transport defines the interface for activation transfer
type Transport interface {
    SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error
    RecvActivation(ctx context.Context, layerID int) (seqID uint64, data []byte, err error)
    Ping(ctx context.Context) (time.Duration, error)
    PeerInfo() PeerDescriptor
    Close() error
}
```

**Acceptance Criteria**:

*Scenario 1: Interface is implementable*
```gherkin
Given the Transport interface definition
When a new transport type implements all methods
Then the Go compiler accepts it as a valid Transport
```

**Checklist**:
- [ ] `PeerDescriptor` struct defined with all fields
- [ ] `Transport` interface defined with all methods
- [ ] Package documentation added
- [ ] Interface is minimal and sufficient

---

#### TASK-007: Transport Router Implementation

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-006 |
| **Milestone** | M2: Transport Layer |

**Description**:
Implement `TransportRouter` that routes activations to the appropriate transport (local or remote) based on layer-to-peer assignment.

**Files to Create/Modify**:
- `pkg/transport/router.go` (create)

**Implementation Details**:
```go
type TransportRouter struct {
    localTransports  map[int]Transport     // Device ID -> transport
    remoteTransports map[string]Transport  // Peer ID -> transport
    layerToPeer      map[int]string        // Layer ID -> Peer ID
    mu               sync.RWMutex
}

func (r *TransportRouter) RouteActivation(ctx context.Context, layerID int,
                                          seqID uint64, data []byte) error {
    r.mu.RLock()
    peerID, ok := r.layerToPeer[layerID]
    r.mu.RUnlock()

    if !ok {
        return fmt.Errorf("no peer assigned for layer %d", layerID)
    }

    // Check local first
    if transport, ok := r.localTransports[deviceIDFromPeer(peerID)]; ok {
        return transport.SendActivation(ctx, layerID, seqID, data)
    }

    // Then remote
    if transport, ok := r.remoteTransports[peerID]; ok {
        return transport.SendActivation(ctx, layerID, seqID, data)
    }

    return fmt.Errorf("no transport for peer %s", peerID)
}
```

**Acceptance Criteria**:

*Scenario 1: Route to local transport*
```gherkin
Given layer 5 is assigned to local GPU 1
When RouteActivation is called for layer 5
Then the local CUDA transport for GPU 1 is used
And activation is sent successfully
```

*Scenario 2: Route to remote transport*
```gherkin
Given layer 10 is assigned to remote peer "peer-A"
When RouteActivation is called for layer 10
Then the P2P transport for "peer-A" is used
And activation is sent over the network
```

**Checklist**:
- [ ] `TransportRouter` struct with thread-safe maps
- [ ] `RouteActivation` method routes correctly
- [ ] `ReceiveActivation` method implemented
- [ ] `RegisterLocalTransport` method implemented
- [ ] `RegisterRemoteTransport` method implemented
- [ ] `UpdateLayerAssignment` method implemented
- [ ] Unit tests for routing logic

---

#### TASK-008: Local CUDA Transport Implementation

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-005, TASK-006 |
| **Milestone** | M2: Transport Layer |

**Description**:
Implement `CUDATransport` that implements the `Transport` interface for local multi-GPU communication using CUDA P2P or staged copies.

**Files to Create**:
- `pkg/transport/local.go` (create)

**Implementation Details**:
```go
type CUDATransport struct {
    srcDevice   int
    dstDevice   int
    recvChan    chan *activationMessage
    bufferPool  *sync.Pool
    peerInfo    PeerDescriptor
    mu          sync.Mutex
}

func (t *CUDATransport) SendActivation(ctx context.Context, layerID int,
                                        seqID uint64, data []byte) error {
    // Allocate on destination device
    dstPtr, err := bindings.AllocOnDevice(uint64(len(data)), t.dstDevice)
    if err != nil {
        return err
    }

    // Copy using cross-device transfer
    srcPtr := unsafe.Pointer(&data[0])
    err = bindings.CrossDeviceCopy(dstPtr, t.dstDevice, srcPtr, t.srcDevice, uint64(len(data)))
    if err != nil {
        return err
    }

    // Signal receive channel
    t.recvChan <- &activationMessage{layerID: layerID, seqID: seqID, ptr: dstPtr}
    return nil
}
```

**Acceptance Criteria**:

*Scenario 1: Send activation between local GPUs*
```gherkin
Given CUDATransport configured for GPU 0 -> GPU 1
When SendActivation is called with 16MB activation tensor
Then data is transferred to GPU 1
And transfer completes in < 5ms (P2P) or < 20ms (staged)
```

*Scenario 2: Receive activation*
```gherkin
Given activation was sent to GPU 1
When RecvActivation is called
Then the activation data is returned
And data matches what was sent
```

**Checklist**:
- [ ] `CUDATransport` struct implements `Transport` interface
- [ ] `SendActivation` uses cross-device copy
- [ ] `RecvActivation` waits on channel
- [ ] Buffer pool reduces allocations
- [ ] `Ping` measures local latency
- [ ] `Close` cleans up resources
- [ ] Unit test verifies data integrity

---

### Milestone 3: libp2p Integration

---

#### TASK-009: libp2p Host Setup

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | None |
| **Milestone** | M3: libp2p Integration |

**Description**:
Create the libp2p host with TCP and QUIC transports, relay support, and hole punching enabled.

**Files to Create**:
- `p2p/host.go` (create)
- `go.mod` (modify - add libp2p dependencies)

**Implementation Details**:
```go
package p2p

import (
    "context"
    "fmt"

    "github.com/libp2p/go-libp2p"
    "github.com/libp2p/go-libp2p/core/host"
)

func NewHost(ctx context.Context, listenPort int) (host.Host, error) {
    h, err := libp2p.New(
        libp2p.ListenAddrStrings(
            fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", listenPort),
            fmt.Sprintf("/ip4/0.0.0.0/udp/%d/quic-v1", listenPort),
        ),
        libp2p.EnableHolePunching(),
        libp2p.EnableRelay(),
    )
    if err != nil {
        return nil, err
    }
    return h, nil
}
```

**Acceptance Criteria**:

*Scenario 1: Create libp2p host*
```gherkin
Given port 9000 is available
When NewHost is called with port 9000
Then a libp2p host is created
And host listens on TCP and QUIC
And host has a unique peer ID
```

**Checklist**:
- [ ] go.mod updated with libp2p v0.46.0
- [ ] Host created with TCP and QUIC transports
- [ ] Hole punching enabled for NAT traversal
- [ ] Relay enabled for connectivity
- [ ] Host can be closed cleanly
- [ ] Unit test creates and destroys host

---

#### TASK-010: mDNS Local Discovery

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | TASK-009 |
| **Milestone** | M3: libp2p Integration |

**Description**:
Implement mDNS-based peer discovery for finding peers on the local network.

**Files to Create/Modify**:
- `p2p/discovery.go` (create)

**Implementation Details**:
```go
const ServiceTag = "neurogrid"

type Discovery struct {
    host     host.Host
    peerChan chan peer.AddrInfo
}

func (d *Discovery) SetupMDNS() error {
    mdnsService := mdns.NewMdnsService(d.host, ServiceTag, d)
    return mdnsService.Start()
}

// HandlePeerFound implements mdns.Notifee
func (d *Discovery) HandlePeerFound(pi peer.AddrInfo) {
    if pi.ID == d.host.ID() {
        return // Skip self
    }
    d.peerChan <- pi
}
```

**Acceptance Criteria**:

*Scenario 1: Discover local peer via mDNS*
```gherkin
Given two hosts running on the same LAN
When both hosts start mDNS discovery
Then each host discovers the other
And peer info is sent to the peerChan
```

**Checklist**:
- [ ] mDNS service created with service tag
- [ ] `HandlePeerFound` filters out self
- [ ] Discovered peers sent to channel
- [ ] Integration test with two processes

---

#### TASK-011: DHT Remote Discovery

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-009 |
| **Milestone** | M3: libp2p Integration |

**Description**:
Implement Kademlia DHT for discovering peers across the internet.

**Files to Create/Modify**:
- `p2p/discovery.go` (modify)

**Implementation Details**:
```go
func (d *Discovery) SetupDHT(ctx context.Context) error {
    kadDHT, err := dht.New(ctx, d.host, dht.Mode(dht.ModeAutoServer))
    if err != nil {
        return err
    }

    if err := kadDHT.Bootstrap(ctx); err != nil {
        return err
    }

    d.dht = kadDHT

    // Start periodic peer discovery
    go d.discoverPeers(ctx)

    return nil
}

func (d *Discovery) discoverPeers(ctx context.Context) {
    routingDiscovery := drouting.NewRoutingDiscovery(d.dht)
    drouting.Advertise(ctx, routingDiscovery, ServiceTag)

    ticker := time.NewTicker(time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            peerChan, err := routingDiscovery.FindPeers(ctx, ServiceTag)
            if err != nil {
                continue
            }
            for pi := range peerChan {
                if pi.ID != d.host.ID() {
                    d.peerChan <- pi
                }
            }
        }
    }
}
```

**Acceptance Criteria**:

*Scenario 1: Bootstrap DHT*
```gherkin
Given a libp2p host
When SetupDHT is called
Then DHT bootstraps successfully
And host advertises on the service tag
```

*Scenario 2: Discover remote peer*
```gherkin
Given two hosts connected to IPFS bootstrap nodes
When both hosts advertise and search
Then they discover each other via DHT
```

**Checklist**:
- [ ] DHT created in ModeAutoServer
- [ ] Bootstrap called successfully
- [ ] Periodic discovery runs every minute
- [ ] Peers advertised under service tag
- [ ] Found peers sent to channel
- [ ] Unit test mocks DHT behavior

---

#### TASK-012: Tensor Transfer Protocol

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-009 |
| **Milestone** | M3: libp2p Integration |

**Description**:
Implement the `/neurogrid/tensor/1.0.0` protocol for transferring activation tensors between peers.

**Files to Create**:
- `p2p/protocol.go` (create)

**Implementation Details**:
```go
const (
    TensorProtocolID = "/neurogrid/tensor/1.0.0"
    PingProtocolID   = "/neurogrid/ping/1.0.0"
    BufferSize       = 4 * 1024 * 1024 // 4MB
)

// Wire format: [4B layerID][8B seqID][8B dataLen][data...]

func (p *Protocol) handleIncoming(s network.Stream) {
    defer s.Close()
    s.SetDeadline(time.Now().Add(30 * time.Second))

    reader := bufio.NewReaderSize(s, BufferSize)

    header := make([]byte, 20)
    if _, err := io.ReadFull(reader, header); err != nil {
        return
    }

    layerID := int(binary.BigEndian.Uint32(header[0:4]))
    seqID := binary.BigEndian.Uint64(header[4:12])
    dataLen := binary.BigEndian.Uint64(header[12:20])

    data := make([]byte, dataLen)
    if _, err := io.ReadFull(reader, data); err != nil {
        return
    }

    p.recvChan <- &TensorMessage{LayerID: layerID, SeqID: seqID, Data: data}
}
```

**Acceptance Criteria**:

*Scenario 1: Send tensor over protocol*
```gherkin
Given two connected libp2p peers
When peer A sends a 16MB tensor via tensor protocol
Then peer B receives complete tensor data
And layerID and seqID are correctly decoded
```

*Scenario 2: Handle timeout*
```gherkin
Given a slow network connection
When tensor transfer exceeds 30 second deadline
Then stream is closed
And error is returned to sender
```

**Checklist**:
- [ ] Protocol ID registered
- [ ] Binary header format implemented
- [ ] Buffered I/O for performance
- [ ] Deadline set on streams
- [ ] Stream handler registered with host
- [ ] Benchmark shows > 100MB/s on local network

---

#### TASK-013: P2P Transport Implementation

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-006, TASK-012 |
| **Milestone** | M3: libp2p Integration |

**Description**:
Implement `P2PTransport` that implements the `Transport` interface using libp2p streams.

**Files to Create**:
- `pkg/transport/p2p.go` (create)

**Implementation Details**:
```go
type P2PTransport struct {
    host     host.Host
    peerID   peer.ID
    peerInfo PeerDescriptor
    recvChan chan *activationMessage
}

func (t *P2PTransport) SendActivation(ctx context.Context, layerID int,
                                       seqID uint64, data []byte) error {
    stream, err := t.host.NewStream(ctx, t.peerID, TensorProtocolID)
    if err != nil {
        return fmt.Errorf("failed to open stream: %w", err)
    }
    defer stream.Close()

    stream.SetDeadline(time.Now().Add(30 * time.Second))
    writer := bufio.NewWriterSize(stream, BufferSize)

    // Write header
    header := make([]byte, 20)
    binary.BigEndian.PutUint32(header[0:4], uint32(layerID))
    binary.BigEndian.PutUint64(header[4:12], seqID)
    binary.BigEndian.PutUint64(header[12:20], uint64(len(data)))

    if _, err := writer.Write(header); err != nil {
        return err
    }
    if _, err := writer.Write(data); err != nil {
        return err
    }

    return writer.Flush()
}
```

**Acceptance Criteria**:

*Scenario 1: P2P transport implements interface*
```gherkin
Given P2PTransport configured with peer ID
When SendActivation is called
Then stream is opened to peer
And tensor data is sent with header
And stream is closed after send
```

**Checklist**:
- [ ] `P2PTransport` implements `Transport` interface
- [ ] `SendActivation` opens stream and sends data
- [ ] `RecvActivation` reads from channel
- [ ] `Ping` measures RTT to peer
- [ ] `Close` closes all connections
- [ ] Integration test between two processes

---

#### TASK-014: FlatBuffers Schema and Code Generation

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | None |
| **Milestone** | M3: libp2p Integration |

**Description**:
Create FlatBuffers schema for tensor serialization and generate Go code.

**Files to Create**:
- `schemas/tensor.fbs` (create)
- `pkg/types/generated/` (generated)
- `Makefile` (modify - add flatbuffers target)

**Implementation Details**:
```flatbuffers
namespace neurogrid;

enum DType : byte {
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    BFloat16 = 3
}

table TensorMeta {
    shape: [int64];
    dtype: DType;
    device_id: int32;
    layer_id: int32;
    position: int32;
}

table Activation {
    meta: TensorMeta;
    data: [ubyte];
    sequence_id: uint64;
}

root_type Activation;
```

**Acceptance Criteria**:

*Scenario 1: Generate Go code*
```gherkin
Given tensor.fbs schema file
When make flatbuffers is run
Then Go code is generated in pkg/types/generated/
And code compiles without errors
```

**Checklist**:
- [ ] FlatBuffers schema defined
- [ ] Makefile target added
- [ ] Go code generated successfully
- [ ] Serialization/deserialization works
- [ ] Zero-copy read verified

---

### Milestone 4: Pipeline Scheduler

---

#### TASK-015: VRAM Memory Tracker

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | TASK-006 |
| **Milestone** | M4: Pipeline Scheduler |

**Description**:
Implement VRAM tracking for each peer to support intelligent layer assignment.

**Files to Create**:
- `pkg/scheduler/memory.go` (create)

**Implementation Details**:
```go
type VRAMTracker struct {
    peers     map[string]*PeerVRAM
    mu        sync.RWMutex
}

type PeerVRAM struct {
    PeerID    string
    Total     uint64
    Used      uint64
    Reserved  uint64  // For pending allocations
}

func (t *VRAMTracker) Reserve(peerID string, size uint64) error {
    t.mu.Lock()
    defer t.mu.Unlock()

    peer, ok := t.peers[peerID]
    if !ok {
        return fmt.Errorf("unknown peer: %s", peerID)
    }

    available := peer.Total - peer.Used - peer.Reserved
    if size > available {
        return fmt.Errorf("insufficient VRAM: need %d, have %d", size, available)
    }

    peer.Reserved += size
    return nil
}
```

**Acceptance Criteria**:

*Scenario 1: Track VRAM usage*
```gherkin
Given a peer with 8GB total VRAM and 2GB used
When Reserve is called for 4GB
Then reservation succeeds
And available VRAM shows 2GB remaining
```

*Scenario 2: Reject over-allocation*
```gherkin
Given a peer with 8GB total and 6GB used/reserved
When Reserve is called for 4GB
Then error is returned
And reservation is not made
```

**Checklist**:
- [ ] `VRAMTracker` struct with thread-safe operations
- [ ] `RegisterPeer` adds peer with VRAM info
- [ ] `Reserve` tracks pending allocations
- [ ] `Commit` converts reservation to usage
- [ ] `Release` frees memory
- [ ] Unit tests for all operations

---

#### TASK-016: Layer Memory Estimation

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | None |
| **Milestone** | M4: Pipeline Scheduler |

**Description**:
Implement accurate memory estimation for transformer layers based on model configuration.

**Files to Create/Modify**:
- `pkg/scheduler/scheduler.go` (create)

**Implementation Details**:
```go
func (s *Scheduler) estimateLayerMemory() uint64 {
    cfg := s.config

    // Weights (INT8): Q, K, V, O projections
    attnWeights := uint64(cfg.HiddenSize * cfg.HiddenSize * 4)

    // FFN weights (INT8): gate, up, down
    ffnWeights := uint64(cfg.HiddenSize * cfg.IntermediateSize * 3)

    // Scales (FP32): one per column
    numScaleParams := cfg.HiddenSize*4 + cfg.IntermediateSize*3
    scales := uint64(numScaleParams * 4)

    // Norms (FP16): attn_norm + ffn_norm
    norms := uint64(cfg.HiddenSize * 2 * 2)

    // KV cache (FP16, 2048 tokens max)
    kvCache := uint64(2048 * cfg.NumKVHeads * cfg.HeadDim * 2 * 2)

    // Activation buffer (FP16)
    activations := uint64(4096 * cfg.HiddenSize * 2)

    total := attnWeights + ffnWeights + scales + norms + kvCache + activations

    // 20% overhead for alignment and fragmentation
    return uint64(float64(total) * 1.2)
}
```

**Acceptance Criteria**:

*Scenario 1: Estimate 7B layer memory*
```gherkin
Given Llama 7B configuration (hidden=4096, intermediate=11008)
When estimateLayerMemory is called
Then returned value is approximately 200-250MB per layer
And estimate includes KV cache and activations
```

**Checklist**:
- [ ] Weight memory calculation correct
- [ ] KV cache memory included
- [ ] Activation buffer included
- [ ] Overhead factor applied
- [ ] Unit test validates against known values

---

#### TASK-017: Layer Assignment Algorithm

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-015, TASK-016 |
| **Milestone** | M4: Pipeline Scheduler |

**Description**:
Implement VRAM-aware greedy algorithm to assign layers to peers.

**Files to Create/Modify**:
- `pkg/scheduler/scheduler.go` (modify)

**Implementation Details**:
```go
func (s *Scheduler) ComputeAssignments() ([]LayerAssignment, error) {
    if len(s.peers) == 0 {
        return nil, fmt.Errorf("no peers registered")
    }

    layerMemory := s.estimateLayerMemory()

    // Sort peers by available VRAM (descending)
    sortedPeers := make([]PeerDescriptor, len(s.peers))
    copy(sortedPeers, s.peers)
    sort.Slice(sortedPeers, func(i, j int) bool {
        return sortedPeers[i].AvailVRAM > sortedPeers[j].AvailVRAM
    })

    // Track available VRAM per peer
    peerVRAM := make(map[string]uint64)
    for _, p := range sortedPeers {
        peerVRAM[p.ID] = p.AvailVRAM
    }

    // Greedy assignment
    assignments := make([]LayerAssignment, 0, s.config.NumLayers)
    for layerID := 0; layerID < s.config.NumLayers; layerID++ {
        assigned := false
        for i := range sortedPeers {
            if peerVRAM[sortedPeers[i].ID] >= layerMemory {
                assignments = append(assignments, LayerAssignment{
                    LayerID:    layerID,
                    PeerID:     sortedPeers[i].ID,
                    DeviceID:   sortedPeers[i].DeviceID,
                    MemoryUsed: layerMemory,
                })
                peerVRAM[sortedPeers[i].ID] -= layerMemory
                assigned = true
                break
            }
        }
        if !assigned {
            return nil, fmt.Errorf("insufficient VRAM for layer %d", layerID)
        }
    }

    return assignments, nil
}
```

**Acceptance Criteria**:

*Scenario 1: Distribute layers across heterogeneous GPUs*
```gherkin
Given 5 GPUs with VRAM: 24GB, 8GB, 8GB, 8GB, 8GB
And 32 layers each requiring ~250MB
When ComputeAssignments is called
Then all 32 layers are assigned
And larger GPU gets more layers
And no GPU exceeds its VRAM limit
```

*Scenario 2: Handle insufficient VRAM*
```gherkin
Given 2 GPUs with 4GB each
And 80 layers each requiring 250MB (total 20GB needed)
When ComputeAssignments is called
Then error is returned indicating insufficient VRAM
```

**Checklist**:
- [ ] Greedy algorithm implemented
- [ ] Peers sorted by available VRAM
- [ ] All layers assigned or error returned
- [ ] Memory tracked per peer during assignment
- [ ] Unit test with heterogeneous VRAM
- [ ] Unit test for insufficient VRAM case

---

#### TASK-018: Prefetch Coordinator

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-017 |
| **Milestone** | M4: Pipeline Scheduler |

**Description**:
Implement prefetching logic to overlap computation with data transfer.

**Files to Create**:
- `pkg/scheduler/prefetch.go` (create)

**Implementation Details**:
```go
type PrefetchCoordinator struct {
    scheduler    *Scheduler
    router       *transport.TransportRouter
    prefetchChan chan PrefetchRequest
    inFlight     map[int]bool  // layerID -> prefetch in progress
    mu           sync.Mutex
}

type PrefetchRequest struct {
    LayerID     int
    Position    int
    Priority    int
}

func (p *PrefetchCoordinator) StartPrefetch(layerID, position int) {
    p.mu.Lock()
    defer p.mu.Unlock()

    if p.inFlight[layerID] {
        return // Already prefetching
    }

    p.inFlight[layerID] = true
    p.prefetchChan <- PrefetchRequest{
        LayerID:  layerID,
        Position: position,
    }
}
```

**Acceptance Criteria**:

*Scenario 1: Prefetch next layer*
```gherkin
Given layer N is currently executing on peer A
When prefetch is triggered for layer N+1
Then activation is sent to peer B (assigned to N+1) asynchronously
And prefetch does not block current execution
```

**Checklist**:
- [ ] `PrefetchCoordinator` struct defined
- [ ] Prefetch requests processed asynchronously
- [ ] Duplicate prefetch requests deduplicated
- [ ] Prefetch can be cancelled
- [ ] Unit test verifies async behavior

---

### Milestone 5: Inference Engine

---

#### TASK-019: Distributed Inference Engine Core

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 8 |
| **Estimated Duration** | 3 days |
| **Dependencies** | TASK-007, TASK-017 |
| **Milestone** | M5: Inference Engine |

**Description**:
Implement the core inference engine that orchestrates distributed forward passes.

**Files to Create**:
- `pkg/inference/engine.go` (create)

**Implementation Details**:
```go
type Engine struct {
    config     *types.LlamaConfig
    scheduler  *scheduler.Scheduler
    router     *transport.TransportRouter
    tokenizer  *Tokenizer
    sampler    *Sampler
    kvCaches   map[int]*KVCache  // layerID -> cache
    embeddings unsafe.Pointer    // Token embedding matrix
    lmHead     unsafe.Pointer    // Output projection
    mu         sync.RWMutex
}

func (e *Engine) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
    // Tokenize input
    inputTokens, err := e.tokenizer.Encode(req.Prompt)
    if err != nil {
        return nil, fmt.Errorf("tokenization failed: %w", err)
    }

    // Prefill phase
    hidden, err := e.prefill(ctx, inputTokens)
    if err != nil {
        return nil, fmt.Errorf("prefill failed: %w", err)
    }

    // Autoregressive decode
    outputTokens := make([]int, 0, req.MaxTokens)
    for i := 0; i < req.MaxTokens; i++ {
        logits, err := e.forwardAllLayers(ctx, hidden, len(inputTokens)+i)
        if err != nil {
            return nil, err
        }

        nextToken := e.sampler.Sample(logits, req.Temperature, req.TopP)
        outputTokens = append(outputTokens, nextToken)

        if nextToken == e.tokenizer.EOSToken() {
            break
        }

        hidden, err = e.embedToken(ctx, nextToken)
        if err != nil {
            return nil, err
        }
    }

    // Decode output
    outputText, err := e.tokenizer.Decode(outputTokens)
    return &GenerateResponse{Text: outputText, TokenCount: len(outputTokens)}, nil
}
```

**Acceptance Criteria**:

*Scenario 1: Complete inference request*
```gherkin
Given model loaded and distributed across peers
When Generate is called with prompt "Hello"
Then prefill processes all input tokens
And autoregressive decode generates output
And output is coherent text
```

*Scenario 2: Handle distributed forward pass*
```gherkin
Given layers 0-5 on local GPU, layers 6-31 on remote peers
When forwardAllLayers is called
Then hidden states are transferred between peers at layer boundaries
And final logits are returned
```

**Checklist**:
- [ ] `Engine` struct with all dependencies
- [ ] `Generate` method implements full pipeline
- [ ] `prefill` handles batch processing
- [ ] `forwardAllLayers` handles distributed execution
- [ ] `embedToken` looks up token embedding
- [ ] Context cancellation respected
- [ ] Integration test with mock peers

---

#### TASK-020: Distributed KV Cache Manager

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-019 |
| **Milestone** | M5: Inference Engine |

**Description**:
Implement distributed KV cache that lives on the same peer as its layer.

**Files to Create**:
- `pkg/inference/kvcache.go` (create)

**Implementation Details**:
```go
type DistributedKVCache struct {
    layerID     int
    peerID      string
    deviceID    int
    maxSeqLen   int
    numKVHeads  int
    headDim     int
    currentLen  int
    localCache  *bindings.KVCache  // nil if remote
    mu          sync.RWMutex
}

func (c *DistributedKVCache) Update(ctx context.Context, k, v []byte, position int) error {
    c.mu.Lock()
    defer c.mu.Unlock()

    if c.localCache != nil {
        // Local update via CGO
        return bindings.UpdateKVCache(c.localCache, k, v, position)
    }

    // Remote update requires coordination with peer
    // This is handled by the remote worker's layer execution
    return nil
}
```

**Acceptance Criteria**:

*Scenario 1: Update local KV cache*
```gherkin
Given KV cache for layer 0 on local GPU
When Update is called with new K, V at position 5
Then cache is updated via CUDA bindings
And currentLen is updated
```

*Scenario 2: KV cache follows layer*
```gherkin
Given layer 10 assigned to remote peer
When KV cache is accessed for layer 10
Then cache operations are delegated to remote peer
```

**Checklist**:
- [ ] `DistributedKVCache` struct defined
- [ ] Local cache update via CGO
- [ ] Remote cache conceptually handled by worker
- [ ] Cache clearing for new sequences
- [ ] Thread-safe access
- [ ] Unit test for local cache

---

#### TASK-021: Token Sampler

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | None |
| **Milestone** | M5: Inference Engine |

**Description**:
Implement temperature and top-p sampling for token generation.

**Files to Create**:
- `pkg/inference/sampler.go` (create)

**Implementation Details**:
```go
type Sampler struct {
    rng *rand.Rand
}

func (s *Sampler) Sample(logits []float32, temperature, topP float32) int {
    if temperature <= 0 {
        // Greedy decoding
        return argmax(logits)
    }

    // Apply temperature
    scaled := make([]float32, len(logits))
    for i, l := range logits {
        scaled[i] = l / temperature
    }

    // Softmax
    probs := softmax(scaled)

    // Top-p (nucleus) sampling
    if topP < 1.0 {
        probs = nucleusSample(probs, topP)
    }

    // Sample from distribution
    return sampleFromDistribution(s.rng, probs)
}

func nucleusSample(probs []float32, p float32) []float32 {
    // Sort by probability descending
    indices := argsort(probs)

    cumSum := float32(0)
    cutoff := -1
    for i, idx := range indices {
        cumSum += probs[idx]
        if cumSum >= p {
            cutoff = i
            break
        }
    }

    // Zero out tokens outside nucleus
    result := make([]float32, len(probs))
    for i := 0; i <= cutoff; i++ {
        result[indices[i]] = probs[indices[i]]
    }

    // Renormalize
    return normalize(result)
}
```

**Acceptance Criteria**:

*Scenario 1: Greedy decoding*
```gherkin
Given logits with highest value at index 42
When Sample is called with temperature=0
Then token 42 is returned
```

*Scenario 2: Top-p sampling*
```gherkin
Given logits with clear top tokens
When Sample is called with temperature=0.7, topP=0.9
Then returned token is from top 90% probability mass
```

**Checklist**:
- [ ] Temperature scaling implemented
- [ ] Softmax implemented
- [ ] Top-p (nucleus) sampling implemented
- [ ] Greedy mode for temperature <= 0
- [ ] Reproducible with seed
- [ ] Unit test for all modes

---

### Milestone 6: Model Loading & Tokenization

---

#### TASK-022: SafeTensors Weight Loader

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-005 |
| **Milestone** | M6: Model Loading |

**Description**:
Implement loading of model weights from SafeTensors format.

**Files to Create**:
- `pkg/model/loader.go` (create)

**Implementation Details**:
```go
type WeightLoader struct {
    basePath string
    index    map[string]TensorInfo
}

type TensorInfo struct {
    File    string
    Offset  uint64
    Size    uint64
    Shape   []int
    Dtype   string
}

func (l *WeightLoader) LoadLayerWeights(layerID, deviceID int) (*bindings.LayerWeights, error) {
    // Load weight tensors for this layer
    prefix := fmt.Sprintf("model.layers.%d.", layerID)

    // Attention weights
    qWeight, err := l.loadTensor(prefix + "self_attn.q_proj.weight")
    kWeight, err := l.loadTensor(prefix + "self_attn.k_proj.weight")
    vWeight, err := l.loadTensor(prefix + "self_attn.v_proj.weight")
    oWeight, err := l.loadTensor(prefix + "self_attn.o_proj.weight")

    // FFN weights
    gateWeight, err := l.loadTensor(prefix + "mlp.gate_proj.weight")
    upWeight, err := l.loadTensor(prefix + "mlp.up_proj.weight")
    downWeight, err := l.loadTensor(prefix + "mlp.down_proj.weight")

    // Norms
    attnNorm, err := l.loadTensor(prefix + "input_layernorm.weight")
    ffnNorm, err := l.loadTensor(prefix + "post_attention_layernorm.weight")

    // Pack into LayerWeights structure and upload to GPU
    return packAndUpload(deviceID, qWeight, kWeight, vWeight, oWeight,
                         gateWeight, upWeight, downWeight, attnNorm, ffnNorm)
}
```

**Acceptance Criteria**:

*Scenario 1: Load layer weights*
```gherkin
Given SafeTensors model at /models/llama-7b/
When LoadLayerWeights is called for layer 0, device 0
Then all weight tensors are loaded
And weights are uploaded to GPU 0
And LayerWeights handle is returned
```

**Checklist**:
- [ ] SafeTensors index parsing
- [ ] Memory-mapped file access
- [ ] Per-layer weight loading
- [ ] GPU upload via bindings
- [ ] Error handling for missing weights
- [ ] Unit test with dummy weights

---

#### TASK-023: Sentencepiece Tokenizer

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | None |
| **Milestone** | M6: Model Loading |

**Description**:
Implement tokenizer wrapper using sentencepiece for Llama models.

**Files to Create**:
- `pkg/model/tokenizer.go` (create)
- `go.mod` (modify - add sentencepiece binding)

**Implementation Details**:
```go
type Tokenizer struct {
    model     *sentencepiece.Model
    bosToken  int
    eosToken  int
    padToken  int
}

func NewTokenizer(modelPath string) (*Tokenizer, error) {
    model, err := sentencepiece.Load(modelPath)
    if err != nil {
        return nil, err
    }

    return &Tokenizer{
        model:    model,
        bosToken: 1,  // <s>
        eosToken: 2,  // </s>
        padToken: 0,  // <unk> as pad
    }, nil
}

func (t *Tokenizer) Encode(text string) ([]int, error) {
    tokens, err := t.model.Encode(text)
    if err != nil {
        return nil, err
    }

    // Add BOS token
    result := make([]int, 0, len(tokens)+1)
    result = append(result, t.bosToken)
    result = append(result, tokens...)

    return result, nil
}

func (t *Tokenizer) Decode(tokens []int) (string, error) {
    // Filter special tokens
    filtered := make([]int, 0, len(tokens))
    for _, tok := range tokens {
        if tok != t.bosToken && tok != t.eosToken && tok != t.padToken {
            filtered = append(filtered, tok)
        }
    }

    return t.model.Decode(filtered)
}
```

**Acceptance Criteria**:

*Scenario 1: Encode text*
```gherkin
Given tokenizer loaded with llama tokenizer.model
When Encode is called with "Hello, world!"
Then token IDs are returned
And BOS token is prepended
```

*Scenario 2: Decode tokens*
```gherkin
Given token IDs from a generation
When Decode is called
Then original text is reconstructed
And special tokens are filtered out
```

**Checklist**:
- [ ] Sentencepiece binding integrated
- [ ] `NewTokenizer` loads model file
- [ ] `Encode` adds BOS token
- [ ] `Decode` filters special tokens
- [ ] `EOSToken()` returns correct ID
- [ ] Unit test with known encodings

---

#### TASK-024: Distributed Model Setup

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-017, TASK-022 |
| **Milestone** | M6: Model Loading |

**Description**:
Implement distributed model loading that loads layers to assigned peers.

**Files to Create**:
- `pkg/model/distributed.go` (create)

**Implementation Details**:
```go
type DistributedModel struct {
    config      *types.LlamaConfig
    scheduler   *scheduler.Scheduler
    loader      *WeightLoader
    embeddings  *types.Tensor  // On coordinator
    lmHead      *types.Tensor  // On coordinator
    layerStatus map[int]bool   // layerID -> loaded
}

func (m *DistributedModel) LoadToCluster(ctx context.Context) error {
    assignments, err := m.scheduler.ComputeAssignments()
    if err != nil {
        return err
    }

    // Load embeddings on coordinator
    m.embeddings, err = m.loader.LoadTensor("model.embed_tokens.weight")

    // Load lm_head on coordinator
    m.lmHead, err = m.loader.LoadTensor("lm_head.weight")

    // Load layers in parallel
    errChan := make(chan error, len(assignments))
    for _, assign := range assignments {
        go func(a scheduler.LayerAssignment) {
            if a.PeerID == "local" {
                err := m.loadLocalLayer(a.LayerID, a.DeviceID)
                errChan <- err
            } else {
                err := m.sendLayerToPeer(ctx, a)
                errChan <- err
            }
        }(assign)
    }

    // Collect errors
    for range assignments {
        if err := <-errChan; err != nil {
            return err
        }
    }

    return nil
}
```

**Acceptance Criteria**:

*Scenario 1: Load model to cluster*
```gherkin
Given scheduler has assigned layers to 5 peers
When LoadToCluster is called
Then embeddings and lm_head load to coordinator
And each layer loads to its assigned peer
And all layers report loaded status
```

**Checklist**:
- [ ] Embeddings loaded on coordinator
- [ ] lm_head loaded on coordinator
- [ ] Parallel layer loading
- [ ] Remote layer transfer to peers
- [ ] Error aggregation
- [ ] Progress reporting

---

### Milestone 7: HTTP API

---

#### TASK-025: HTTP Server Setup

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | None |
| **Milestone** | M7: HTTP API |

**Description**:
Create HTTP server with routing, middleware, and graceful shutdown.

**Files to Create**:
- `api/server.go` (create)
- `api/middleware.go` (create)

**Implementation Details**:
```go
type Server struct {
    engine *inference.Engine
    addr   string
    server *http.Server
}

func NewServer(engine *inference.Engine, addr string) *Server {
    return &Server{
        engine: engine,
        addr:   addr,
    }
}

func (s *Server) Start() error {
    mux := http.NewServeMux()

    mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
    mux.HandleFunc("/v1/models", s.handleListModels)
    mux.HandleFunc("/health", s.handleHealth)

    handler := corsMiddleware(loggingMiddleware(mux))

    s.server = &http.Server{
        Addr:         s.addr,
        Handler:      handler,
        ReadTimeout:  30 * time.Second,
        WriteTimeout: 120 * time.Second,
    }

    return s.server.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
    return s.server.Shutdown(ctx)
}
```

**Acceptance Criteria**:

*Scenario 1: Start server*
```gherkin
Given inference engine is initialized
When Server.Start is called on port 8080
Then server listens on 0.0.0.0:8080
And /health endpoint returns 200 OK
```

**Checklist**:
- [ ] HTTP server with timeouts
- [ ] CORS middleware
- [ ] Logging middleware
- [ ] Graceful shutdown
- [ ] Health endpoint
- [ ] Unit test for middleware

---

#### TASK-026: Chat Completions Handler

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-019, TASK-025 |
| **Milestone** | M7: HTTP API |

**Description**:
Implement `/v1/chat/completions` endpoint with OpenAI-compatible request/response format.

**Files to Create**:
- `api/handlers.go` (create)
- `api/types.go` (create)

**Implementation Details**:
```go
type ChatCompletionRequest struct {
    Model       string    `json:"model"`
    Messages    []Message `json:"messages"`
    MaxTokens   int       `json:"max_tokens,omitempty"`
    Temperature float32   `json:"temperature,omitempty"`
    TopP        float32   `json:"top_p,omitempty"`
    Stream      bool      `json:"stream,omitempty"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req ChatCompletionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // Build prompt from messages
    prompt := buildLlamaPrompt(req.Messages)

    // Generate
    genReq := &inference.GenerateRequest{
        Prompt:      prompt,
        MaxTokens:   req.MaxTokens,
        Temperature: req.Temperature,
        TopP:        req.TopP,
    }

    genResp, err := s.engine.Generate(r.Context(), genReq)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    // Build response
    resp := ChatCompletionResponse{
        ID:      generateID(),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   req.Model,
        Choices: []Choice{{
            Index:        0,
            Message:      Message{Role: "assistant", Content: genResp.Text},
            FinishReason: genResp.FinishReason,
        }},
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

func buildLlamaPrompt(messages []Message) string {
    var prompt strings.Builder
    for _, msg := range messages {
        switch msg.Role {
        case "system":
            prompt.WriteString("[INST] <<SYS>>\n")
            prompt.WriteString(msg.Content)
            prompt.WriteString("\n<</SYS>>\n\n")
        case "user":
            prompt.WriteString(msg.Content)
            prompt.WriteString(" [/INST] ")
        case "assistant":
            prompt.WriteString(msg.Content)
            prompt.WriteString(" </s><s>[INST] ")
        }
    }
    return prompt.String()
}
```

**Acceptance Criteria**:

*Scenario 1: Successful chat completion*
```gherkin
Given POST /v1/chat/completions with valid request
When request contains messages with user asking "What is 2+2?"
Then response contains assistant message with answer
And response format matches OpenAI API
```

*Scenario 2: Invalid request*
```gherkin
Given POST /v1/chat/completions with malformed JSON
When request is processed
Then 400 Bad Request is returned
And error message explains the issue
```

**Checklist**:
- [ ] Request parsing and validation
- [ ] Llama prompt format construction
- [ ] OpenAI-compatible response format
- [ ] Error handling with appropriate status codes
- [ ] Request timeout handling
- [ ] Integration test with curl

---

#### TASK-027: Streaming Response Support

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-026 |
| **Milestone** | M7: HTTP API |

**Description**:
Implement Server-Sent Events (SSE) streaming for chat completions.

**Files to Create/Modify**:
- `api/handlers.go` (modify)
- `api/streaming.go` (create)

**Implementation Details**:
```go
func (s *Server) handleChatCompletionsStream(w http.ResponseWriter, r *http.Request, req ChatCompletionRequest) {
    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "Streaming not supported", http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    // Create streaming channel
    tokenChan := make(chan string, 10)
    errChan := make(chan error, 1)

    go func() {
        err := s.engine.GenerateStream(r.Context(), req, tokenChan)
        errChan <- err
        close(tokenChan)
    }()

    for token := range tokenChan {
        chunk := ChatCompletionChunk{
            ID:      generateID(),
            Object:  "chat.completion.chunk",
            Created: time.Now().Unix(),
            Choices: []ChunkChoice{{
                Index: 0,
                Delta: Delta{Content: token},
            }},
        }

        data, _ := json.Marshal(chunk)
        fmt.Fprintf(w, "data: %s\n\n", data)
        flusher.Flush()
    }

    // Send [DONE] marker
    fmt.Fprintf(w, "data: [DONE]\n\n")
    flusher.Flush()
}
```

**Acceptance Criteria**:

*Scenario 1: Stream tokens*
```gherkin
Given POST /v1/chat/completions with stream=true
When generation produces tokens
Then each token is sent as SSE event
And events are flushed immediately
And [DONE] marker sent at end
```

**Checklist**:
- [ ] SSE headers set correctly
- [ ] Token channel from engine
- [ ] Chunk format matches OpenAI
- [ ] [DONE] marker sent at end
- [ ] Client disconnect handled
- [ ] Integration test with streaming client

---

### Milestone 8: Integration & Testing

---

#### TASK-028: Worker Node Binary

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-009, TASK-012, TASK-022 |
| **Milestone** | M8: Integration & Testing |

**Description**:
Create the worker node binary that runs on remote machines.

**Files to Create**:
- `cmd/worker/main.go` (create)

**Implementation Details**:
```go
func main() {
    port := flag.Int("port", 9000, "libp2p listen port")
    gpuID := flag.Int("gpu", 0, "GPU device ID")
    modelPath := flag.String("model", "", "Path to model weights")
    flag.Parse()

    // Initialize GPU
    if err := bindings.InitGPU(*gpuID); err != nil {
        log.Fatal(err)
    }
    defer bindings.ShutdownGPU()

    // Create libp2p host
    ctx := context.Background()
    host, err := p2p.NewHost(ctx, *port)
    if err != nil {
        log.Fatal(err)
    }

    // Setup tensor protocol handler
    protocol := p2p.NewProtocol(host)
    protocol.OnLayerRequest(func(req *p2p.LayerRequest) (*p2p.LayerResponse, error) {
        // Execute layer forward pass
        output, err := executeLayer(req.LayerID, req.Activation, req.Position)
        if err != nil {
            return nil, err
        }
        return &p2p.LayerResponse{Activation: output}, nil
    })

    // Setup discovery
    discovery := p2p.NewDiscovery(host)
    discovery.SetupMDNS()
    discovery.SetupDHT(ctx)

    log.Printf("Worker started: %s", host.ID())
    log.Printf("GPU: %d, Port: %d", *gpuID, *port)

    // Wait for shutdown
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    <-sigChan
}
```

**Acceptance Criteria**:

*Scenario 1: Start worker*
```gherkin
Given worker binary built
When ./worker --port 9001 --gpu 1 --model /models/llama-7b
Then worker initializes GPU 1
And listens on port 9001
And advertises via mDNS/DHT
```

**Checklist**:
- [ ] CLI flags for port, GPU, model path
- [ ] GPU initialization
- [ ] libp2p host setup
- [ ] Protocol handler registration
- [ ] Discovery setup
- [ ] Graceful shutdown
- [ ] Systemd service file (optional)

---

#### TASK-029: Coordinator Binary

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-025, TASK-026, TASK-028 |
| **Milestone** | M8: Integration & Testing |

**Description**:
Create the coordinator binary that exposes the HTTP API and orchestrates inference.

**Files to Create**:
- `cmd/neurogrid/main.go` (create)

**Implementation Details**:
```go
func main() {
    httpPort := flag.Int("http-port", 8080, "HTTP API port")
    p2pPort := flag.Int("p2p-port", 9000, "libp2p port")
    gpuID := flag.Int("gpu", 0, "Local GPU device ID")
    modelPath := flag.String("model", "", "Path to model")
    flag.Parse()

    ctx := context.Background()

    // Initialize local GPU
    if err := bindings.InitGPU(*gpuID); err != nil {
        log.Fatal(err)
    }

    // Setup libp2p for peer communication
    host, _ := p2p.NewHost(ctx, *p2pPort)

    // Discover peers
    discovery := p2p.NewDiscovery(host)
    discovery.SetupMDNS()

    // Wait for minimum peers
    peers := waitForPeers(discovery, 4)

    // Setup scheduler
    scheduler := scheduler.NewScheduler(types.Llama7BConfig())
    for _, peer := range peers {
        scheduler.RegisterPeer(peer)
    }

    // Load model distributed
    model := model.NewDistributedModel(*modelPath, scheduler)
    model.LoadToCluster(ctx)

    // Create inference engine
    engine := inference.NewEngine(types.Llama7BConfig(), scheduler, router)

    // Start HTTP server
    server := api.NewServer(engine, fmt.Sprintf(":%d", *httpPort))
    log.Printf("API server starting on :%d", *httpPort)
    log.Fatal(server.Start())
}
```

**Acceptance Criteria**:

*Scenario 1: Start coordinator*
```gherkin
Given 4 worker nodes running
When ./neurogrid --http-port 8080 --model /models/llama-7b
Then coordinator discovers workers
And model is distributed across cluster
And HTTP API becomes available
```

**Checklist**:
- [ ] CLI flags for all configuration
- [ ] Peer discovery and waiting
- [ ] Scheduler setup
- [ ] Model loading
- [ ] Engine initialization
- [ ] HTTP server startup
- [ ] Health check endpoint

---

#### TASK-030: End-to-End Integration Test

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Story Points** | 5 |
| **Estimated Duration** | 2 days |
| **Dependencies** | TASK-028, TASK-029 |
| **Milestone** | M8: Integration & Testing |

**Description**:
Create comprehensive end-to-end test that validates the full pipeline.

**Files to Create**:
- `scripts/test_e2e.sh` (create)
- `tests/integration/e2e_test.go` (create)

**Implementation Details**:
```bash
#!/bin/bash
# scripts/test_e2e.sh

set -e

# Start 4 workers on different ports
for i in {1..4}; do
    ./build/worker --port $((9000 + i)) --gpu 0 &
    WORKER_PIDS+=($!)
done

sleep 5  # Wait for workers to start

# Start coordinator
./build/neurogrid --http-port 8080 --p2p-port 9000 --gpu 0 &
COORD_PID=$!

sleep 10  # Wait for model loading

# Test API
RESPONSE=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-7b",
        "messages": [{"role": "user", "content": "Qual e a capital da Franca?"}]
    }')

# Verify response contains "Paris"
if echo "$RESPONSE" | grep -q "Paris"; then
    echo "TEST PASSED: Response contains 'Paris'"
else
    echo "TEST FAILED: Response does not contain 'Paris'"
    echo "$RESPONSE"
    exit 1
fi

# Cleanup
kill $COORD_PID ${WORKER_PIDS[@]}
```

**Acceptance Criteria**:

*Scenario 1: Full E2E test*
```gherkin
Given workers and coordinator are running
When API request asks "Qual e a capital da Franca?"
Then response contains "Paris"
And latency is under 5 seconds
And all nodes participated in inference
```

**Checklist**:
- [ ] Shell script starts all components
- [ ] Workers discovered by coordinator
- [ ] Model loaded successfully
- [ ] API request succeeds
- [ ] Response validation
- [ ] Clean shutdown
- [ ] CI integration

---

#### TASK-031: Performance Benchmarks

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Story Points** | 3 |
| **Estimated Duration** | 1 day |
| **Dependencies** | TASK-030 |
| **Milestone** | M8: Integration & Testing |

**Description**:
Create benchmarks to measure and validate performance targets.

**Files to Create**:
- `tests/benchmark/throughput_test.go` (create)
- `scripts/benchmark.sh` (create)

**Implementation Details**:
```go
func BenchmarkTokenGeneration(b *testing.B) {
    // Setup engine
    engine := setupTestEngine()

    req := &inference.GenerateRequest{
        Prompt:      "Hello, how are you?",
        MaxTokens:   100,
        Temperature: 0.7,
    }

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := engine.Generate(context.Background(), req)
        if err != nil {
            b.Fatal(err)
        }
    }

    // Report tokens per second
    tokensGenerated := b.N * 100
    b.ReportMetric(float64(tokensGenerated)/b.Elapsed().Seconds(), "tokens/sec")
}

func BenchmarkFirstTokenLatency(b *testing.B) {
    engine := setupTestEngine()

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        start := time.Now()
        tokenChan := make(chan string, 1)
        go engine.GenerateStream(context.Background(), req, tokenChan)
        <-tokenChan  // First token
        b.ReportMetric(float64(time.Since(start).Milliseconds()), "ms/first_token")
    }
}
```

**Acceptance Criteria**:

*Scenario 1: Meet performance targets*
```gherkin
Given cluster with 5 GPUs
When running benchmark suite
Then first token latency < 2s
And throughput > 5 tokens/sec
And GPU utilization > 50%
```

**Checklist**:
- [ ] Throughput benchmark
- [ ] First token latency benchmark
- [ ] GPU utilization monitoring
- [ ] Results compared to targets
- [ ] Benchmark results logged

---

#### TASK-032: Makefile Updates

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Story Points** | 2 |
| **Estimated Duration** | 0.5 days |
| **Dependencies** | TASK-028, TASK-029 |
| **Milestone** | M8: Integration & Testing |

**Description**:
Update Makefile with new build targets for distributed components.

**Files to Modify**:
- `Makefile` (modify)

**Implementation Details**:
```makefile
# Add to existing Makefile

.PHONY: deps flatbuffers server worker test-e2e bench-e2e

# Dependencies
deps:
	go mod download
	go get github.com/libp2p/go-libp2p@v0.46.0
	go get github.com/libp2p/go-libp2p-kad-dht@v0.28.0
	go get github.com/google/flatbuffers/go

# FlatBuffers generation
flatbuffers:
	flatc --go -o pkg/types/generated schemas/tensor.fbs

# Build coordinator
server: cuda flatbuffers
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	go build -tags cuda -o $(BUILD_DIR)/neurogrid ./cmd/neurogrid

# Build worker
worker: cuda flatbuffers
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(CUDA_INCLUDE)" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -L$(CUDA_LIB) -lgpu_engine -lcudart -lcublas" \
	go build -tags cuda -o $(BUILD_DIR)/worker ./cmd/worker

# E2E test
test-e2e: server worker
	./scripts/test_e2e.sh

# E2E benchmarks
bench-e2e: server worker
	./scripts/benchmark.sh
```

**Acceptance Criteria**:

*Scenario 1: Build all binaries*
```gherkin
Given source code is complete
When make server worker is run
Then both binaries are built
And placed in build/ directory
```

**Checklist**:
- [ ] `deps` target for Go dependencies
- [ ] `flatbuffers` target for code generation
- [ ] `server` target for coordinator
- [ ] `worker` target for worker node
- [ ] `test-e2e` target for integration test
- [ ] `bench-e2e` target for benchmarks

---

## Critical Path Analysis

### Critical Path Tasks
The following tasks are on the critical path and any delay will impact the overall schedule:

1. **TASK-001** -> **TASK-003** -> **TASK-005** (Multi-GPU foundation)
2. **TASK-006** -> **TASK-008** (Transport abstraction)
3. **TASK-009** -> **TASK-012** -> **TASK-013** (libp2p networking)
4. **TASK-017** (Layer assignment)
5. **TASK-019** (Inference engine core)
6. **TASK-022** -> **TASK-023** (Model loading)
7. **TASK-026** (API handler)
8. **TASK-030** (E2E integration)

### Parallelization Opportunities

| Phase | Parallel Work Streams |
|-------|----------------------|
| M1 + M2 | CUDA multi-GPU + Transport interface (different developers) |
| M3 | mDNS discovery + DHT discovery (can run in parallel) |
| M4 + M5 | Scheduler algorithm + Token sampler (independent) |
| M6 | Weight loader + Tokenizer (no dependencies between them) |
| M7 | HTTP server + Handlers (same developer, sequential) |

### Potential Bottlenecks

1. **CGO Integration**: Multi-GPU bindings require both CUDA and Go expertise
2. **libp2p Complexity**: DHT and hole punching may require debugging
3. **Model Loading**: Large model files require efficient I/O handling
4. **End-to-End Testing**: Requires all components working together

---

## Implementation Recommendations

### Team Structure
- **CUDA Engineer**: TASK-001 through TASK-005
- **Go Backend Engineer**: TASK-006 through TASK-008, TASK-015 through TASK-021
- **Networking Engineer**: TASK-009 through TASK-014
- **ML Engineer**: TASK-022, TASK-023, TASK-024
- **Full Stack**: TASK-025 through TASK-027
- **DevOps**: TASK-028 through TASK-032

### Sprint Planning (2-week sprints)

**Sprint 1**: Milestones M1 + M2
- TASK-001 through TASK-008
- Deliverable: Multi-GPU working with local transport

**Sprint 2**: Milestone M3
- TASK-009 through TASK-014
- Deliverable: libp2p networking functional

**Sprint 3**: Milestones M4 + M5
- TASK-015 through TASK-021
- Deliverable: Distributed inference working

**Sprint 4**: Milestones M6 + M7
- TASK-022 through TASK-027
- Deliverable: Full API with model loading

**Sprint 5**: Milestone M8
- TASK-028 through TASK-032
- Deliverable: Production-ready system

### Risk Mitigations

| Risk | Mitigation |
|------|------------|
| P2P across NAT fails | Use relay servers as fallback |
| Insufficient VRAM | Implement CPU offload early |
| Network latency too high | Add prefetching, overlap compute/transfer |
| Model loading slow | Implement parallel loading, mmap |

---

## Story Points Summary

| Milestone | Tasks | Total Story Points |
|-----------|-------|-------------------|
| M1: Multi-GPU Infrastructure | 5 | 21 |
| M2: Transport Layer | 3 | 13 |
| M3: libp2p Integration | 6 | 26 |
| M4: Pipeline Scheduler | 4 | 16 |
| M5: Inference Engine | 3 | 16 |
| M6: Model Loading | 3 | 15 |
| M7: HTTP API | 3 | 13 |
| M8: Integration & Testing | 5 | 20 |
| **Total** | **32** | **140** |

### Velocity Estimate
- Assuming team velocity of 30-40 story points per 2-week sprint
- Estimated completion: 4-5 sprints (8-10 weeks)

---

## Appendix: Task Dependency Graph

```
TASK-001 ─┬─ TASK-002 ─┬─ TASK-003 ──┐
          │            │             │
          └─ TASK-004 ─┴─ TASK-005 ──┤
                                     │
TASK-006 ────────────────────────────┼─ TASK-007 ─┬─ TASK-008
                                     │            │
                                     │            └─ TASK-019 ───┬─ TASK-020
                                     │                          │
TASK-009 ─┬─ TASK-010               │                          └─ TASK-021
          │                          │
          ├─ TASK-011               │
          │                          │
          └─ TASK-012 ─── TASK-013 ──┘
                    │
TASK-014 ───────────┘

TASK-015 ─┬─ TASK-016 ─── TASK-017 ─── TASK-018
          │                    │
          │                    └─────────────────────── TASK-024
          │
TASK-022 ────────────────────────────────────────────── TASK-024
          │
TASK-023 ─┘

TASK-025 ─── TASK-026 ─── TASK-027
                │
TASK-019 ──────┘

TASK-028 ─┬─ TASK-029 ─── TASK-030 ─── TASK-031
          │       │
TASK-032 ─┘       └────────────────────────────
```

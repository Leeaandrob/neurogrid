# Transport Layer - Architecture Document

## Overview

The Transport Layer provides abstractions for transferring activation tensors between compute peers. It supports both local multi-GPU communication (via CUDA) and remote peer-to-peer networking (via libp2p).

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Inference Engine                              │
├─────────────────────────────────────────────────────────────────────┤
│                     TransportRouter                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │ RouteActivation│  │ AssignLayer-   │  │ Get/Register   │        │
│  │                │  │ ToPeer         │  │ Transport      │        │
│  └───────┬────────┘  └────────────────┘  └────────────────┘        │
│          │                                                          │
├──────────┼──────────────────────────────────────────────────────────┤
│          │            Transport Interface                           │
│          │                                                          │
│  ┌───────▼────────────────────────────────────────────────────┐    │
│  │                    Transport Interface                      │    │
│  │  SendActivation() | RecvActivation() | PeerInfo() | Close() │    │
│  └───────┬────────────────────────┬────────────────────────────┘    │
│          │                        │                                  │
│  ┌───────▼──────────┐    ┌───────▼──────────┐                       │
│  │   CUDATransport   │    │   P2PTransport   │                       │
│  │   (Local GPU)     │    │   (Remote Node)  │                       │
│  └───────┬──────────┘    └───────┬──────────┘                       │
│          │                        │                                  │
├──────────┼────────────────────────┼─────────────────────────────────┤
│          │                        │                                  │
│  ┌───────▼──────────┐    ┌───────▼──────────┐                       │
│  │  CUDA Bindings   │    │   libp2p Host    │                       │
│  │  CrossDeviceCopy │    │   Stream API     │                       │
│  └──────────────────┘    └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Structures

### PeerDescriptor

Describes a compute peer:

```go
type PeerDescriptor struct {
    ID          string // Unique peer identifier
    Address     string // Network address (for remote peers)
    IsLocal     bool   // True if peer is on local machine
    DeviceID    int    // GPU device ID (for local peers)
    TotalMemory uint64 // Total VRAM in bytes
    FreeMemory  uint64 // Available VRAM in bytes
}
```

### ActivationMessage

Represents data being transferred:

```go
type ActivationMessage struct {
    LayerID   int       // Source or destination layer
    SeqID     uint64    // Sequence ID for ordering
    Data      []byte    // Activation tensor data
    Timestamp time.Time // When the message was created
}
```

## API Reference

### Transport Interface

| Method | Description |
|--------|-------------|
| `SendActivation(ctx, layerID, seqID, data)` | Send activation to peer |
| `RecvActivation(ctx)` | Receive activation from peer |
| `PeerInfo()` | Get peer descriptor |
| `Close()` | Release resources |

### TransportRouter

| Method | Description |
|--------|-------------|
| `NewTransportRouter()` | Create new router |
| `RegisterLocalTransport(deviceID, transport)` | Register local GPU transport |
| `RegisterRemoteTransport(peerID, transport)` | Register remote peer transport |
| `AssignLayerToPeer(layerID, peerID)` | Map layer to peer |
| `RouteActivation(ctx, layerID, seqID, data)` | Route to appropriate transport |
| `Close()` | Close all transports |

### CUDATransport

| Method | Description |
|--------|-------------|
| `NewCUDATransport(srcDevice, dstDevice)` | Create CUDA transport |
| `SendActivation(ctx, layerID, seqID, data)` | Send via CUDA cross-device copy |
| `RecvActivation(ctx)` | Receive from internal queue |

## Routing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RouteActivation Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ RouteActivation │                                            │
│  │   (layerID)     │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                         │
│  │ Lookup layerToPeer │                                         │
│  │   map[layerID]     │                                         │
│  └────────┬───────────┘                                         │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐    Not found    ┌───────────────────┐  │
│  │ Peer assigned?     ├────────────────►│ Return            │  │
│  └────────┬───────────┘                 │ ErrLayerNotAssigned│  │
│           │ Yes                          └───────────────────┘  │
│           ▼                                                      │
│  ┌────────────────────┐    Found        ┌───────────────────┐  │
│  │ Check remote-      ├────────────────►│ transport.Send-   │  │
│  │ Transports[peerID] │                 │ Activation()      │  │
│  └────────┬───────────┘                 └───────────────────┘  │
│           │ Not found                                           │
│           ▼                                                      │
│  ┌────────────────────┐    Found        ┌───────────────────┐  │
│  │ Check local-       ├────────────────►│ transport.Send-   │  │
│  │ Transports         │                 │ Activation()      │  │
│  │ (by PeerInfo.ID)   │                 └───────────────────┘  │
│  └────────┬───────────┘                                         │
│           │ Not found                                           │
│           ▼                                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Return "transport not found for peer"                      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### CUDATransport

Uses internal channel for async send/recv:
1. `SendActivation` copies data to internal buffer and queues
2. `RecvActivation` blocks on channel until data available
3. Actual CUDA cross-device copy happens in background (future: integrate with bindings)

### Thread Safety

- `TransportRouter` uses `sync.RWMutex` for concurrent access
- `CUDATransport` uses `sync.Mutex` for state protection
- All map operations are protected by locks

## Error Handling

| Error | Condition |
|-------|-----------|
| `ErrPeerNotFound` | Peer ID not registered |
| `ErrLayerNotAssigned` | Layer has no peer assignment |
| `ErrTransportNotFound` | Transport not found for device/peer |
| `context.Canceled` | Context canceled during operation |
| `context.DeadlineExceeded` | Context timeout during operation |

## Files

| File | Description |
|------|-------------|
| `pkg/transport/transport.go` | Interface and type definitions |
| `pkg/transport/router.go` | TransportRouter implementation |
| `pkg/transport/local.go` | CUDATransport implementation |
| `tests/e2e/transport_test.go` | E2E test suite |

## Future: P2PTransport

The `P2PTransport` (TASK-013) will implement the same interface using libp2p:
- Uses libp2p streams for network transfer
- Implements same SendActivation/RecvActivation pattern
- Enables distributed inference across machines

## Related Tasks

- TASK-006: Transport Interface Definition (this document)
- TASK-007: Transport Router Implementation (this document)
- TASK-008: CUDA Transport Implementation (this document)
- TASK-013: P2P Transport Implementation (future)

---

*Generated by TDD E2E Workflow - TASK-006/007/008 Transport Layer*

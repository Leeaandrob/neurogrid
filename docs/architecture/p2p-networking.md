# P2P Networking Layer - Architecture Document

## Overview

The P2P Networking Layer enables distributed inference across multiple machines using libp2p. It provides peer discovery (local via mDNS, remote via DHT) and a custom tensor transfer protocol for activation exchange.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NeuroGrid P2P Layer                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                        libp2p Host                            │  │
│  │  ┌────────────┐  ┌─────────────┐  ┌────────────────────────┐ │  │
│  │  │ TCP/QUIC   │  │ Hole Punch  │  │ NAT Port Mapping       │ │  │
│  │  │ Transports │  │ + Relay     │  │                        │ │  │
│  │  └────────────┘  └─────────────┘  └────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                               │                                      │
│               ┌───────────────┼───────────────┐                     │
│               ▼               ▼               ▼                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐   │
│  │    Discovery    │ │    Discovery    │ │     Protocol        │   │
│  │      mDNS       │ │      DHT        │ │   Tensor/1.0.0      │   │
│  │  (Local LAN)    │ │  (Kademlia)     │ │   (Activations)     │   │
│  └────────┬────────┘ └────────┬────────┘ └─────────┬───────────┘   │
│           │                   │                     │               │
│           └───────────────────┼─────────────────────┘               │
│                               ▼                                      │
│                    ┌─────────────────────┐                          │
│                    │    Peer Channel     │                          │
│                    │  (Discovered Peers) │                          │
│                    └─────────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
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

### Message Format

```
┌─────────────────────────────────────────────────────────────┐
│                     Tensor Message                           │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────┬───────────┬───────────┬─────────────────┐   │
│  │ LayerID   │  SeqID    │ DataLen   │    Data         │   │
│  │ (4 bytes) │ (8 bytes) │ (4 bytes) │  (variable)     │   │
│  └───────────┴───────────┴───────────┴─────────────────┘   │
│                                                              │
│  Header: 16 bytes total (fixed)                             │
│  Data: Variable length (up to DataLen bytes)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Field Descriptions

| Field | Size | Description |
|-------|------|-------------|
| LayerID | 4 bytes | Source/destination layer in the model |
| SeqID | 8 bytes | Sequence number for ordering |
| DataLen | 4 bytes | Length of tensor data in bytes |
| Data | Variable | Raw tensor bytes (f16/f32/bf16) |

### API Reference

```go
// Create protocol handler
protocol := p2p.NewProtocol(host)

// Register receive callback
protocol.OnTensorReceived(func(msg *p2p.TensorMessage) {
    // Process received tensor
    fmt.Printf("Received layer %d from %s\n", msg.LayerID, msg.From)
})

// Send tensor to peer
err := protocol.SendTensor(ctx, peerID, layerID, seqID, data)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Tensor Transfer Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Node A (Sender)                 Node B (Receiver)          │
│  ─────────────────              ──────────────────          │
│                                                              │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │ SendTensor()    │           │ OnTensorReceived │          │
│  │ - layerID: 5    │           │ callback         │          │
│  │ - seqID: 100    │           └────────▲────────┘          │
│  │ - data: [...]   │                    │                    │
│  └────────┬────────┘                    │                    │
│           │                             │                    │
│           ▼                             │                    │
│  ┌─────────────────┐                    │                    │
│  │ NewStream()     │                    │                    │
│  │ to peerID       │                    │                    │
│  └────────┬────────┘                    │                    │
│           │                             │                    │
│           ▼                             │                    │
│  ┌─────────────────┐           ┌────────┴────────┐          │
│  │ Write Header    │──────────►│ Read Header     │          │
│  │ (16 bytes)      │  network  │ decodeHeader()  │          │
│  └────────┬────────┘           └────────┬────────┘          │
│           │                             │                    │
│           ▼                             ▼                    │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │ Write Data      │──────────►│ Read Data       │          │
│  │ (dataLen bytes) │  network  │ io.ReadFull()   │          │
│  └─────────────────┘           └────────┬────────┘          │
│                                         │                    │
│                                         ▼                    │
│                                ┌─────────────────┐          │
│                                │ TensorMessage   │          │
│                                │ - LayerID: 5    │          │
│                                │ - SeqID: 100    │          │
│                                │ - Data: [...]   │          │
│                                │ - From: nodeA   │          │
│                                └─────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Thread Safety

| Component | Synchronization | Notes |
|-----------|-----------------|-------|
| Discovery | sync.Mutex | Protects mDNS and DHT instances |
| Protocol | sync.RWMutex | Protects handler callback |
| PeerChan | Buffered channel | Non-blocking sends (100 capacity) |

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| Failed to open stream | Peer unreachable | Retry with backoff, check connectivity |
| Failed to write header | Stream closed | Retry connection |
| Failed to write data | Network error | Retry or failover to another peer |

## Files

| File | Description |
|------|-------------|
| `p2p/host.go` | libp2p host creation and utilities |
| `p2p/discovery.go` | mDNS and DHT discovery |
| `p2p/protocol.go` | Tensor transfer protocol |
| `tests/e2e/p2p_test.go` | E2E test suite |

## Integration with Transport Layer

The P2P networking layer will integrate with the Transport layer (TASK-013) through `P2PTransport`:

```go
// Future: P2PTransport implements Transport interface
type P2PTransport struct {
    protocol *p2p.Protocol
    peerID   peer.ID
    // ...
}

func (t *P2PTransport) SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error {
    return t.protocol.SendTensor(ctx, t.peerID, layerID, seqID, data)
}
```

## Related Tasks

- TASK-009: libp2p Host Setup (this document)
- TASK-010: mDNS Local Discovery (this document)
- TASK-011: DHT Remote Discovery (this document)
- TASK-012: Tensor Protocol (this document)
- TASK-013: P2P Transport Implementation (future)

---

*Generated by TDD E2E Workflow - TASK-009/010/011/012 P2P Networking*

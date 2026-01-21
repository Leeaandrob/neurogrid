# PRP: Multi-Cluster Federation (DHT)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Multi-Cluster Federation via DHT |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Long-Term (Scale) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 7/10 |
| **Dependencies** | All Short/Medium-Term PRPs |

---

## Discovery Summary

### Initial Task Analysis

Extend the P2P network to support federation of multiple clusters across the internet using DHT (Distributed Hash Table) for peer discovery and routing, enabling a global network of GPU nodes.

### User Clarifications Received

- **Question**: What defines a "cluster"?
- **Answer**: Nodes under same coordinator with low-latency interconnect
- **Impact**: Need cluster-aware routing and topology management

### Missing Requirements Identified

- Cross-cluster authentication
- Geographic routing optimization
- Bandwidth/latency-aware placement
- NAT traversal for home nodes

---

## Goal

Implement DHT-based cluster federation allowing independent NeuroGrid clusters to form a global inference network, sharing capacity and routing requests to optimal nodes.

## Why

- **Scale**: Support millions of consumer GPUs worldwide
- **Resilience**: No single point of failure
- **Accessibility**: Allow anyone to contribute GPU capacity
- **Cost**: Leverage spare consumer GPU cycles

## What

### Success Criteria

- [ ] Clusters discover each other via DHT
- [ ] Cross-cluster inference routing works
- [ ] Latency-aware cluster selection (< 50ms preference)
- [ ] NAT traversal for home nodes (STUN/TURN)
- [ ] Authentication between clusters
- [ ] Graceful handling of cluster disconnects
- [ ] Load balancing across clusters

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: `p2p/host.go` with libp2p DHT
- **External research needed**: Yes - Kademlia DHT, libp2p relay
- **Knowledge gaps identified**: NAT traversal complexity, global DHT bootstrap

### Documentation & References

```yaml
- url: https://docs.libp2p.io/concepts/nat/
  why: NAT traversal with libp2p

- url: https://github.com/libp2p/go-libp2p-kad-dht
  why: Kademlia DHT implementation

- url: https://docs.libp2p.io/concepts/circuit-relay/
  why: Relay for unreachable peers

- file: p2p/host.go
  why: Existing libp2p host setup

- file: p2p/discovery.go
  why: Current mDNS + DHT discovery
```

### Current Codebase tree

```
p2p/
├── host.go           # libp2p host with mDNS + DHT
├── protocol.go       # /neurogrid/tensor/1.0.0 protocol
├── discovery.go      # Peer discovery
└── peer.go           # Peer management
```

### Desired Codebase tree

```
p2p/
├── host.go           # MODIFY: Add relay, hole-punching
├── protocol.go       # Existing
├── discovery.go      # MODIFY: Enhanced DHT discovery
├── peer.go           # Existing
├── federation/       # NEW: Federation layer
│   ├── cluster.go    # Cluster definition and management
│   ├── router.go     # Cross-cluster routing
│   ├── topology.go   # Global topology awareness
│   └── auth.go       # Cluster authentication
├── nat/              # NEW: NAT traversal
│   ├── stun.go       # STUN client
│   ├── turn.go       # TURN relay
│   └── holepunch.go  # Hole punching coordination
└── bootstrap/        # NEW: Bootstrap nodes
    └── bootstrap.go  # Well-known bootstrap peers
```

### Known Gotchas

```go
// CRITICAL: DHT convergence takes time (~30s for new peers)
// CRITICAL: NAT traversal has ~70% success rate
// CRITICAL: Relay fallback needed for symmetric NAT
// CRITICAL: Cross-region latency can be >200ms
// CRITICAL: Bootstrap nodes are single points of failure
```

---

## Implementation Blueprint

### Data Models

```go
// p2p/federation/cluster.go

type Cluster struct {
    ID           string          `json:"id"`
    Name         string          `json:"name"`
    Coordinator  peer.ID         `json:"coordinator"`
    Members      []peer.ID       `json:"members"`
    Region       string          `json:"region"`
    Capacity     ClusterCapacity `json:"capacity"`
    PublicKey    []byte          `json:"public_key"`
    LastSeen     time.Time       `json:"last_seen"`
}

type ClusterCapacity struct {
    TotalGPUs       int     `json:"total_gpus"`
    AvailableGPUs   int     `json:"available_gpus"`
    TotalVRAM       uint64  `json:"total_vram"`
    AvailableVRAM   uint64  `json:"available_vram"`
    MaxBatchSize    int     `json:"max_batch_size"`
    SupportedModels []string `json:"supported_models"`
}

// p2p/federation/router.go

type FederationRouter struct {
    localCluster  *Cluster
    knownClusters map[string]*Cluster
    dht           *dht.IpfsDHT
    topology      *TopologyManager
    mu            sync.RWMutex
}

type RouteDecision struct {
    TargetCluster  string
    TargetPeer     peer.ID
    EstimatedRTT   time.Duration
    Reason         string
}

// p2p/federation/topology.go

type TopologyManager struct {
    clusters      map[string]*ClusterInfo
    latencyMatrix map[string]map[string]time.Duration
    updateChan    chan TopologyUpdate
}

type ClusterInfo struct {
    Cluster     *Cluster
    AvgLatency  time.Duration
    Reliability float64  // 0-1, based on recent success rate
    LastProbe   time.Time
}

// p2p/federation/auth.go

type ClusterAuth struct {
    privateKey ed25519.PrivateKey
    publicKey  ed25519.PublicKey
    trustedKeys map[string]ed25519.PublicKey  // clusterID -> pubkey
}

type AuthChallenge struct {
    ClusterID string
    Nonce     []byte
    Timestamp int64
}

type AuthResponse struct {
    ClusterID string
    Signature []byte
    Capacity  ClusterCapacity
}
```

### Task List

```yaml
Task 1: Add bootstrap node support
  CREATE p2p/bootstrap/bootstrap.go:
    - Hardcoded bootstrap peer addresses
    - Bootstrap connection on startup
    - Bootstrap node discovery via DNS

Task 2: Enhance DHT discovery for federation
  MODIFY p2p/discovery.go:
    - Advertise cluster info in DHT
    - Discover remote clusters
    - Periodic re-announcement

Task 3: Create cluster management
  CREATE p2p/federation/cluster.go:
    - Cluster struct and methods
    - Local cluster initialization
    - Cluster capacity tracking

Task 4: Create federation router
  CREATE p2p/federation/router.go:
    - FederationRouter for cross-cluster routing
    - Latency-aware cluster selection
    - Fallback to local cluster

Task 5: Create topology manager
  CREATE p2p/federation/topology.go:
    - TopologyManager for global view
    - Periodic latency probing
    - Reliability tracking

Task 6: Create cluster authentication
  CREATE p2p/federation/auth.go:
    - Ed25519 key pair generation
    - Challenge-response auth
    - Trusted cluster registry

Task 7: Add NAT traversal
  MODIFY p2p/host.go:
    - Enable AutoNAT
    - Enable hole punching
    - Configure circuit relay

Task 8: Create STUN/TURN support
  CREATE p2p/nat/stun.go:
    - STUN client for NAT detection
  CREATE p2p/nat/turn.go:
    - TURN relay fallback

Task 9: Add cross-cluster inference
  MODIFY pkg/inference/engine.go:
    - Route to remote cluster if needed
    - Handle cross-cluster failures
    - Timeout and retry logic

Task 10: Add federation metrics
  MODIFY pkg/metrics/cluster.go:
    - Cross-cluster request counts
    - Latency per cluster
    - Federation health

Task 11: Add tests
  CREATE tests/federation/federation_test.go:
    - Test cluster discovery
    - Test cross-cluster routing
    - Test NAT traversal
    - Test authentication
```

### Per-Task Pseudocode

```go
// Task 1: Bootstrap nodes
var BootstrapPeers = []string{
    "/dnsaddr/bootstrap1.neurogrid.io",
    "/dnsaddr/bootstrap2.neurogrid.io",
    "/ip4/34.123.45.67/tcp/9000/p2p/QmBootstrap1...",
}

func (h *Host) ConnectBootstrap(ctx context.Context) error {
    for _, addr := range BootstrapPeers {
        ma, _ := multiaddr.NewMultiaddr(addr)
        pi, _ := peer.AddrInfoFromP2pAddr(ma)

        if err := h.Connect(ctx, *pi); err != nil {
            log.Printf("Bootstrap %s failed: %v", addr, err)
            continue
        }
        log.Printf("Connected to bootstrap: %s", pi.ID)
    }
    return nil
}

// Task 2: DHT cluster advertisement
const ClusterNamespace = "/neurogrid/clusters"

func (d *Discovery) AdvertiseCluster(ctx context.Context, cluster *Cluster) error {
    // Serialize cluster info
    data, _ := json.Marshal(cluster)

    // Store in DHT under cluster ID
    key := path.Join(ClusterNamespace, cluster.ID)
    return d.dht.PutValue(ctx, key, data)
}

func (d *Discovery) DiscoverClusters(ctx context.Context) ([]*Cluster, error) {
    // Find all cluster records
    results, err := d.dht.SearchValue(ctx, ClusterNamespace)
    if err != nil {
        return nil, err
    }

    var clusters []*Cluster
    for result := range results {
        var cluster Cluster
        if err := json.Unmarshal(result, &cluster); err != nil {
            continue
        }
        clusters = append(clusters, &cluster)
    }
    return clusters, nil
}

// Task 4: Federation router
func (r *FederationRouter) SelectCluster(ctx context.Context, model string, requirements RouteRequirements) (*RouteDecision, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()

    var bestCluster *Cluster
    var bestLatency time.Duration = time.Hour

    for _, cluster := range r.knownClusters {
        // Check if cluster supports model
        if !containsModel(cluster.Capacity.SupportedModels, model) {
            continue
        }

        // Check capacity
        if cluster.Capacity.AvailableGPUs < requirements.MinGPUs {
            continue
        }

        // Check latency
        latency := r.topology.GetLatency(r.localCluster.ID, cluster.ID)
        if latency < bestLatency {
            bestLatency = latency
            bestCluster = cluster
        }
    }

    if bestCluster == nil {
        return nil, ErrNoSuitableCluster
    }

    return &RouteDecision{
        TargetCluster: bestCluster.ID,
        TargetPeer:    bestCluster.Coordinator,
        EstimatedRTT:  bestLatency,
        Reason:        "lowest latency with capacity",
    }, nil
}

// Task 6: Cluster authentication
func (a *ClusterAuth) GenerateChallenge(targetClusterID string) *AuthChallenge {
    nonce := make([]byte, 32)
    rand.Read(nonce)

    return &AuthChallenge{
        ClusterID: a.localClusterID,
        Nonce:     nonce,
        Timestamp: time.Now().Unix(),
    }
}

func (a *ClusterAuth) VerifyResponse(challenge *AuthChallenge, response *AuthResponse) error {
    // Get trusted public key for cluster
    pubKey, ok := a.trustedKeys[response.ClusterID]
    if !ok {
        return ErrUntrustedCluster
    }

    // Verify signature
    message := append(challenge.Nonce, []byte(fmt.Sprintf("%d", challenge.Timestamp))...)
    if !ed25519.Verify(pubKey, message, response.Signature) {
        return ErrInvalidSignature
    }

    return nil
}

// Task 7: NAT traversal
func NewHostWithNAT(ctx context.Context, port int) (*Host, error) {
    h, err := libp2p.New(
        libp2p.ListenAddrStrings(
            fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", port),
            fmt.Sprintf("/ip4/0.0.0.0/udp/%d/quic-v1", port),
        ),
        // Enable NAT detection
        libp2p.EnableNATService(),
        // Enable hole punching
        libp2p.EnableHolePunching(),
        // Enable relay (for fallback)
        libp2p.EnableRelay(),
        libp2p.EnableRelayService(),
        // Auto-relay for unreachable peers
        libp2p.EnableAutoRelayWithStaticRelays(relayPeers),
    )
    if err != nil {
        return nil, err
    }

    return &Host{Host: h}, nil
}
```

### Integration Points

```yaml
HOST:
  - modify: p2p/host.go
  - add: NAT traversal, relay support
  - add: Bootstrap connection

DISCOVERY:
  - modify: p2p/discovery.go
  - add: Cluster advertisement
  - add: Remote cluster discovery

INFERENCE:
  - modify: pkg/inference/engine.go
  - add: Cross-cluster routing
  - add: Remote cluster fallback

CONFIG:
  - add: federation.enabled setting
  - add: cluster.name, cluster.region
  - add: bootstrap peers list
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./p2p/...
go vet ./p2p/...
golangci-lint run ./p2p/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/federation/...

# Expected: All tests pass
```

### Level 3: Multi-Cluster Test

```bash
# Start cluster A (region: us-west)
CLUSTER_NAME=cluster-a CLUSTER_REGION=us-west ./build/neurogrid

# Start cluster B (region: eu-west)
CLUSTER_NAME=cluster-b CLUSTER_REGION=eu-west ./build/neurogrid

# Verify discovery (wait ~30s for DHT)
curl http://localhost:8080/v1/clusters

# Expected: Both clusters visible
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./p2p/federation/...`
- [ ] No linting errors: `golangci-lint run ./p2p/...`
- [ ] Clusters discover each other via DHT
- [ ] Cross-cluster routing works
- [ ] NAT traversal succeeds (>70% of cases)
- [ ] Relay fallback works
- [ ] Authentication prevents unauthorized clusters

---

## Anti-Patterns to Avoid

- ❌ Don't rely on single bootstrap node
- ❌ Don't trust cluster announcements without auth
- ❌ Don't ignore latency in routing decisions
- ❌ Don't assume NAT traversal always works
- ❌ Don't forget relay fallback

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Cluster discovery | < 30s |
| Cross-cluster latency overhead | < 50ms |
| NAT traversal success | > 70% |
| Federation convergence | < 60s |

---

## Security Considerations

1. **Cluster Authentication**: Ed25519 challenge-response
2. **Traffic Encryption**: libp2p TLS by default
3. **Trusted Registry**: Manual trust for production clusters
4. **Rate Limiting**: Per-cluster request limits

---

**PRP Confidence Score: 7/10**

**Rationale**:
- +2: libp2p provides DHT and relay
- +2: Well-documented NAT traversal
- +1: Existing P2P infrastructure
- -1: DHT convergence is slow
- -1: NAT traversal has variable success
- -2: Bootstrap node availability is critical

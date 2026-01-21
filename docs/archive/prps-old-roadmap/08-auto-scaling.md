# PRP: Auto-Scaling (Dynamic Peer Management)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Auto-Scaling with Dynamic Peers |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Long-Term (Scale) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 7/10 |
| **Dependencies** | PRP-05, PRP-06, PRP-07 |

---

## Discovery Summary

### Initial Task Analysis

Implement automatic scaling that dynamically adds/removes worker peers based on demand, allowing the cluster to grow and shrink without manual intervention.

### User Clarifications Received

- **Question**: Scale based on what metric?
- **Answer**: Queue depth, latency percentiles, GPU utilization
- **Impact**: Multi-metric scaling policy needed

### Missing Requirements Identified

- Peer warm-up time (weight loading)
- Graceful draining before removal
- Scale-down cooldown period
- Resource reservation

---

## Goal

Implement auto-scaling that automatically recruits new peers when demand increases and releases them when demand decreases, maintaining target latency and throughput SLOs.

## Why

- **Cost efficiency**: Only use resources when needed
- **Responsiveness**: Handle traffic spikes automatically
- **Simplicity**: No manual capacity planning
- **Reliability**: Maintain SLOs under variable load

## What

### Success Criteria

- [ ] Scale up when queue depth > threshold for 30s
- [ ] Scale up when p99 latency > SLO for 60s
- [ ] Scale down when utilization < 30% for 5 minutes
- [ ] Graceful draining before peer removal
- [ ] Peer warm-up (weight loading) before traffic
- [ ] Cooldown period between scale events
- [ ] Min/max peer limits configurable

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: `pkg/scheduler/scheduler.go` for peer management
- **External research needed**: Yes - Kubernetes HPA patterns
- **Knowledge gaps identified**: Peer warm-up coordination

### Documentation & References

```yaml
- url: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
  why: HPA algorithm and patterns

- file: pkg/scheduler/scheduler.go
  why: Existing peer registration

- file: pkg/metrics/registry.go
  why: Metrics for scaling decisions

- file: p2p/discovery.go
  why: Peer discovery for recruiting
```

### Current Codebase tree

```
pkg/
├── scheduler/
│   └── scheduler.go   # Layer assignment, peer registration
├── metrics/
│   └── registry.go    # Prometheus metrics
└── inference/
    └── engine.go      # Request processing
```

### Desired Codebase tree

```
pkg/
├── scheduler/
│   └── scheduler.go   # MODIFY: Dynamic peer management
├── autoscale/         # NEW: Auto-scaling logic
│   ├── autoscaler.go  # Main autoscaler
│   ├── policy.go      # Scaling policies
│   ├── metrics.go     # Metrics aggregation
│   └── warmup.go      # Peer warm-up coordination
├── metrics/
│   └── registry.go
└── inference/
    └── engine.go
```

### Known Gotchas

```go
// CRITICAL: Weight loading takes 30-120s per peer
// CRITICAL: Scale-down during inference causes errors
// CRITICAL: Too aggressive scaling causes oscillation
// CRITICAL: Consumer GPUs may be unavailable suddenly
// CRITICAL: Network latency affects scaling decisions
```

---

## Implementation Blueprint

### Data Models

```go
// pkg/autoscale/autoscaler.go

type Autoscaler struct {
    scheduler    *scheduler.Scheduler
    discovery    *p2p.Discovery
    policy       *ScalingPolicy
    metrics      *MetricsAggregator
    warmupMgr    *WarmupManager
    state        AutoscalerState
    mu           sync.Mutex
}

type AutoscalerState struct {
    CurrentPeers     int
    TargetPeers      int
    LastScaleUp      time.Time
    LastScaleDown    time.Time
    PendingWarmup    []peer.ID
    DrainingPeers    []peer.ID
}

// pkg/autoscale/policy.go

type ScalingPolicy struct {
    MinPeers           int           `yaml:"min_peers"`
    MaxPeers           int           `yaml:"max_peers"`
    TargetUtilization  float64       `yaml:"target_utilization"`  // 0-1
    TargetLatencyP99   time.Duration `yaml:"target_latency_p99"`
    QueueThreshold     int           `yaml:"queue_threshold"`
    ScaleUpCooldown    time.Duration `yaml:"scale_up_cooldown"`
    ScaleDownCooldown  time.Duration `yaml:"scale_down_cooldown"`
    StabilizationWindow time.Duration `yaml:"stabilization_window"`
    WarmupTimeout      time.Duration `yaml:"warmup_timeout"`
}

type ScaleDecision struct {
    Action      ScaleAction
    Delta       int
    Reason      string
    Confidence  float64
    Timestamp   time.Time
}

type ScaleAction int

const (
    ScaleNone ScaleAction = iota
    ScaleUp
    ScaleDown
)

// pkg/autoscale/metrics.go

type MetricsAggregator struct {
    windowSize time.Duration
    samples    []MetricsSample
    mu         sync.RWMutex
}

type MetricsSample struct {
    Timestamp     time.Time
    QueueDepth    int
    LatencyP50    time.Duration
    LatencyP99    time.Duration
    Utilization   float64
    ActivePeers   int
    TokensPerSec  float64
}

// pkg/autoscale/warmup.go

type WarmupManager struct {
    loader      *model.ModelLoader
    coordinator peer.ID
    warmingUp   map[peer.ID]*WarmupState
    mu          sync.Mutex
}

type WarmupState struct {
    PeerID      peer.ID
    StartedAt   time.Time
    Progress    float64  // 0-1
    LayersLoaded int
    TotalLayers int
    Ready       bool
    Error       error
}
```

### Task List

```yaml
Task 1: Create scaling policy
  CREATE pkg/autoscale/policy.go:
    - ScalingPolicy struct with thresholds
    - Default policy values
    - Policy validation

Task 2: Create metrics aggregator
  CREATE pkg/autoscale/metrics.go:
    - MetricsAggregator with sliding window
    - Sample collection from Prometheus
    - Aggregation functions (avg, p99)

Task 3: Create warmup manager
  CREATE pkg/autoscale/warmup.go:
    - WarmupManager for coordinating weight loading
    - Progress tracking
    - Timeout handling

Task 4: Create main autoscaler
  CREATE pkg/autoscale/autoscaler.go:
    - Autoscaler main loop
    - Scale decision logic
    - Peer recruitment/release

Task 5: Implement scale-up logic
  MODIFY pkg/autoscale/autoscaler.go:
    - Discover available peers
    - Initiate warmup
    - Add to scheduler when ready

Task 6: Implement scale-down logic
  MODIFY pkg/autoscale/autoscaler.go:
    - Select peer for removal (least loaded)
    - Initiate graceful drain
    - Remove from scheduler

Task 7: Add draining support
  MODIFY pkg/scheduler/scheduler.go:
    - DrainPeer method
    - Stop sending new layers
    - Wait for in-flight to complete

Task 8: Add autoscaler metrics
  MODIFY pkg/metrics/registry.go:
    - autoscale_decisions_total
    - autoscale_peer_count
    - autoscale_warmup_duration

Task 9: Add configuration
  CREATE configs/autoscale.yaml:
    - Default scaling policy
    - Override per model

Task 10: Add tests
  CREATE tests/autoscale/autoscaler_test.go:
    - Test scaling decisions
    - Test warmup coordination
    - Test draining
    - Test oscillation prevention
```

### Per-Task Pseudocode

```go
// Task 4: Autoscaler main loop
func (a *Autoscaler) Run(ctx context.Context) {
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            a.evaluate(ctx)
        }
    }
}

func (a *Autoscaler) evaluate(ctx context.Context) {
    a.mu.Lock()
    defer a.mu.Unlock()

    // Collect current metrics
    sample := a.metrics.GetLatestAggregate()

    // Make scaling decision
    decision := a.makeDecision(sample)

    if decision.Action == ScaleNone {
        return
    }

    // Check cooldown
    if !a.canScale(decision.Action) {
        log.Printf("Scaling blocked by cooldown: %s", decision.Reason)
        return
    }

    // Execute scaling
    switch decision.Action {
    case ScaleUp:
        a.scaleUp(ctx, decision.Delta)
    case ScaleDown:
        a.scaleDown(ctx, decision.Delta)
    }

    // Record metrics
    AutoscaleDecisions.WithLabelValues(decision.Action.String()).Inc()
}

func (a *Autoscaler) makeDecision(sample MetricsSample) *ScaleDecision {
    // Check for scale-up conditions
    if sample.QueueDepth > a.policy.QueueThreshold {
        return &ScaleDecision{
            Action:     ScaleUp,
            Delta:      1,
            Reason:     fmt.Sprintf("queue depth %d > threshold %d", sample.QueueDepth, a.policy.QueueThreshold),
            Confidence: 0.9,
        }
    }

    if sample.LatencyP99 > a.policy.TargetLatencyP99 {
        return &ScaleDecision{
            Action:     ScaleUp,
            Delta:      1,
            Reason:     fmt.Sprintf("p99 latency %v > target %v", sample.LatencyP99, a.policy.TargetLatencyP99),
            Confidence: 0.8,
        }
    }

    // Check for scale-down conditions
    if sample.Utilization < 0.3 && sample.ActivePeers > a.policy.MinPeers {
        return &ScaleDecision{
            Action:     ScaleDown,
            Delta:      1,
            Reason:     fmt.Sprintf("utilization %.1f%% < 30%%", sample.Utilization*100),
            Confidence: 0.7,
        }
    }

    return &ScaleDecision{Action: ScaleNone}
}

// Task 5: Scale-up implementation
func (a *Autoscaler) scaleUp(ctx context.Context, count int) error {
    // Find available peers via discovery
    available := a.discovery.GetAvailablePeers(count)
    if len(available) == 0 {
        log.Println("No available peers for scale-up")
        return ErrNoPeersAvailable
    }

    for _, peerID := range available[:min(count, len(available))] {
        // Start warmup (async)
        go a.warmupAndAdd(ctx, peerID)
        a.state.PendingWarmup = append(a.state.PendingWarmup, peerID)
    }

    a.state.LastScaleUp = time.Now()
    return nil
}

func (a *Autoscaler) warmupAndAdd(ctx context.Context, peerID peer.ID) {
    log.Printf("Starting warmup for peer %s", peerID)

    // Coordinate weight loading
    if err := a.warmupMgr.WarmupPeer(ctx, peerID); err != nil {
        log.Printf("Warmup failed for peer %s: %v", peerID, err)
        return
    }

    // Add to scheduler
    a.mu.Lock()
    defer a.mu.Unlock()

    peerInfo := a.discovery.GetPeerInfo(peerID)
    a.scheduler.RegisterPeer(peerInfo)

    // Recompute layer assignments
    if _, err := a.scheduler.ComputeAssignments(); err != nil {
        log.Printf("Failed to compute assignments: %v", err)
    }

    // Remove from pending
    a.state.PendingWarmup = removePeer(a.state.PendingWarmup, peerID)
    a.state.CurrentPeers++

    log.Printf("Peer %s added successfully, total peers: %d", peerID, a.state.CurrentPeers)
}

// Task 6: Scale-down implementation
func (a *Autoscaler) scaleDown(ctx context.Context, count int) error {
    // Select peer to remove (least loaded, not local)
    peer := a.selectPeerForRemoval()
    if peer == "" {
        return ErrNoPeerToRemove
    }

    // Initiate graceful drain
    go a.drainAndRemove(ctx, peer)
    a.state.DrainingPeers = append(a.state.DrainingPeers, peer)
    a.state.LastScaleDown = time.Now()

    return nil
}

func (a *Autoscaler) drainAndRemove(ctx context.Context, peerID peer.ID) {
    log.Printf("Starting drain for peer %s", peerID)

    // Stop sending new work
    a.scheduler.DrainPeer(peerID)

    // Wait for in-flight requests to complete
    timeout := time.After(60 * time.Second)
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-timeout:
            log.Printf("Drain timeout for peer %s, forcing removal", peerID)
            goto remove
        case <-ticker.C:
            if a.scheduler.GetPeerInflight(peerID) == 0 {
                log.Printf("Peer %s drained successfully", peerID)
                goto remove
            }
        }
    }

remove:
    a.mu.Lock()
    defer a.mu.Unlock()

    a.scheduler.UnregisterPeer(peerID)
    a.state.DrainingPeers = removePeer(a.state.DrainingPeers, peerID)
    a.state.CurrentPeers--

    log.Printf("Peer %s removed, total peers: %d", peerID, a.state.CurrentPeers)
}

// Task 7: Scheduler draining support
func (s *Scheduler) DrainPeer(peerID peer.ID) {
    s.mu.Lock()
    defer s.mu.Unlock()

    s.drainingPeers[peerID] = true

    // Reassign layers to other peers
    for _, assignment := range s.assignments {
        if assignment.PeerID == string(peerID) {
            // Find alternative peer
            altPeer := s.findAlternativePeer(assignment.LayerID)
            if altPeer != "" {
                assignment.PeerID = altPeer
            }
        }
    }
}

func (s *Scheduler) GetPeerInflight(peerID peer.ID) int {
    s.mu.RLock()
    defer s.mu.RUnlock()

    return s.inflightCounts[peerID]
}
```

### Integration Points

```yaml
SCHEDULER:
  - modify: pkg/scheduler/scheduler.go
  - add: DrainPeer, UnregisterPeer methods
  - add: Inflight tracking

DISCOVERY:
  - modify: p2p/discovery.go
  - add: GetAvailablePeers method
  - add: Peer availability tracking

COORDINATOR:
  - modify: cmd/neurogrid/main.go
  - add: Start autoscaler goroutine
  - add: autoscale config loading

CONFIG:
  - add: configs/autoscale.yaml
  - add: Autoscale policy settings
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./pkg/autoscale/...
go vet ./pkg/autoscale/...
golangci-lint run ./pkg/autoscale/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/autoscale/...

# Expected: All tests pass
```

### Level 3: Load Test

```bash
# Start with 1 peer
./build/neurogrid --min-peers=1 --max-peers=5

# Generate load
./scripts/load_test.sh --rps=100 --duration=5m

# Verify scaling
# Expected: Peers scale up to handle load, scale down after
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./pkg/autoscale/...`
- [ ] No linting errors: `golangci-lint run ./pkg/autoscale/...`
- [ ] Scale-up triggers on high queue/latency
- [ ] Scale-down triggers on low utilization
- [ ] Warmup completes before receiving traffic
- [ ] Graceful drain works without errors
- [ ] No oscillation under stable load
- [ ] Min/max limits respected

---

## Anti-Patterns to Avoid

- ❌ Don't scale on instantaneous metrics
- ❌ Don't remove peers with in-flight requests
- ❌ Don't add peers without warmup
- ❌ Don't scale too aggressively (oscillation)
- ❌ Don't ignore warmup failures

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Scale-up response | < 30s (after decision) |
| Warmup time | < 120s |
| Drain time | < 60s |
| Decision frequency | Every 10s |
| Stabilization | 5 minute window |

---

## Configuration

```yaml
# configs/autoscale.yaml
autoscale:
  enabled: true
  min_peers: 1
  max_peers: 10
  target_utilization: 0.7
  target_latency_p99: 2s
  queue_threshold: 50
  scale_up_cooldown: 60s
  scale_down_cooldown: 300s
  stabilization_window: 300s
  warmup_timeout: 180s
```

---

**PRP Confidence Score: 7/10**

**Rationale**:
- +2: Well-known scaling patterns (K8s HPA)
- +2: Metrics infrastructure exists
- +1: Clear integration points
- -1: Warmup coordination complexity
- -1: Consumer peer reliability unknown
- -2: Oscillation prevention needs tuning

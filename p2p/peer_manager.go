// Package p2p provides libp2p-based peer-to-peer networking for NeuroGrid.
package p2p

import (
	"context"
	"log"
	"math"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/neurogrid/engine/pkg/metrics"
)

const (
	// Reconnection settings
	initialReconnectDelay = 1 * time.Second
	maxReconnectDelay     = 60 * time.Second
	reconnectBackoffFactor = 2.0

	// Health check settings
	healthCheckInterval = 30 * time.Second
	healthCheckTimeout  = 10 * time.Second
	maxHealthFailures   = 3
)

// PeerState represents the current state of a peer.
type PeerState int

const (
	PeerStateUnknown PeerState = iota
	PeerStateConnected
	PeerStateDisconnected
	PeerStateReconnecting
	PeerStateFailed
)

func (s PeerState) String() string {
	switch s {
	case PeerStateConnected:
		return "connected"
	case PeerStateDisconnected:
		return "disconnected"
	case PeerStateReconnecting:
		return "reconnecting"
	case PeerStateFailed:
		return "failed"
	default:
		return "unknown"
	}
}

// PeerInfo holds information about a tracked peer.
type PeerInfo struct {
	ID              peer.ID
	Addrs           peer.AddrInfo
	State           PeerState
	ReconnectCount  int
	HealthFailures  int
	LastSeen        time.Time
	LastReconnect   time.Time
	mu              sync.RWMutex
}

// PeerManager manages peer connections, health checks, and reconnection.
type PeerManager struct {
	host      host.Host
	peers     map[peer.ID]*PeerInfo
	mu        sync.RWMutex
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup

	// Callbacks
	onPeerConnected    func(peer.ID)
	onPeerDisconnected func(peer.ID)
}

// NewPeerManager creates a new PeerManager for the given host.
func NewPeerManager(h host.Host) *PeerManager {
	ctx, cancel := context.WithCancel(context.Background())

	pm := &PeerManager{
		host:   h,
		peers:  make(map[peer.ID]*PeerInfo),
		ctx:    ctx,
		cancel: cancel,
	}

	// Subscribe to network events
	h.Network().Notify(&network.NotifyBundle{
		ConnectedF:    pm.handleConnected,
		DisconnectedF: pm.handleDisconnected,
	})

	return pm
}

// SetCallbacks sets the connection state callbacks.
func (pm *PeerManager) SetCallbacks(onConnected, onDisconnected func(peer.ID)) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.onPeerConnected = onConnected
	pm.onPeerDisconnected = onDisconnected
}

// AddPeer adds a peer to be managed.
func (pm *PeerManager) AddPeer(pi peer.AddrInfo) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if _, exists := pm.peers[pi.ID]; !exists {
		pm.peers[pi.ID] = &PeerInfo{
			ID:       pi.ID,
			Addrs:    pi,
			State:    PeerStateUnknown,
			LastSeen: time.Now(),
		}
		log.Printf("Added peer %s to manager", pi.ID.ShortString())
	}
}

// RemovePeer removes a peer from management.
func (pm *PeerManager) RemovePeer(id peer.ID) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	delete(pm.peers, id)
	log.Printf("Removed peer %s from manager", id.ShortString())
}

// GetPeerState returns the current state of a peer.
func (pm *PeerManager) GetPeerState(id peer.ID) PeerState {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if info, exists := pm.peers[id]; exists {
		info.mu.RLock()
		defer info.mu.RUnlock()
		return info.State
	}
	return PeerStateUnknown
}

// GetConnectedPeers returns a list of currently connected peer IDs.
func (pm *PeerManager) GetConnectedPeers() []peer.ID {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	var connected []peer.ID
	for id, info := range pm.peers {
		info.mu.RLock()
		if info.State == PeerStateConnected {
			connected = append(connected, id)
		}
		info.mu.RUnlock()
	}
	return connected
}

// Start begins the peer management goroutines.
func (pm *PeerManager) Start() {
	pm.wg.Add(1)
	go pm.healthCheckLoop()
	log.Printf("PeerManager started with %d peers", len(pm.peers))
}

// Stop stops the peer manager and waits for goroutines to finish.
func (pm *PeerManager) Stop() {
	pm.cancel()
	pm.wg.Wait()
	log.Printf("PeerManager stopped")
}

// handleConnected is called when a peer connects.
func (pm *PeerManager) handleConnected(n network.Network, c network.Conn) {
	peerID := c.RemotePeer()

	pm.mu.Lock()
	info, exists := pm.peers[peerID]
	if !exists {
		// Auto-add unknown peers
		info = &PeerInfo{
			ID:    peerID,
			State: PeerStateUnknown,
		}
		pm.peers[peerID] = info
	}
	pm.mu.Unlock()

	info.mu.Lock()
	info.State = PeerStateConnected
	info.LastSeen = time.Now()
	info.HealthFailures = 0
	info.mu.Unlock()

	// Update metrics
	metrics.PeersConnected.Inc()

	log.Printf("Peer connected: %s", peerID.ShortString())

	// Call callback if set
	pm.mu.RLock()
	callback := pm.onPeerConnected
	pm.mu.RUnlock()
	if callback != nil {
		callback(peerID)
	}
}

// handleDisconnected is called when a peer disconnects.
func (pm *PeerManager) handleDisconnected(n network.Network, c network.Conn) {
	peerID := c.RemotePeer()

	pm.mu.RLock()
	info, exists := pm.peers[peerID]
	pm.mu.RUnlock()

	if !exists {
		return
	}

	info.mu.Lock()
	wasConnected := info.State == PeerStateConnected
	info.State = PeerStateDisconnected
	info.mu.Unlock()

	if wasConnected {
		// Update metrics
		metrics.PeersConnected.Dec()

		log.Printf("Peer disconnected: %s", peerID.ShortString())

		// Call callback if set
		pm.mu.RLock()
		callback := pm.onPeerDisconnected
		pm.mu.RUnlock()
		if callback != nil {
			callback(peerID)
		}

		// Start reconnection in background
		go pm.reconnectWithBackoff(info)
	}
}

// reconnectWithBackoff attempts to reconnect to a peer with exponential backoff.
func (pm *PeerManager) reconnectWithBackoff(info *PeerInfo) {
	info.mu.Lock()
	if info.State == PeerStateReconnecting {
		info.mu.Unlock()
		return
	}
	info.State = PeerStateReconnecting
	info.mu.Unlock()

	for {
		select {
		case <-pm.ctx.Done():
			return
		default:
		}

		info.mu.RLock()
		count := info.ReconnectCount
		addrs := info.Addrs
		info.mu.RUnlock()

		// Calculate delay with exponential backoff
		delay := time.Duration(float64(initialReconnectDelay) * math.Pow(reconnectBackoffFactor, float64(count)))
		if delay > maxReconnectDelay {
			delay = maxReconnectDelay
		}

		log.Printf("Attempting reconnect to %s in %v (attempt %d)", info.ID.ShortString(), delay, count+1)

		select {
		case <-pm.ctx.Done():
			return
		case <-time.After(delay):
		}

		// Update metrics
		metrics.PeerReconnectionsTotal.Inc()

		// Attempt connection
		ctx, cancel := context.WithTimeout(pm.ctx, 30*time.Second)
		err := pm.host.Connect(ctx, addrs)
		cancel()

		if err == nil {
			log.Printf("Successfully reconnected to %s", info.ID.ShortString())
			info.mu.Lock()
			info.ReconnectCount = 0
			info.LastReconnect = time.Now()
			info.mu.Unlock()
			return
		}

		log.Printf("Reconnect failed for %s: %v", info.ID.ShortString(), err)

		info.mu.Lock()
		info.ReconnectCount++

		// Give up after too many attempts
		if info.ReconnectCount > 10 {
			info.State = PeerStateFailed
			info.mu.Unlock()
			log.Printf("Giving up on peer %s after %d attempts", info.ID.ShortString(), info.ReconnectCount)
			return
		}
		info.mu.Unlock()
	}
}

// healthCheckLoop periodically checks peer health.
func (pm *PeerManager) healthCheckLoop() {
	defer pm.wg.Done()

	ticker := time.NewTicker(healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-pm.ctx.Done():
			return
		case <-ticker.C:
			pm.checkAllPeers()
		}
	}
}

// checkAllPeers performs health checks on all managed peers.
func (pm *PeerManager) checkAllPeers() {
	pm.mu.RLock()
	peers := make([]*PeerInfo, 0, len(pm.peers))
	for _, info := range pm.peers {
		peers = append(peers, info)
	}
	pm.mu.RUnlock()

	for _, info := range peers {
		pm.checkPeer(info)
	}
}

// checkPeer performs a health check on a single peer.
func (pm *PeerManager) checkPeer(info *PeerInfo) {
	info.mu.RLock()
	if info.State != PeerStateConnected {
		info.mu.RUnlock()
		return
	}
	peerID := info.ID
	info.mu.RUnlock()

	// Check if we're still connected
	conns := pm.host.Network().ConnsToPeer(peerID)
	if len(conns) == 0 {
		info.mu.Lock()
		info.HealthFailures++
		failures := info.HealthFailures
		info.mu.Unlock()

		metrics.PeerHealthCheckFailures.WithLabelValues(peerID.ShortString()).Inc()
		log.Printf("Health check failed for peer %s (%d/%d)", peerID.ShortString(), failures, maxHealthFailures)

		if failures >= maxHealthFailures {
			// Trigger disconnect handling
			pm.handleDisconnected(pm.host.Network(), nil)
		}
	} else {
		// Peer is healthy
		info.mu.Lock()
		info.HealthFailures = 0
		info.LastSeen = time.Now()
		info.mu.Unlock()
	}
}

// ConnectToPeer attempts to connect to a peer.
func (pm *PeerManager) ConnectToPeer(ctx context.Context, pi peer.AddrInfo) error {
	pm.AddPeer(pi)
	return pm.host.Connect(ctx, pi)
}

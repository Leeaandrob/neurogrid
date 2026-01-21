// Package scheduler provides VRAM-aware layer scheduling for distributed inference.
package scheduler

import (
	"fmt"
	"sync"
)

// PeerVRAM tracks VRAM usage for a single peer.
type PeerVRAM struct {
	PeerID   string
	Total    uint64 // Total VRAM in bytes
	Used     uint64 // Currently used VRAM
	Reserved uint64 // Reserved for pending allocations
}

// Available returns the available VRAM (total - used - reserved).
func (p *PeerVRAM) Available() uint64 {
	if p.Total <= p.Used+p.Reserved {
		return 0
	}
	return p.Total - p.Used - p.Reserved
}

// VRAMTracker tracks VRAM usage across all peers.
type VRAMTracker struct {
	peers map[string]*PeerVRAM
	mu    sync.RWMutex
}

// NewVRAMTracker creates a new VRAM tracker.
func NewVRAMTracker() *VRAMTracker {
	return &VRAMTracker{
		peers: make(map[string]*PeerVRAM),
	}
}

// RegisterPeer adds a peer with its VRAM information.
func (t *VRAMTracker) RegisterPeer(peerID string, total, used uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.peers[peerID]; exists {
		return fmt.Errorf("peer already registered: %s", peerID)
	}

	t.peers[peerID] = &PeerVRAM{
		PeerID:   peerID,
		Total:    total,
		Used:     used,
		Reserved: 0,
	}

	return nil
}

// UnregisterPeer removes a peer from tracking.
func (t *VRAMTracker) UnregisterPeer(peerID string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.peers[peerID]; !exists {
		return fmt.Errorf("unknown peer: %s", peerID)
	}

	delete(t.peers, peerID)
	return nil
}

// Reserve reserves VRAM on a peer for pending allocation.
// Returns error if insufficient VRAM is available.
func (t *VRAMTracker) Reserve(peerID string, size uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	peer, ok := t.peers[peerID]
	if !ok {
		return fmt.Errorf("unknown peer: %s", peerID)
	}

	available := peer.Available()
	if size > available {
		return fmt.Errorf("insufficient VRAM on %s: need %d bytes, have %d bytes", peerID, size, available)
	}

	peer.Reserved += size
	return nil
}

// Commit converts a reservation to actual usage.
func (t *VRAMTracker) Commit(peerID string, size uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	peer, ok := t.peers[peerID]
	if !ok {
		return fmt.Errorf("unknown peer: %s", peerID)
	}

	if size > peer.Reserved {
		return fmt.Errorf("commit size %d exceeds reservation %d", size, peer.Reserved)
	}

	peer.Reserved -= size
	peer.Used += size
	return nil
}

// Release frees used VRAM on a peer.
func (t *VRAMTracker) Release(peerID string, size uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	peer, ok := t.peers[peerID]
	if !ok {
		return fmt.Errorf("unknown peer: %s", peerID)
	}

	if size > peer.Used {
		return fmt.Errorf("release size %d exceeds used %d", size, peer.Used)
	}

	peer.Used -= size
	return nil
}

// CancelReservation cancels a pending reservation.
func (t *VRAMTracker) CancelReservation(peerID string, size uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	peer, ok := t.peers[peerID]
	if !ok {
		return fmt.Errorf("unknown peer: %s", peerID)
	}

	if size > peer.Reserved {
		return fmt.Errorf("cancel size %d exceeds reservation %d", size, peer.Reserved)
	}

	peer.Reserved -= size
	return nil
}

// GetPeerInfo returns VRAM info for a peer.
func (t *VRAMTracker) GetPeerInfo(peerID string) (*PeerVRAM, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	peer, ok := t.peers[peerID]
	if !ok {
		return nil, fmt.Errorf("unknown peer: %s", peerID)
	}

	// Return a copy to avoid race conditions
	return &PeerVRAM{
		PeerID:   peer.PeerID,
		Total:    peer.Total,
		Used:     peer.Used,
		Reserved: peer.Reserved,
	}, nil
}

// GetAllPeers returns info for all registered peers.
func (t *VRAMTracker) GetAllPeers() []*PeerVRAM {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make([]*PeerVRAM, 0, len(t.peers))
	for _, peer := range t.peers {
		result = append(result, &PeerVRAM{
			PeerID:   peer.PeerID,
			Total:    peer.Total,
			Used:     peer.Used,
			Reserved: peer.Reserved,
		})
	}

	return result
}

// TotalAvailable returns the sum of available VRAM across all peers.
func (t *VRAMTracker) TotalAvailable() uint64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var total uint64
	for _, peer := range t.peers {
		total += peer.Available()
	}
	return total
}

// UpdateUsage updates the current usage for a peer.
func (t *VRAMTracker) UpdateUsage(peerID string, used uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	peer, ok := t.peers[peerID]
	if !ok {
		return fmt.Errorf("unknown peer: %s", peerID)
	}

	peer.Used = used
	return nil
}

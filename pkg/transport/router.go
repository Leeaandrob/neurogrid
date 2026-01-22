// Package transport provides the TransportRouter for routing activations to peers.
package transport

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

var (
	// ErrPeerNotFound is returned when a peer is not registered.
	ErrPeerNotFound = errors.New("peer not found")

	// ErrLayerNotAssigned is returned when a layer has no peer assignment.
	ErrLayerNotAssigned = errors.New("layer not assigned to any peer")

	// ErrTransportNotFound is returned when a transport is not found.
	ErrTransportNotFound = errors.New("transport not found")
)

// TransportRouter routes activations to the appropriate transport based on layer assignment.
type TransportRouter struct {
	localTransports  map[int]Transport    // Device ID -> transport
	remoteTransports map[string]Transport // Peer ID -> transport
	layerToPeer      map[int]string       // Layer ID -> Peer ID
	mu               sync.RWMutex
}

// NewTransportRouter creates a new transport router.
func NewTransportRouter() *TransportRouter {
	return &TransportRouter{
		localTransports:  make(map[int]Transport),
		remoteTransports: make(map[string]Transport),
		layerToPeer:      make(map[int]string),
	}
}

// RegisterLocalTransport registers a transport for a local GPU device.
func (r *TransportRouter) RegisterLocalTransport(deviceID int, t Transport) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.localTransports[deviceID] = t
	return nil
}

// GetLocalTransport returns the transport for a local device.
func (r *TransportRouter) GetLocalTransport(deviceID int) (Transport, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	t, ok := r.localTransports[deviceID]
	if !ok {
		return nil, ErrTransportNotFound
	}
	return t, nil
}

// RegisterRemoteTransport registers a transport for a remote peer.
func (r *TransportRouter) RegisterRemoteTransport(peerID string, t Transport) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.remoteTransports[peerID] = t
	return nil
}

// GetRemoteTransport returns the transport for a remote peer.
func (r *TransportRouter) GetRemoteTransport(peerID string) (Transport, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	t, ok := r.remoteTransports[peerID]
	if !ok {
		return nil, ErrTransportNotFound
	}
	return t, nil
}

// AssignLayerToPeer assigns a layer to a peer for routing.
func (r *TransportRouter) AssignLayerToPeer(layerID int, peerID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.layerToPeer[layerID] = peerID
	return nil
}

// GetPeerForLayer returns the peer ID assigned to a layer.
func (r *TransportRouter) GetPeerForLayer(layerID int) (string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	peerID, ok := r.layerToPeer[layerID]
	if !ok {
		return "", ErrLayerNotAssigned
	}
	return peerID, nil
}

// RouteActivation routes activation data to the appropriate transport.
func (r *TransportRouter) RouteActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error {
	r.mu.RLock()
	peerID, ok := r.layerToPeer[layerID]
	r.mu.RUnlock()

	if !ok {
		return fmt.Errorf("no peer assigned for layer %d", layerID)
	}

	// Check local first (using device ID derived from peer ID)
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Try remote transport
	if transport, ok := r.remoteTransports[peerID]; ok {
		return transport.SendActivation(ctx, layerID, seqID, data)
	}

	// Try to find by examining local transports
	for _, transport := range r.localTransports {
		if transport.PeerInfo().ID == peerID {
			return transport.SendActivation(ctx, layerID, seqID, data)
		}
	}

	return fmt.Errorf("transport not found for peer %s", peerID)
}

// ReceiveActivation receives activation data from any registered transport.
// It polls all transports and returns the first activation received.
// This is typically used by a worker waiting for incoming activations.
func (r *TransportRouter) ReceiveActivation(ctx context.Context) (int, uint64, []byte, string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Create channels for results
	type recvResult struct {
		layerID int
		seqID   uint64
		data    []byte
		peerID  string
		err     error
	}

	resultChan := make(chan recvResult, len(r.localTransports)+len(r.remoteTransports))

	// Try to receive from all transports concurrently
	var wg sync.WaitGroup

	// Poll local transports
	for deviceID, transport := range r.localTransports {
		wg.Add(1)
		go func(devID int, t Transport) {
			defer wg.Done()
			layerID, seqID, data, err := t.RecvActivation(ctx)
			if err == nil {
				resultChan <- recvResult{layerID, seqID, data, t.PeerInfo().ID, nil}
			}
		}(deviceID, transport)
	}

	// Poll remote transports
	for peerID, transport := range r.remoteTransports {
		wg.Add(1)
		go func(pID string, t Transport) {
			defer wg.Done()
			layerID, seqID, data, err := t.RecvActivation(ctx)
			if err == nil {
				resultChan <- recvResult{layerID, seqID, data, pID, nil}
			}
		}(peerID, transport)
	}

	// Wait for first result or context cancellation
	select {
	case result := <-resultChan:
		return result.layerID, result.seqID, result.data, result.peerID, result.err
	case <-ctx.Done():
		return 0, 0, nil, "", ctx.Err()
	}
}

// UpdateLayerAssignments updates layer assignments in bulk.
// This is typically called after the scheduler computes new assignments.
func (r *TransportRouter) UpdateLayerAssignments(assignments map[int]string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for layerID, peerID := range assignments {
		r.layerToPeer[layerID] = peerID
	}
}

// GetAllLayerAssignments returns a copy of all layer-to-peer assignments.
func (r *TransportRouter) GetAllLayerAssignments() map[int]string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[int]string, len(r.layerToPeer))
	for k, v := range r.layerToPeer {
		result[k] = v
	}
	return result
}

// GetRegisteredPeers returns the list of registered remote peer IDs.
func (r *TransportRouter) GetRegisteredPeers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	peers := make([]string, 0, len(r.remoteTransports))
	for peerID := range r.remoteTransports {
		peers = append(peers, peerID)
	}
	return peers
}

// GetRegisteredDevices returns the list of registered local device IDs.
func (r *TransportRouter) GetRegisteredDevices() []int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	devices := make([]int, 0, len(r.localTransports))
	for deviceID := range r.localTransports {
		devices = append(devices, deviceID)
	}
	return devices
}

// Close closes all registered transports.
func (r *TransportRouter) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var lastErr error
	for _, t := range r.localTransports {
		if err := t.Close(); err != nil {
			lastErr = err
		}
	}
	for _, t := range r.remoteTransports {
		if err := t.Close(); err != nil {
			lastErr = err
		}
	}

	r.localTransports = make(map[int]Transport)
	r.remoteTransports = make(map[string]Transport)
	r.layerToPeer = make(map[int]string)

	return lastErr
}

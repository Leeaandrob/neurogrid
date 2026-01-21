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

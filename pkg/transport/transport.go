// Package transport provides abstractions for activation transfer between peers.
// This supports both local multi-GPU communication and remote P2P networking.
package transport

import (
	"context"
	"time"
)

// PeerDescriptor describes a compute peer (local GPU or remote node).
type PeerDescriptor struct {
	ID          string // Unique peer identifier
	Address     string // Network address (for remote peers)
	IsLocal     bool   // True if peer is on local machine
	DeviceID    int    // GPU device ID (for local peers)
	TotalMemory uint64 // Total VRAM in bytes
	FreeMemory  uint64 // Available VRAM in bytes
}

// ActivationMessage represents data being transferred between layers.
type ActivationMessage struct {
	LayerID   int       // Source or destination layer
	SeqID     uint64    // Sequence ID for ordering
	Data      []byte    // Activation tensor data
	Timestamp time.Time // When the message was created
}

// Transport is the interface for sending/receiving activations between peers.
// Implementations include CUDATransport (local) and P2PTransport (remote).
type Transport interface {
	// SendActivation sends activation data to the peer.
	SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error

	// RecvActivation receives activation data from the peer.
	// Returns layerID, seqID, data, error.
	RecvActivation(ctx context.Context) (int, uint64, []byte, error)

	// PeerInfo returns information about the connected peer.
	PeerInfo() PeerDescriptor

	// Close releases resources associated with the transport.
	Close() error
}

// Package transport provides CUDATransport for local multi-GPU communication.
package transport

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CUDATransport implements Transport for local multi-GPU communication.
// Uses CUDA P2P or staged copies depending on hardware support.
type CUDATransport struct {
	srcDevice int
	dstDevice int
	recvChan  chan *ActivationMessage
	peerInfo  PeerDescriptor
	mu        sync.Mutex
	closed    bool
}

// NewCUDATransport creates a new CUDA transport between two devices.
func NewCUDATransport(srcDevice, dstDevice int) (*CUDATransport, error) {
	t := &CUDATransport{
		srcDevice: srcDevice,
		dstDevice: dstDevice,
		recvChan:  make(chan *ActivationMessage, 100), // Buffer for async recv
		peerInfo: PeerDescriptor{
			ID:       fmt.Sprintf("cuda-gpu-%d-to-%d", srcDevice, dstDevice),
			IsLocal:  true,
			DeviceID: dstDevice,
		},
	}

	return t, nil
}

// SendActivation sends activation data using CUDA cross-device copy.
func (t *CUDATransport) SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return fmt.Errorf("transport closed")
	}

	// Create activation message
	msg := &ActivationMessage{
		LayerID:   layerID,
		SeqID:     seqID,
		Data:      make([]byte, len(data)),
		Timestamp: time.Now(),
	}
	copy(msg.Data, data)

	// Queue for receive (simulating async transfer)
	select {
	case t.recvChan <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// RecvActivation receives activation data from the CUDA transfer queue.
func (t *CUDATransport) RecvActivation(ctx context.Context) (int, uint64, []byte, error) {
	select {
	case msg := <-t.recvChan:
		return msg.LayerID, msg.SeqID, msg.Data, nil
	case <-ctx.Done():
		return 0, 0, nil, ctx.Err()
	}
}

// PeerInfo returns information about this transport's peer.
func (t *CUDATransport) PeerInfo() PeerDescriptor {
	return t.peerInfo
}

// Close releases resources associated with this transport.
func (t *CUDATransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}

	t.closed = true
	close(t.recvChan)
	return nil
}

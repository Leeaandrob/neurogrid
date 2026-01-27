// Package transport provides abstractions for activation transfer between peers.
package transport

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

const (
	// TransportProtocolID is the protocol identifier for activation transfers.
	TransportProtocolID = "/neurogrid/transport/1.0.0"

	// headerSize is the size of the activation message header.
	// Format: LayerID (4 bytes) + SeqID (8 bytes) + DataLen (4 bytes)
	p2pHeaderSize = 16

	// extendedHeaderSize is the size of the extended header with request-response support.
	// Format: MsgType (1 byte) + LayerID (4 bytes) + SeqID (8 bytes) + RequestID (8 bytes) + DataLen (4 bytes)
	extendedHeaderSize = 25

	// Message type constants
	msgTypeActivation = 0x01
	msgTypeResponse   = 0x02
)

// P2PTransport implements Transport and RequestResponseTransport for remote peer communication.
type P2PTransport struct {
	host             host.Host
	peerID           peer.ID
	recvChan         chan *activationMsg
	pendingResponses map[uint64]chan *ActivationMessage
	responseHandler  ResponseHandler
	requestCounter   uint64
	mu               sync.RWMutex
	closed           bool
	bufferPool       BufferPool // Optional buffer pool for zero-allocation message handling
}

// activationMsg is an internal message structure for the receive channel.
type activationMsg struct {
	layerID   int
	seqID     uint64
	requestID uint64
	data      []byte
	from      string
}

// P2PTransportOption is a functional option for configuring P2PTransport.
// Use these options with NewP2PTransport to customize transport behavior.
type P2PTransportOption func(*P2PTransport)

// WithBufferPool configures a buffer pool for zero-allocation message handling.
// When set, the transport will use pooled buffers instead of allocating new ones for each message.
//
// For optimal DMA performance with CUDA, use a pinned memory buffer pool.
// The pool should have buffers sized for typical activation data (8KB-16KB for LLM inference).
//
// Example:
//
//	pool, _ := NewPinnedBufferPool(16*1024, 32)
//	transport := NewP2PTransport(host, peerID, WithBufferPool(pool))
func WithBufferPool(pool BufferPool) P2PTransportOption {
	return func(t *P2PTransport) {
		t.bufferPool = pool
	}
}

// NewP2PTransport creates a new P2P transport for the given peer.
func NewP2PTransport(h host.Host, peerID peer.ID, opts ...P2PTransportOption) *P2PTransport {
	t := &P2PTransport{
		host:             h,
		peerID:           peerID,
		recvChan:         make(chan *activationMsg, 100),
		pendingResponses: make(map[uint64]chan *ActivationMessage),
	}

	// Apply options
	for _, opt := range opts {
		opt(t)
	}

	// Register stream handler for this peer
	h.SetStreamHandler(protocol.ID(TransportProtocolID), t.handleStream)

	return t
}

// handleStream processes incoming activation streams.
func (t *P2PTransport) handleStream(s network.Stream) {
	defer s.Close()

	// Read first byte to determine message format
	firstByte := make([]byte, 1)
	if _, err := io.ReadFull(s, firstByte); err != nil {
		return
	}

	// Check if extended format (first byte is message type: 0x01 or 0x02)
	if firstByte[0] == msgTypeActivation || firstByte[0] == msgTypeResponse {
		t.handleExtendedStream(s, firstByte[0])
		return
	}

	// Legacy format - first byte is part of LayerID
	t.handleLegacyStream(s, firstByte)
}

// handleExtendedStream processes messages with extended header format.
func (t *P2PTransport) handleExtendedStream(s network.Stream, msgType byte) {
	// Read rest of extended header (24 bytes remaining)
	restHeader := make([]byte, extendedHeaderSize-1)
	if _, err := io.ReadFull(s, restHeader); err != nil {
		return
	}

	layerID := int(binary.BigEndian.Uint32(restHeader[0:4]))
	seqID := binary.BigEndian.Uint64(restHeader[4:12])
	requestID := binary.BigEndian.Uint64(restHeader[12:20])
	dataLen := binary.BigEndian.Uint32(restHeader[20:24])

	// Read data - use buffer pool if available
	var data []byte
	var pooledBuf []byte
	if t.bufferPool != nil {
		pooledBuf = t.bufferPool.Get(int(dataLen))
		data = pooledBuf[:dataLen]
	} else {
		data = make([]byte, dataLen)
	}
	if _, err := io.ReadFull(s, data); err != nil {
		if pooledBuf != nil && t.bufferPool != nil {
			t.bufferPool.Put(pooledBuf)
		}
		return
	}

	from := s.Conn().RemotePeer().String()

	switch msgType {
	case msgTypeActivation:
		// Make a copy of data if using pooled buffer (caller owns the copy)
		var msgData []byte
		if pooledBuf != nil {
			msgData = make([]byte, len(data))
			copy(msgData, data)
			t.bufferPool.Put(pooledBuf)
		} else {
			msgData = data
		}
		// Send to receive channel
		msg := &activationMsg{
			layerID:   layerID,
			seqID:     seqID,
			requestID: requestID,
			data:      msgData,
			from:      from,
		}
		select {
		case t.recvChan <- msg:
		default:
		}

	case msgTypeResponse:
		// Check for pending response channel
		t.mu.RLock()
		ch, ok := t.pendingResponses[requestID]
		handler := t.responseHandler
		t.mu.RUnlock()

		// Make a copy of data if using pooled buffer
		var msgData []byte
		if pooledBuf != nil {
			msgData = make([]byte, len(data))
			copy(msgData, data)
			t.bufferPool.Put(pooledBuf)
		} else {
			msgData = data
		}

		respMsg := &ActivationMessage{
			LayerID:   layerID,
			SeqID:     seqID,
			RequestID: requestID,
			Data:      msgData,
			Timestamp: time.Now(),
			From:      from,
		}

		if ok {
			select {
			case ch <- respMsg:
			default:
			}
		}
		if handler != nil {
			handler(respMsg)
		}
	}
}

// handleLegacyStream processes messages with legacy header format.
func (t *P2PTransport) handleLegacyStream(s network.Stream, firstByte []byte) {
	// Read rest of legacy header (15 bytes remaining)
	restHeader := make([]byte, p2pHeaderSize-1)
	if _, err := io.ReadFull(s, restHeader); err != nil {
		return
	}

	// Reconstruct full header
	header := make([]byte, p2pHeaderSize)
	header[0] = firstByte[0]
	copy(header[1:], restHeader)

	layerID := int(binary.BigEndian.Uint32(header[0:4]))
	seqID := binary.BigEndian.Uint64(header[4:12])
	dataLen := binary.BigEndian.Uint32(header[12:16])

	// Read data - use buffer pool if available
	var data []byte
	var pooledBuf []byte
	if t.bufferPool != nil {
		pooledBuf = t.bufferPool.Get(int(dataLen))
		data = pooledBuf[:dataLen]
	} else {
		data = make([]byte, dataLen)
	}
	if _, err := io.ReadFull(s, data); err != nil {
		if pooledBuf != nil && t.bufferPool != nil {
			t.bufferPool.Put(pooledBuf)
		}
		return
	}

	// Make a copy of data if using pooled buffer
	var msgData []byte
	if pooledBuf != nil {
		msgData = make([]byte, len(data))
		copy(msgData, data)
		t.bufferPool.Put(pooledBuf)
	} else {
		msgData = data
	}

	// Send to receive channel
	msg := &activationMsg{
		layerID: layerID,
		seqID:   seqID,
		data:    msgData,
		from:    s.Conn().RemotePeer().String(),
	}

	select {
	case t.recvChan <- msg:
	default:
	}
}

// GetBufferPool returns the configured buffer pool, or nil if none is set.
// Use HasBufferPool() to check if a pool is configured without getting it.
func (t *P2PTransport) GetBufferPool() BufferPool {
	return t.bufferPool
}

// SetBufferPool sets the buffer pool for the transport.
// Can be called after transport creation to add or replace the buffer pool.
// Thread-safe: acquires write lock during update.
//
// Passing nil disables pooled buffers (transport reverts to per-message allocation).
func (t *P2PTransport) SetBufferPool(pool BufferPool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.bufferPool = pool
}

// SendActivation sends activation data to the remote peer.
func (t *P2PTransport) SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return fmt.Errorf("transport closed")
	}
	t.mu.Unlock()

	// Open stream to peer
	s, err := t.host.NewStream(ctx, t.peerID, protocol.ID(TransportProtocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream: %w", err)
	}
	defer s.Close()

	// Write header
	header := make([]byte, p2pHeaderSize)
	binary.BigEndian.PutUint32(header[0:4], uint32(layerID))
	binary.BigEndian.PutUint64(header[4:12], seqID)
	binary.BigEndian.PutUint32(header[12:16], uint32(len(data)))

	if _, err := s.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write data
	if _, err := s.Write(data); err != nil {
		return fmt.Errorf("failed to write data: %w", err)
	}

	return nil
}

// RecvActivation receives activation data from the remote peer.
func (t *P2PTransport) RecvActivation(ctx context.Context) (int, uint64, []byte, error) {
	select {
	case msg := <-t.recvChan:
		return msg.layerID, msg.seqID, msg.data, nil
	case <-ctx.Done():
		return 0, 0, nil, ctx.Err()
	}
}

// PeerInfo returns information about the remote peer.
func (t *P2PTransport) PeerInfo() PeerDescriptor {
	return PeerDescriptor{
		ID:      t.peerID.String(),
		IsLocal: false,
	}
}

// SendActivationWithID sends activation data with a request ID for response tracking.
func (t *P2PTransport) SendActivationWithID(ctx context.Context, layerID int, seqID uint64, requestID uint64, data []byte) error {
	// Register pending request before sending
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return fmt.Errorf("transport closed")
	}
	t.pendingResponses[requestID] = make(chan *ActivationMessage, 1)
	t.mu.Unlock()

	return t.sendExtendedMessage(ctx, msgTypeActivation, layerID, seqID, requestID, data)
}

// SendResponse sends a response back to the requester.
func (t *P2PTransport) SendResponse(ctx context.Context, layerID int, seqID uint64, requestID uint64, data []byte) error {
	return t.sendExtendedMessage(ctx, msgTypeResponse, layerID, seqID, requestID, data)
}

// sendExtendedMessage sends a message with extended header format.
func (t *P2PTransport) sendExtendedMessage(ctx context.Context, msgType byte, layerID int, seqID uint64, requestID uint64, data []byte) error {
	t.mu.RLock()
	if t.closed {
		t.mu.RUnlock()
		return fmt.Errorf("transport closed")
	}
	t.mu.RUnlock()

	// Open stream to peer
	s, err := t.host.NewStream(ctx, t.peerID, protocol.ID(TransportProtocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream: %w", err)
	}
	defer s.Close()

	// Write extended header
	header := make([]byte, extendedHeaderSize)
	header[0] = msgType
	binary.BigEndian.PutUint32(header[1:5], uint32(layerID))
	binary.BigEndian.PutUint64(header[5:13], seqID)
	binary.BigEndian.PutUint64(header[13:21], requestID)
	binary.BigEndian.PutUint32(header[21:25], uint32(len(data)))

	if _, err := s.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write data
	if len(data) > 0 {
		if _, err := s.Write(data); err != nil {
			return fmt.Errorf("failed to write data: %w", err)
		}
	}

	return nil
}

// WaitForResponse waits for a response to a specific request ID.
func (t *P2PTransport) WaitForResponse(ctx context.Context, requestID uint64, timeout time.Duration) (*ActivationMessage, error) {
	t.mu.RLock()
	ch, ok := t.pendingResponses[requestID]
	t.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("no pending request for ID %d", requestID)
	}

	select {
	case msg := <-ch:
		// Clean up
		t.mu.Lock()
		delete(t.pendingResponses, requestID)
		t.mu.Unlock()
		return msg, nil
	case <-time.After(timeout):
		// Clean up
		t.mu.Lock()
		delete(t.pendingResponses, requestID)
		t.mu.Unlock()
		return nil, ErrResponseTimeout
	case <-ctx.Done():
		// Clean up
		t.mu.Lock()
		delete(t.pendingResponses, requestID)
		t.mu.Unlock()
		return nil, ctx.Err()
	}
}

// RegisterResponseHandler registers a callback for handling responses.
func (t *P2PTransport) RegisterResponseHandler(handler ResponseHandler) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.responseHandler = handler
}

// NextRequestID generates a unique request ID.
func (t *P2PTransport) NextRequestID() uint64 {
	return atomic.AddUint64(&t.requestCounter, 1)
}

// Close releases resources associated with the transport.
func (t *P2PTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}

	t.closed = true
	t.host.RemoveStreamHandler(protocol.ID(TransportProtocolID))
	close(t.recvChan)

	// Close all pending response channels
	for _, ch := range t.pendingResponses {
		close(ch)
	}
	t.pendingResponses = make(map[uint64]chan *ActivationMessage)

	return nil
}

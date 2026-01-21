// Package transport provides abstractions for activation transfer between peers.
package transport

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"sync"

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
)

// P2PTransport implements Transport for remote peer communication via libp2p.
type P2PTransport struct {
	host     host.Host
	peerID   peer.ID
	recvChan chan *activationMsg
	mu       sync.Mutex
	closed   bool
}

// activationMsg is an internal message structure for the receive channel.
type activationMsg struct {
	layerID int
	seqID   uint64
	data    []byte
}

// NewP2PTransport creates a new P2P transport for the given peer.
func NewP2PTransport(h host.Host, peerID peer.ID) *P2PTransport {
	t := &P2PTransport{
		host:     h,
		peerID:   peerID,
		recvChan: make(chan *activationMsg, 100),
	}

	// Register stream handler for this peer
	h.SetStreamHandler(protocol.ID(TransportProtocolID), t.handleStream)

	return t
}

// handleStream processes incoming activation streams.
func (t *P2PTransport) handleStream(s network.Stream) {
	defer s.Close()

	// Read header
	header := make([]byte, p2pHeaderSize)
	if _, err := io.ReadFull(s, header); err != nil {
		return
	}

	layerID := int(binary.BigEndian.Uint32(header[0:4]))
	seqID := binary.BigEndian.Uint64(header[4:12])
	dataLen := binary.BigEndian.Uint32(header[12:16])

	// Read data
	data := make([]byte, dataLen)
	if _, err := io.ReadFull(s, data); err != nil {
		return
	}

	// Send to receive channel
	msg := &activationMsg{
		layerID: layerID,
		seqID:   seqID,
		data:    data,
	}

	select {
	case t.recvChan <- msg:
	default:
		// Channel full, drop message
	}
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

	return nil
}

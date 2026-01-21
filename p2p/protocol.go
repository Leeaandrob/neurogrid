// Package p2p provides libp2p-based peer-to-peer networking for NeuroGrid.
package p2p

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
	// TensorProtocolID is the protocol identifier for tensor transfers.
	TensorProtocolID = "/neurogrid/tensor/1.0.0"

	// headerSize is the fixed size of the tensor message header.
	// Format: LayerID (4 bytes) + SeqID (8 bytes) + DataLen (4 bytes)
	headerSize = 16
)

// TensorMessage represents a tensor being transferred over the network.
type TensorMessage struct {
	LayerID int
	SeqID   uint64
	Data    []byte
	From    peer.ID
}

// TensorHandler is a callback for received tensors.
type TensorHandler func(msg *TensorMessage)

// Protocol manages the tensor transfer protocol.
type Protocol struct {
	host    host.Host
	handler TensorHandler
	mu      sync.RWMutex
}

// NewProtocol creates a new tensor transfer protocol handler.
func NewProtocol(h host.Host) *Protocol {
	p := &Protocol{
		host: h,
	}

	// Register stream handler
	h.SetStreamHandler(protocol.ID(TensorProtocolID), p.handleStream)

	return p
}

// OnTensorReceived sets the callback for received tensors.
func (p *Protocol) OnTensorReceived(handler TensorHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.handler = handler
}

// decodeHeader extracts layerID, seqID, and dataLen from a header buffer.
func decodeHeader(header []byte) (layerID int, seqID uint64, dataLen uint32) {
	layerID = int(binary.BigEndian.Uint32(header[0:4]))
	seqID = binary.BigEndian.Uint64(header[4:12])
	dataLen = binary.BigEndian.Uint32(header[12:16])
	return
}

// encodeHeader writes layerID, seqID, and dataLen into a header buffer.
func encodeHeader(header []byte, layerID int, seqID uint64, dataLen int) {
	binary.BigEndian.PutUint32(header[0:4], uint32(layerID))
	binary.BigEndian.PutUint64(header[4:12], seqID)
	binary.BigEndian.PutUint32(header[12:16], uint32(dataLen))
}

// handleStream processes incoming tensor streams.
func (p *Protocol) handleStream(s network.Stream) {
	defer s.Close()

	// Read header
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(s, header); err != nil {
		return
	}

	layerID, seqID, dataLen := decodeHeader(header)

	// Read tensor data
	data := make([]byte, dataLen)
	if _, err := io.ReadFull(s, data); err != nil {
		return
	}

	// Create message
	msg := &TensorMessage{
		LayerID: layerID,
		SeqID:   seqID,
		Data:    data,
		From:    s.Conn().RemotePeer(),
	}

	// Invoke handler
	p.mu.RLock()
	handler := p.handler
	p.mu.RUnlock()

	if handler != nil {
		handler(msg)
	}
}

// SendTensor sends tensor data to a remote peer.
func (p *Protocol) SendTensor(ctx context.Context, peerID peer.ID, layerID int, seqID uint64, data []byte) error {
	// Open stream to peer
	s, err := p.host.NewStream(ctx, peerID, protocol.ID(TensorProtocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream: %w", err)
	}
	defer s.Close()

	// Write header
	header := make([]byte, headerSize)
	encodeHeader(header, layerID, seqID, len(data))

	if _, err := s.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write tensor data
	if _, err := s.Write(data); err != nil {
		return fmt.Errorf("failed to write data: %w", err)
	}

	return nil
}

// Close shuts down the protocol handler.
func (p *Protocol) Close() error {
	p.host.RemoveStreamHandler(protocol.ID(TensorProtocolID))
	return nil
}

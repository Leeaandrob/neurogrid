// Package p2p provides libp2p-based peer-to-peer networking for NeuroGrid.
package p2p

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

const (
	// TensorProtocolID is the protocol identifier for tensor transfers.
	TensorProtocolID = "/neurogrid/tensor/1.0.0"

	// headerSize is the fixed size of the tensor message header (legacy).
	// Format: LayerID (4 bytes) + SeqID (8 bytes) + DataLen (4 bytes)
	headerSize = 16

	// ExtendedHeaderSize is the size of the extended header with message type and request ID.
	// Format: MsgType (1 byte) + LayerID (4 bytes) + SeqID (8 bytes) + RequestID (8 bytes) + DataLen (4 bytes)
	ExtendedHeaderSize = 25

	// Message types for the tensor protocol
	MsgTypeActivation = 0x01 // Forward activation to remote peer
	MsgTypeResponse   = 0x02 // Return computed hidden state
	MsgTypeWeights    = 0x03 // Transfer layer weights
	MsgTypeWeightsAck = 0x04 // Acknowledge weights received

	// WeightChunkSize is the maximum size of a weight chunk (1MB)
	WeightChunkSize = 1024 * 1024
)

// ErrResponseTimeout is returned when waiting for a response times out.
var ErrResponseTimeout = fmt.Errorf("response timeout")

// TensorMessage represents a tensor being transferred over the network.
type TensorMessage struct {
	MsgType   uint8
	LayerID   int
	SeqID     uint64
	RequestID uint64
	Data      []byte
	From      peer.ID
}

// TensorHandler is a callback for received tensors.
type TensorHandler func(msg *TensorMessage)

// WeightsHandler is a callback for received weight chunks.
type WeightsHandler func(layerID int, chunkIndex int, totalChunks int, data []byte)

// Protocol manages the tensor transfer protocol.
type Protocol struct {
	host              host.Host
	handler           TensorHandler        // Legacy handler
	activationHandler TensorHandler        // Handler for activation messages
	responseHandler   TensorHandler        // Handler for response messages
	weightsHandler    WeightsHandler       // Handler for weight chunks
	pendingResponses  map[uint64]chan *TensorMessage // RequestID -> response channel
	pendingWeightsAck map[string]chan struct{}       // peerID:layerID -> ack channel
	mu                sync.RWMutex
}

// NewProtocol creates a new tensor transfer protocol handler.
func NewProtocol(h host.Host) *Protocol {
	p := &Protocol{
		host:              h,
		pendingResponses:  make(map[uint64]chan *TensorMessage),
		pendingWeightsAck: make(map[string]chan struct{}),
	}

	// Register stream handler
	h.SetStreamHandler(protocol.ID(TensorProtocolID), p.handleStream)

	return p
}

// OnTensorReceived sets the callback for received tensors (legacy).
func (p *Protocol) OnTensorReceived(handler TensorHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.handler = handler
}

// OnActivationReceived sets the callback for activation messages.
func (p *Protocol) OnActivationReceived(handler TensorHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.activationHandler = handler
}

// OnResponseReceived sets the callback for response messages.
func (p *Protocol) OnResponseReceived(handler TensorHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.responseHandler = handler
}

// OnWeightsReceived sets the callback for weight chunk messages.
func (p *Protocol) OnWeightsReceived(handler WeightsHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.weightsHandler = handler
}

// RegisterPendingRequest creates a pending response channel for a request ID.
func (p *Protocol) RegisterPendingRequest(requestID uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pendingResponses[requestID] = make(chan *TensorMessage, 1)
}

// decodeHeader extracts layerID, seqID, and dataLen from a header buffer (legacy).
func decodeHeader(header []byte) (layerID int, seqID uint64, dataLen uint32) {
	layerID = int(binary.BigEndian.Uint32(header[0:4]))
	seqID = binary.BigEndian.Uint64(header[4:12])
	dataLen = binary.BigEndian.Uint32(header[12:16])
	return
}

// encodeHeader writes layerID, seqID, and dataLen into a header buffer (legacy).
func encodeHeader(header []byte, layerID int, seqID uint64, dataLen int) {
	binary.BigEndian.PutUint32(header[0:4], uint32(layerID))
	binary.BigEndian.PutUint64(header[4:12], seqID)
	binary.BigEndian.PutUint32(header[12:16], uint32(dataLen))
}

// DecodeExtendedHeader extracts all fields from an extended header buffer.
// Format: MsgType (1B) + LayerID (4B) + SeqID (8B) + RequestID (8B) + DataLen (4B) = 25B
func DecodeExtendedHeader(header []byte) (msgType uint8, layerID int, seqID uint64, requestID uint64, dataLen uint32) {
	msgType = header[0]
	layerID = int(binary.BigEndian.Uint32(header[1:5]))
	seqID = binary.BigEndian.Uint64(header[5:13])
	requestID = binary.BigEndian.Uint64(header[13:21])
	dataLen = binary.BigEndian.Uint32(header[21:25])
	return
}

// EncodeExtendedHeader writes all fields into an extended header buffer.
func EncodeExtendedHeader(header []byte, msgType uint8, layerID int, seqID uint64, requestID uint64, dataLen int) {
	header[0] = msgType
	binary.BigEndian.PutUint32(header[1:5], uint32(layerID))
	binary.BigEndian.PutUint64(header[5:13], seqID)
	binary.BigEndian.PutUint64(header[13:21], requestID)
	binary.BigEndian.PutUint32(header[21:25], uint32(dataLen))
}

// handleStream processes incoming tensor streams.
func (p *Protocol) handleStream(s network.Stream) {
	defer s.Close()

	// Peek first byte to determine header type
	firstByte := make([]byte, 1)
	if _, err := io.ReadFull(s, firstByte); err != nil {
		return
	}

	// Check if this is an extended header (message type byte)
	msgType := firstByte[0]
	if msgType >= MsgTypeActivation && msgType <= MsgTypeWeightsAck {
		// Extended header format
		p.handleExtendedMessage(s, msgType)
	} else {
		// Legacy header format - first byte was part of LayerID
		p.handleLegacyMessage(s, firstByte)
	}
}

// handleExtendedMessage processes messages with extended header format.
func (p *Protocol) handleExtendedMessage(s network.Stream, msgType uint8) {
	// Read rest of extended header (24 bytes remaining)
	restHeader := make([]byte, ExtendedHeaderSize-1)
	if _, err := io.ReadFull(s, restHeader); err != nil {
		return
	}

	// Reconstruct full header
	header := make([]byte, ExtendedHeaderSize)
	header[0] = msgType
	copy(header[1:], restHeader)

	_, layerID, seqID, requestID, dataLen := DecodeExtendedHeader(header)

	// Read data
	data := make([]byte, dataLen)
	if _, err := io.ReadFull(s, data); err != nil {
		return
	}

	// Create message
	msg := &TensorMessage{
		MsgType:   msgType,
		LayerID:   layerID,
		SeqID:     seqID,
		RequestID: requestID,
		Data:      data,
		From:      s.Conn().RemotePeer(),
	}

	// Route by message type
	p.mu.RLock()
	activationHandler := p.activationHandler
	responseHandler := p.responseHandler
	weightsHandler := p.weightsHandler
	p.mu.RUnlock()

	switch msgType {
	case MsgTypeActivation:
		if activationHandler != nil {
			activationHandler(msg)
		}
	case MsgTypeResponse:
		// Check for pending response channel
		p.mu.RLock()
		ch, ok := p.pendingResponses[requestID]
		p.mu.RUnlock()
		if ok {
			select {
			case ch <- msg:
			default:
				// Channel full, response dropped
			}
		}
		if responseHandler != nil {
			responseHandler(msg)
		}
	case MsgTypeWeights:
		if weightsHandler != nil {
			// Extract chunk info from data header (chunkIndex:4B + totalChunks:4B + actualData)
			if len(data) >= 8 {
				chunkIndex := int(binary.BigEndian.Uint32(data[0:4]))
				totalChunks := int(binary.BigEndian.Uint32(data[4:8]))
				weightsHandler(layerID, chunkIndex, totalChunks, data[8:])
			}
		}
	case MsgTypeWeightsAck:
		// Signal weights ack
		key := fmt.Sprintf("%s:%d", s.Conn().RemotePeer().String(), layerID)
		p.mu.RLock()
		ch, ok := p.pendingWeightsAck[key]
		p.mu.RUnlock()
		if ok {
			select {
			case ch <- struct{}{}:
			default:
			}
		}
	}
}

// handleLegacyMessage processes messages with legacy header format.
func (p *Protocol) handleLegacyMessage(s network.Stream, firstByte []byte) {
	// Read rest of legacy header (15 bytes remaining)
	restHeader := make([]byte, headerSize-1)
	if _, err := io.ReadFull(s, restHeader); err != nil {
		return
	}

	// Reconstruct full header
	header := make([]byte, headerSize)
	header[0] = firstByte[0]
	copy(header[1:], restHeader)

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

	// Invoke legacy handler
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

// SendActivation sends an activation tensor to a remote peer with request ID.
// Automatically registers the request for response tracking.
func (p *Protocol) SendActivation(ctx context.Context, peerID peer.ID, layerID int, seqID uint64, requestID uint64, data []byte) error {
	// Register pending request before sending so response can be caught
	p.RegisterPendingRequest(requestID)
	return p.sendExtendedMessage(ctx, peerID, MsgTypeActivation, layerID, seqID, requestID, data)
}

// SendResponse sends a response tensor back to a peer.
func (p *Protocol) SendResponse(ctx context.Context, peerID peer.ID, layerID int, seqID uint64, requestID uint64, data []byte) error {
	return p.sendExtendedMessage(ctx, peerID, MsgTypeResponse, layerID, seqID, requestID, data)
}

// SendWeights sends layer weights to a remote peer in chunks.
func (p *Protocol) SendWeights(ctx context.Context, peerID peer.ID, layerID int, data []byte) error {
	// Calculate number of chunks
	totalChunks := (len(data) + WeightChunkSize - 1) / WeightChunkSize
	if totalChunks == 0 {
		totalChunks = 1
	}

	// Register pending ack
	key := fmt.Sprintf("%s:%d", peerID.String(), layerID)
	p.mu.Lock()
	p.pendingWeightsAck[key] = make(chan struct{}, 1)
	p.mu.Unlock()

	// Send each chunk
	for i := 0; i < totalChunks; i++ {
		start := i * WeightChunkSize
		end := start + WeightChunkSize
		if end > len(data) {
			end = len(data)
		}
		chunk := data[start:end]

		// Prepend chunk header (chunkIndex:4B + totalChunks:4B)
		chunkData := make([]byte, 8+len(chunk))
		binary.BigEndian.PutUint32(chunkData[0:4], uint32(i))
		binary.BigEndian.PutUint32(chunkData[4:8], uint32(totalChunks))
		copy(chunkData[8:], chunk)

		err := p.sendExtendedMessage(ctx, peerID, MsgTypeWeights, layerID, 0, 0, chunkData)
		if err != nil {
			return fmt.Errorf("failed to send chunk %d/%d: %w", i+1, totalChunks, err)
		}
	}

	return nil
}

// SendWeightsAck sends acknowledgment for received weights.
func (p *Protocol) SendWeightsAck(ctx context.Context, peerID peer.ID, layerID int) error {
	return p.sendExtendedMessage(ctx, peerID, MsgTypeWeightsAck, layerID, 0, 0, nil)
}

// WaitForResponse waits for a response to a specific request ID.
func (p *Protocol) WaitForResponse(ctx context.Context, requestID uint64, timeout time.Duration) (*TensorMessage, error) {
	p.mu.RLock()
	ch, ok := p.pendingResponses[requestID]
	p.mu.RUnlock()

	if !ok {
		// Register if not already registered
		p.RegisterPendingRequest(requestID)
		p.mu.RLock()
		ch = p.pendingResponses[requestID]
		p.mu.RUnlock()
	}

	select {
	case msg := <-ch:
		// Clean up
		p.mu.Lock()
		delete(p.pendingResponses, requestID)
		p.mu.Unlock()
		return msg, nil
	case <-time.After(timeout):
		// Clean up
		p.mu.Lock()
		delete(p.pendingResponses, requestID)
		p.mu.Unlock()
		return nil, ErrResponseTimeout
	case <-ctx.Done():
		// Clean up
		p.mu.Lock()
		delete(p.pendingResponses, requestID)
		p.mu.Unlock()
		return nil, ctx.Err()
	}
}

// WaitForWeightsAck waits for weight transfer acknowledgment.
func (p *Protocol) WaitForWeightsAck(ctx context.Context, peerID peer.ID, layerID int, timeout time.Duration) error {
	key := fmt.Sprintf("%s:%d", peerID.String(), layerID)

	p.mu.RLock()
	ch, ok := p.pendingWeightsAck[key]
	p.mu.RUnlock()

	if !ok {
		return fmt.Errorf("no pending ack for layer %d", layerID)
	}

	select {
	case <-ch:
		// Clean up
		p.mu.Lock()
		delete(p.pendingWeightsAck, key)
		p.mu.Unlock()
		return nil
	case <-time.After(timeout):
		// Clean up
		p.mu.Lock()
		delete(p.pendingWeightsAck, key)
		p.mu.Unlock()
		return fmt.Errorf("timeout waiting for weights ack for layer %d", layerID)
	case <-ctx.Done():
		// Clean up
		p.mu.Lock()
		delete(p.pendingWeightsAck, key)
		p.mu.Unlock()
		return ctx.Err()
	}
}

// sendExtendedMessage sends a message with extended header format.
func (p *Protocol) sendExtendedMessage(ctx context.Context, peerID peer.ID, msgType uint8, layerID int, seqID uint64, requestID uint64, data []byte) error {
	// Open stream to peer
	s, err := p.host.NewStream(ctx, peerID, protocol.ID(TensorProtocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream: %w", err)
	}
	defer s.Close()

	// Write extended header
	header := make([]byte, ExtendedHeaderSize)
	EncodeExtendedHeader(header, msgType, layerID, seqID, requestID, len(data))

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

// Close shuts down the protocol handler.
func (p *Protocol) Close() error {
	p.host.RemoveStreamHandler(protocol.ID(TensorProtocolID))
	return nil
}

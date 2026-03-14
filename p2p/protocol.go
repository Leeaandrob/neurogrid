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
	"github.com/neurogrid/engine/pkg/telemetry"
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

	// TracedHeaderSize is the size of the header with trace context for distributed tracing.
	// Format: ExtendedHeader (25 bytes) + TraceID (16 bytes) + SpanID (8 bytes) + TraceFlags (1 byte)
	TracedHeaderSize = ExtendedHeaderSize + telemetry.TraceContextSize

	// Message types for the tensor protocol
	MsgTypeActivation       = 0x01 // Forward activation to remote peer
	MsgTypeResponse         = 0x02 // Return computed hidden state
	MsgTypeWeights          = 0x03 // Transfer layer weights
	MsgTypeWeightsAck       = 0x04 // Acknowledge weights received
	MsgTypeLayerStatus      = 0x05 // Worker reports which layers it has locally
	MsgTypeLayerRequest     = 0x06 // Coordinator requests specific layers to be loaded
	MsgTypeModelConfig      = 0x07 // Transfer model configuration to stateless worker
	MsgTypeGPUInfo          = 0x08 // Worker reports GPU info (VRAM) to coordinator
	MsgTypeGPUInfoRequest   = 0x09 // Coordinator requests GPU info from worker
	MsgTypeTracedActivation = 0x11 // Forward activation with trace context
	MsgTypeTracedResponse   = 0x12 // Return computed hidden state with trace context

	// WeightChunkSize is the maximum size of a weight chunk (1MB)
	WeightChunkSize = 1024 * 1024
)

// ErrResponseTimeout is returned when waiting for a response times out.
var ErrResponseTimeout = fmt.Errorf("response timeout")

// TensorMessage represents a tensor being transferred over the network.
type TensorMessage struct {
	MsgType      uint8
	LayerID      int
	SeqID        uint64
	RequestID    uint64
	Data         []byte
	From         peer.ID
	TraceContext telemetry.TraceContext // Trace context for distributed tracing
}

// TensorHandler is a callback for received tensors.
type TensorHandler func(msg *TensorMessage)

// WeightsHandler is a callback for received weight chunks.
type WeightsHandler func(layerID int, chunkIndex int, totalChunks int, data []byte)

// LayerStatusHandler is a callback for when a peer reports its loaded layers.
type LayerStatusHandler func(peerID peer.ID, loadedLayers []int)

// LayerRequestHandler is a callback for when coordinator requests specific layers.
type LayerRequestHandler func(peerID peer.ID, requestedLayers []int)

// ModelConfigHandler is a callback for received model config.
type ModelConfigHandler func(config []byte, from peer.ID)

// GPUInfo contains GPU information reported by a worker.
type GPUInfo struct {
	TotalVRAM uint64 // Total VRAM in bytes
	UsedVRAM  uint64 // Used VRAM in bytes
	GPUName   string // GPU device name
}

// GPUInfoHandler is a callback for when a peer reports its GPU info.
type GPUInfoHandler func(peerID peer.ID, info GPUInfo)

// GPUInfoRequestHandler is a callback for when coordinator requests GPU info.
type GPUInfoRequestHandler func(peerID peer.ID)

// BufferPool provides reusable byte buffers for transport operations.
// This mirrors the transport.BufferPool interface for protocol layer use.
//
// Implementations should be thread-safe for concurrent Get/Put calls.
// Buffers returned by Get may be larger than requested size.
// Put accepts nil buffers gracefully (no-op).
//
// Note: This interface is intentionally duplicated from transport.BufferPool
// to avoid circular import between p2p and transport packages.
type BufferPool interface {
	// Get returns a buffer of at least the requested size.
	// Returns nil if allocation fails and pool is exhausted.
	Get(size int) []byte

	// Put returns a buffer to the pool for reuse.
	// Safe to call with nil buffer.
	Put(buf []byte)

	// Close releases all pooled resources.
	Close() error
}

// Protocol manages the tensor transfer protocol.
type Protocol struct {
	host                  host.Host
	handler               TensorHandler                  // Legacy handler
	activationHandler     TensorHandler                  // Handler for activation messages
	responseHandler       TensorHandler                  // Handler for response messages
	weightsHandler        WeightsHandler                 // Handler for weight chunks
	layerStatusHandler    LayerStatusHandler             // Handler for layer status reports
	layerRequestHandler   LayerRequestHandler            // Handler for layer requests
	modelConfigHandler    ModelConfigHandler             // Handler for model config messages
	gpuInfoHandler        GPUInfoHandler                 // Handler for GPU info reports
	gpuInfoRequestHandler GPUInfoRequestHandler          // Handler for GPU info requests
	pendingResponses      map[uint64]chan *TensorMessage // RequestID -> response channel
	pendingWeightsAck     map[string]chan struct{}       // peerID:layerID -> ack channel
	pendingLayerStatus    map[string]chan []int          // peerID -> layer status channel
	pendingGPUInfo        map[string]chan GPUInfo        // peerID -> GPU info channel
	bufferPool            BufferPool                     // Optional buffer pool for zero-allocation message handling
	mu                    sync.RWMutex
}

// NewProtocol creates a new tensor transfer protocol handler.
func NewProtocol(h host.Host) *Protocol {
	p := &Protocol{
		host:               h,
		pendingResponses:   make(map[uint64]chan *TensorMessage),
		pendingWeightsAck:  make(map[string]chan struct{}),
		pendingLayerStatus: make(map[string]chan []int),
		pendingGPUInfo:     make(map[string]chan GPUInfo),
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

// OnLayerStatusReceived sets the callback for layer status messages from workers.
func (p *Protocol) OnLayerStatusReceived(handler LayerStatusHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.layerStatusHandler = handler
}

// OnLayerRequestReceived sets the callback for layer request messages from coordinator.
func (p *Protocol) OnLayerRequestReceived(handler LayerRequestHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.layerRequestHandler = handler
}

// OnModelConfigReceived sets the callback for model config messages.
func (p *Protocol) OnModelConfigReceived(handler ModelConfigHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.modelConfigHandler = handler
}

// OnGPUInfoReceived sets the callback for GPU info messages from workers.
func (p *Protocol) OnGPUInfoReceived(handler GPUInfoHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.gpuInfoHandler = handler
}

// OnGPUInfoRequestReceived sets the callback for GPU info request messages from coordinator.
func (p *Protocol) OnGPUInfoRequestReceived(handler GPUInfoRequestHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.gpuInfoRequestHandler = handler
}

// RegisterPendingRequest creates a pending response channel for a request ID.
func (p *Protocol) RegisterPendingRequest(requestID uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pendingResponses[requestID] = make(chan *TensorMessage, 1)
}

// SetBufferPool sets the buffer pool for zero-allocation message handling.
// When set, incoming messages will use pooled buffers instead of allocating new ones.
//
// The pool should have buffers sized for typical activation data (8KB-16KB for LLM inference).
// For CUDA-optimized transfers, use a pinned memory pool implementation.
//
// Thread-safe: can be called while protocol is handling messages.
func (p *Protocol) SetBufferPool(pool BufferPool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.bufferPool = pool
}

// GetBufferPool returns the configured buffer pool, or nil if none is set.
// Thread-safe.
func (p *Protocol) GetBufferPool() BufferPool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.bufferPool
}

// HasBufferPool returns whether the protocol has a buffer pool configured.
// Thread-safe.
func (p *Protocol) HasBufferPool() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.bufferPool != nil
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
	if msgType == MsgTypeTracedActivation || msgType == MsgTypeTracedResponse {
		// Traced message with trace context
		p.handleTracedMessage(s, msgType)
	} else if msgType >= MsgTypeActivation && msgType <= MsgTypeGPUInfoRequest {
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

	// Read data - use buffer pool if available
	p.mu.RLock()
	pool := p.bufferPool
	p.mu.RUnlock()

	var data []byte
	var pooledBuf []byte
	if pool != nil {
		pooledBuf = pool.Get(int(dataLen))
		data = pooledBuf[:dataLen]
	} else {
		data = make([]byte, dataLen)
	}
	if _, err := io.ReadFull(s, data); err != nil {
		if pooledBuf != nil && pool != nil {
			pool.Put(pooledBuf)
		}
		return
	}

	// Make a copy of data if using pooled buffer
	var msgData []byte
	if pooledBuf != nil {
		msgData = make([]byte, len(data))
		copy(msgData, data)
		pool.Put(pooledBuf)
	} else {
		msgData = data
	}

	// Create message
	msg := &TensorMessage{
		MsgType:   msgType,
		LayerID:   layerID,
		SeqID:     seqID,
		RequestID: requestID,
		Data:      msgData,
		From:      s.Conn().RemotePeer(),
	}

	// Route by message type
	p.mu.RLock()
	activationHandler := p.activationHandler
	responseHandler := p.responseHandler
	weightsHandler := p.weightsHandler
	layerStatusHandler := p.layerStatusHandler
	layerRequestHandler := p.layerRequestHandler
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
	case MsgTypeLayerStatus:
		// Worker reporting which layers it has locally
		// Data format: count (4B) + layerIDs (4B each)
		peerID := s.Conn().RemotePeer()
		layers := decodeLayerList(data)

		// Check for pending layer status channel (coordinator waiting)
		p.mu.RLock()
		ch, ok := p.pendingLayerStatus[peerID.String()]
		p.mu.RUnlock()
		if ok {
			select {
			case ch <- layers:
			default:
			}
		}

		if layerStatusHandler != nil {
			layerStatusHandler(peerID, layers)
		}
	case MsgTypeLayerRequest:
		// Coordinator requesting worker to load specific layers
		// Data format: count (4B) + layerIDs (4B each)
		peerID := s.Conn().RemotePeer()
		layers := decodeLayerList(data)
		if layerRequestHandler != nil {
			layerRequestHandler(peerID, layers)
		}
	case MsgTypeModelConfig:
		// Coordinator sending model config to stateless worker
		p.mu.RLock()
		modelConfigHandler := p.modelConfigHandler
		p.mu.RUnlock()
		if modelConfigHandler != nil {
			modelConfigHandler(data, s.Conn().RemotePeer())
		}
	case MsgTypeGPUInfo:
		// Worker reporting GPU info to coordinator
		// Data format: TotalVRAM (8B) + UsedVRAM (8B) + GPUNameLen (4B) + GPUName (variable)
		peerID := s.Conn().RemotePeer()
		gpuInfo := decodeGPUInfo(data)

		// Check for pending GPU info channel (coordinator waiting)
		p.mu.RLock()
		ch, ok := p.pendingGPUInfo[peerID.String()]
		gpuInfoHandler := p.gpuInfoHandler
		p.mu.RUnlock()
		if ok {
			select {
			case ch <- gpuInfo:
			default:
			}
		}

		if gpuInfoHandler != nil {
			gpuInfoHandler(peerID, gpuInfo)
		}
	case MsgTypeGPUInfoRequest:
		// Coordinator requesting GPU info from worker
		peerID := s.Conn().RemotePeer()
		p.mu.RLock()
		gpuInfoRequestHandler := p.gpuInfoRequestHandler
		p.mu.RUnlock()
		if gpuInfoRequestHandler != nil {
			gpuInfoRequestHandler(peerID)
		}
	}
}

// decodeLayerList decodes a list of layer IDs from binary format.
// Format: count (4B) + layerIDs (4B each)
func decodeLayerList(data []byte) []int {
	if len(data) < 4 {
		return nil
	}
	count := int(binary.BigEndian.Uint32(data[0:4]))
	if len(data) < 4+count*4 {
		return nil
	}
	layers := make([]int, count)
	for i := 0; i < count; i++ {
		layers[i] = int(binary.BigEndian.Uint32(data[4+i*4 : 8+i*4]))
	}
	return layers
}

// encodeLayerList encodes a list of layer IDs to binary format.
// Format: count (4B) + layerIDs (4B each)
func encodeLayerList(layers []int) []byte {
	data := make([]byte, 4+len(layers)*4)
	binary.BigEndian.PutUint32(data[0:4], uint32(len(layers)))
	for i, layerID := range layers {
		binary.BigEndian.PutUint32(data[4+i*4:8+i*4], uint32(layerID))
	}
	return data
}

// decodeGPUInfo decodes GPU info from binary format.
// Format: TotalVRAM (8B) + UsedVRAM (8B) + GPUNameLen (4B) + GPUName (variable)
func decodeGPUInfo(data []byte) GPUInfo {
	if len(data) < 20 {
		return GPUInfo{}
	}
	totalVRAM := binary.BigEndian.Uint64(data[0:8])
	usedVRAM := binary.BigEndian.Uint64(data[8:16])
	nameLen := int(binary.BigEndian.Uint32(data[16:20]))
	gpuName := ""
	if len(data) >= 20+nameLen {
		gpuName = string(data[20 : 20+nameLen])
	}
	return GPUInfo{
		TotalVRAM: totalVRAM,
		UsedVRAM:  usedVRAM,
		GPUName:   gpuName,
	}
}

// encodeGPUInfo encodes GPU info to binary format.
// Format: TotalVRAM (8B) + UsedVRAM (8B) + GPUNameLen (4B) + GPUName (variable)
func encodeGPUInfo(info GPUInfo) []byte {
	nameBytes := []byte(info.GPUName)
	data := make([]byte, 20+len(nameBytes))
	binary.BigEndian.PutUint64(data[0:8], info.TotalVRAM)
	binary.BigEndian.PutUint64(data[8:16], info.UsedVRAM)
	binary.BigEndian.PutUint32(data[16:20], uint32(len(nameBytes)))
	copy(data[20:], nameBytes)
	return data
}

// handleTracedMessage processes messages with trace context for distributed tracing.
func (p *Protocol) handleTracedMessage(s network.Stream, msgType uint8) {
	// Read rest of traced header (TracedHeaderSize - 1 bytes remaining)
	restHeader := make([]byte, TracedHeaderSize-1)
	if _, err := io.ReadFull(s, restHeader); err != nil {
		return
	}

	// Reconstruct full header
	header := make([]byte, TracedHeaderSize)
	header[0] = msgType
	copy(header[1:], restHeader)

	// Decode extended header part
	_, layerID, seqID, requestID, dataLen := DecodeExtendedHeader(header[:ExtendedHeaderSize])

	// Decode trace context
	var traceCtx telemetry.TraceContext
	traceCtx.Deserialize(header[ExtendedHeaderSize:])

	// Read data - use buffer pool if available
	p.mu.RLock()
	pool := p.bufferPool
	p.mu.RUnlock()

	var data []byte
	var pooledBuf []byte
	if pool != nil {
		pooledBuf = pool.Get(int(dataLen))
		data = pooledBuf[:dataLen]
	} else {
		data = make([]byte, dataLen)
	}
	if _, err := io.ReadFull(s, data); err != nil {
		if pooledBuf != nil && pool != nil {
			pool.Put(pooledBuf)
		}
		return
	}

	// Make a copy of data if using pooled buffer
	var msgData []byte
	if pooledBuf != nil {
		msgData = make([]byte, len(data))
		copy(msgData, data)
		pool.Put(pooledBuf)
	} else {
		msgData = data
	}

	// Create message with trace context
	msg := &TensorMessage{
		MsgType:      msgType,
		LayerID:      layerID,
		SeqID:        seqID,
		RequestID:    requestID,
		Data:         msgData,
		From:         s.Conn().RemotePeer(),
		TraceContext: traceCtx,
	}

	// Route by message type (same as non-traced but handlers can access trace context)
	p.mu.RLock()
	activationHandler := p.activationHandler
	responseHandler := p.responseHandler
	p.mu.RUnlock()

	switch msgType {
	case MsgTypeTracedActivation:
		if activationHandler != nil {
			activationHandler(msg)
		}
	case MsgTypeTracedResponse:
		// Check for pending response channel
		p.mu.RLock()
		ch, ok := p.pendingResponses[requestID]
		p.mu.RUnlock()
		if ok {
			select {
			case ch <- msg:
			default:
			}
		}
		if responseHandler != nil {
			responseHandler(msg)
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

// SendTracedActivation sends an activation tensor with trace context for distributed tracing.
// This enables end-to-end request tracing across coordinator and workers.
func (p *Protocol) SendTracedActivation(ctx context.Context, peerID peer.ID, layerID int, seqID uint64, requestID uint64, data []byte) error {
	// Register pending request before sending
	p.RegisterPendingRequest(requestID)
	return p.sendTracedMessage(ctx, peerID, MsgTypeTracedActivation, layerID, seqID, requestID, data)
}

// SendTracedResponse sends a response tensor with trace context.
func (p *Protocol) SendTracedResponse(ctx context.Context, peerID peer.ID, layerID int, seqID uint64, requestID uint64, data []byte) error {
	return p.sendTracedMessage(ctx, peerID, MsgTypeTracedResponse, layerID, seqID, requestID, data)
}

// sendTracedMessage sends a message with trace context for distributed tracing.
func (p *Protocol) sendTracedMessage(ctx context.Context, peerID peer.ID, msgType uint8, layerID int, seqID uint64, requestID uint64, data []byte) error {
	// Open stream to peer
	s, err := p.host.NewStream(ctx, peerID, protocol.ID(TensorProtocolID))
	if err != nil {
		return fmt.Errorf("failed to open stream: %w", err)
	}
	defer s.Close()

	// Build traced header: ExtendedHeader (25B) + TraceContext (25B) = 50B
	header := make([]byte, TracedHeaderSize)
	EncodeExtendedHeader(header[:ExtendedHeaderSize], msgType, layerID, seqID, requestID, len(data))

	// Extract and append trace context from context.Context
	traceCtx := telemetry.ExtractTraceContext(ctx)
	copy(header[ExtendedHeaderSize:], traceCtx.Serialize())

	if _, err := s.Write(header); err != nil {
		return fmt.Errorf("failed to write traced header: %w", err)
	}

	// Write data
	if len(data) > 0 {
		if _, err := s.Write(data); err != nil {
			return fmt.Errorf("failed to write data: %w", err)
		}
	}

	return nil
}

// ContextFromMessage creates a context.Context with trace context extracted from a TensorMessage.
// This should be used by workers to continue traces initiated by the coordinator.
func ContextFromMessage(ctx context.Context, msg *TensorMessage) context.Context {
	if msg.TraceContext.IsEmpty() {
		return ctx
	}
	return telemetry.ExtractTraceContextFromBytes(ctx, msg.TraceContext.Serialize())
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

// SendLayerStatus sends the list of locally loaded layers to a peer (coordinator).
// This is sent by the worker to inform which layers it already has cached.
func (p *Protocol) SendLayerStatus(ctx context.Context, peerID peer.ID, loadedLayers []int) error {
	data := encodeLayerList(loadedLayers)
	return p.sendExtendedMessage(ctx, peerID, MsgTypeLayerStatus, 0, 0, 0, data)
}

// SendLayerRequest sends a request for the worker to load specific layers.
// This is sent by the coordinator to tell the worker which layers to load locally.
func (p *Protocol) SendLayerRequest(ctx context.Context, peerID peer.ID, layerIDs []int) error {
	data := encodeLayerList(layerIDs)
	return p.sendExtendedMessage(ctx, peerID, MsgTypeLayerRequest, 0, 0, 0, data)
}

// SendModelConfig sends model configuration to a remote peer.
// Config is sent as a single message (< 1KB), not chunked.
func (p *Protocol) SendModelConfig(ctx context.Context, peerID peer.ID, configData []byte) error {
	return p.sendExtendedMessage(ctx, peerID, MsgTypeModelConfig, 0, 0, 0, configData)
}

// SendGPUInfo sends GPU information (VRAM) to a remote peer.
// This is sent by the worker to inform the coordinator about its GPU capabilities.
func (p *Protocol) SendGPUInfo(ctx context.Context, peerID peer.ID, info GPUInfo) error {
	data := encodeGPUInfo(info)
	return p.sendExtendedMessage(ctx, peerID, MsgTypeGPUInfo, 0, 0, 0, data)
}

// SendGPUInfoRequest sends a request for GPU info to a worker.
// The worker should respond with SendGPUInfo.
func (p *Protocol) SendGPUInfoRequest(ctx context.Context, peerID peer.ID) error {
	return p.sendExtendedMessage(ctx, peerID, MsgTypeGPUInfoRequest, 0, 0, 0, nil)
}

// RequestGPUInfo requests GPU info from a peer and waits for response.
// Returns the GPU information (VRAM, name) from the peer.
func (p *Protocol) RequestGPUInfo(ctx context.Context, peerID peer.ID, timeout time.Duration) (GPUInfo, error) {
	// Register pending response
	key := peerID.String()
	p.mu.Lock()
	p.pendingGPUInfo[key] = make(chan GPUInfo, 1)
	p.mu.Unlock()

	// Send GPU info request
	if err := p.SendGPUInfoRequest(ctx, peerID); err != nil {
		p.mu.Lock()
		delete(p.pendingGPUInfo, key)
		p.mu.Unlock()
		return GPUInfo{}, fmt.Errorf("failed to request GPU info: %w", err)
	}

	// Wait for response
	p.mu.RLock()
	ch := p.pendingGPUInfo[key]
	p.mu.RUnlock()

	select {
	case info := <-ch:
		p.mu.Lock()
		delete(p.pendingGPUInfo, key)
		p.mu.Unlock()
		return info, nil
	case <-time.After(timeout):
		p.mu.Lock()
		delete(p.pendingGPUInfo, key)
		p.mu.Unlock()
		return GPUInfo{}, fmt.Errorf("timeout waiting for GPU info from peer")
	case <-ctx.Done():
		p.mu.Lock()
		delete(p.pendingGPUInfo, key)
		p.mu.Unlock()
		return GPUInfo{}, ctx.Err()
	}
}

// RequestLayerStatus requests layer status from a peer and waits for response.
// Returns the list of layers the peer has locally loaded.
func (p *Protocol) RequestLayerStatus(ctx context.Context, peerID peer.ID, timeout time.Duration) ([]int, error) {
	// Register pending response
	key := peerID.String()
	p.mu.Lock()
	p.pendingLayerStatus[key] = make(chan []int, 1)
	p.mu.Unlock()

	// Send empty layer request to trigger status response
	if err := p.SendLayerRequest(ctx, peerID, []int{}); err != nil {
		p.mu.Lock()
		delete(p.pendingLayerStatus, key)
		p.mu.Unlock()
		return nil, fmt.Errorf("failed to request layer status: %w", err)
	}

	// Wait for response
	p.mu.RLock()
	ch := p.pendingLayerStatus[key]
	p.mu.RUnlock()

	select {
	case layers := <-ch:
		p.mu.Lock()
		delete(p.pendingLayerStatus, key)
		p.mu.Unlock()
		return layers, nil
	case <-time.After(timeout):
		p.mu.Lock()
		delete(p.pendingLayerStatus, key)
		p.mu.Unlock()
		return nil, fmt.Errorf("timeout waiting for layer status from peer")
	case <-ctx.Done():
		p.mu.Lock()
		delete(p.pendingLayerStatus, key)
		p.mu.Unlock()
		return nil, ctx.Err()
	}
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

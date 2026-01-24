// Package telemetry provides OpenTelemetry tracing support for NeuroGrid.
package telemetry

import (
	"context"
	"encoding/binary"

	"go.opentelemetry.io/otel/trace"
)

const (
	// TraceContextSize is the size of the serialized trace context in bytes.
	// TraceID (16 bytes) + SpanID (8 bytes) + TraceFlags (1 byte) = 25 bytes
	TraceContextSize = 25
)

// TraceContext represents a serializable trace context for P2P propagation.
// This follows the W3C Trace Context format.
type TraceContext struct {
	TraceID    [16]byte
	SpanID     [8]byte
	TraceFlags uint8
}

// ExtractTraceContext extracts the trace context from a context.Context.
// Returns an empty TraceContext if no span is present.
func ExtractTraceContext(ctx context.Context) TraceContext {
	span := trace.SpanFromContext(ctx)
	if span == nil || !span.SpanContext().IsValid() {
		return TraceContext{}
	}

	sc := span.SpanContext()
	return TraceContext{
		TraceID:    sc.TraceID(),
		SpanID:     sc.SpanID(),
		TraceFlags: byte(sc.TraceFlags()),
	}
}

// InjectTraceContext serializes a trace context to bytes for P2P transmission.
func InjectTraceContext(ctx context.Context) []byte {
	tc := ExtractTraceContext(ctx)
	return tc.Serialize()
}

// ExtractTraceContextFromBytes deserializes trace context from bytes.
// Returns a new context with the span context attached if valid.
func ExtractTraceContextFromBytes(ctx context.Context, data []byte) context.Context {
	if len(data) < TraceContextSize {
		return ctx
	}

	tc := TraceContext{}
	tc.Deserialize(data)

	if tc.IsEmpty() {
		return ctx
	}

	// Create a remote span context from the extracted trace context
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    tc.TraceID,
		SpanID:     tc.SpanID,
		TraceFlags: trace.TraceFlags(tc.TraceFlags),
		Remote:     true, // Mark as remote span
	})

	if !sc.IsValid() {
		return ctx
	}

	return trace.ContextWithRemoteSpanContext(ctx, sc)
}

// Serialize converts the trace context to bytes.
func (tc *TraceContext) Serialize() []byte {
	buf := make([]byte, TraceContextSize)
	copy(buf[0:16], tc.TraceID[:])
	copy(buf[16:24], tc.SpanID[:])
	buf[24] = tc.TraceFlags
	return buf
}

// Deserialize populates the trace context from bytes.
func (tc *TraceContext) Deserialize(data []byte) {
	if len(data) < TraceContextSize {
		return
	}
	copy(tc.TraceID[:], data[0:16])
	copy(tc.SpanID[:], data[16:24])
	tc.TraceFlags = data[24]
}

// IsEmpty returns true if the trace context has no valid trace ID.
func (tc *TraceContext) IsEmpty() bool {
	return tc.TraceID == [16]byte{}
}

// ExtendedMessageHeader represents a P2P message header with trace context.
// Original P2P header (25 bytes) + Trace context (25 bytes) = 50 bytes.
type ExtendedMessageHeader struct {
	// Original fields (25 bytes)
	MsgType   uint8
	LayerID   uint32
	SeqID     uint64
	RequestID uint64
	DataLen   uint32

	// Trace context (25 bytes)
	TraceContext TraceContext
}

const (
	// OriginalHeaderSize is the size of the original P2P header.
	OriginalHeaderSize = 25

	// ExtendedHeaderSize is the size of the extended header with trace context.
	ExtendedHeaderSize = OriginalHeaderSize + TraceContextSize
)

// Serialize converts the extended header to bytes.
func (h *ExtendedMessageHeader) Serialize() []byte {
	buf := make([]byte, ExtendedHeaderSize)

	buf[0] = h.MsgType
	binary.BigEndian.PutUint32(buf[1:5], h.LayerID)
	binary.BigEndian.PutUint64(buf[5:13], h.SeqID)
	binary.BigEndian.PutUint64(buf[13:21], h.RequestID)
	binary.BigEndian.PutUint32(buf[21:25], h.DataLen)

	// Append trace context
	copy(buf[25:50], h.TraceContext.Serialize())

	return buf
}

// DeserializeExtendedHeader populates the header from bytes.
func DeserializeExtendedHeader(data []byte) *ExtendedMessageHeader {
	if len(data) < ExtendedHeaderSize {
		return nil
	}

	h := &ExtendedMessageHeader{
		MsgType:   data[0],
		LayerID:   binary.BigEndian.Uint32(data[1:5]),
		SeqID:     binary.BigEndian.Uint64(data[5:13]),
		RequestID: binary.BigEndian.Uint64(data[13:21]),
		DataLen:   binary.BigEndian.Uint32(data[21:25]),
	}
	h.TraceContext.Deserialize(data[25:50])

	return h
}

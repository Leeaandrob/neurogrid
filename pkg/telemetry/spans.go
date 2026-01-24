// Package telemetry provides OpenTelemetry tracing support for NeuroGrid.
package telemetry

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Span attribute keys
const (
	AttrRequestID   = "neurogrid.request_id"
	AttrModel       = "neurogrid.model"
	AttrMaxTokens   = "neurogrid.max_tokens"
	AttrSeqLen      = "neurogrid.seq_len"
	AttrNumLayers   = "neurogrid.num_layers"
	AttrLayerID     = "neurogrid.layer_id"
	AttrLocation    = "neurogrid.location"
	AttrPeerID      = "neurogrid.peer_id"
	AttrOperation   = "neurogrid.operation"
	AttrTokensGen   = "neurogrid.tokens_generated"
	AttrStopReason  = "neurogrid.stop_reason"
	AttrPosition    = "neurogrid.position"
	AttrTensorSize  = "neurogrid.tensor_size_bytes"
)

// StartInferenceSpan starts a span for the entire inference request.
// This is the root span for a generation request.
func StartInferenceSpan(ctx context.Context, requestID, model string, maxTokens int) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.generate",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String(AttrRequestID, requestID),
			attribute.String(AttrModel, model),
			attribute.Int(AttrMaxTokens, maxTokens),
		),
	)
}

// StartPrefillSpan starts a span for the prefill phase.
func StartPrefillSpan(ctx context.Context, seqLen, numLayers int) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.prefill",
		trace.WithAttributes(
			attribute.Int(AttrSeqLen, seqLen),
			attribute.Int(AttrNumLayers, numLayers),
		),
	)
}

// StartDecodeSpan starts a span for the decode/generation phase.
func StartDecodeSpan(ctx context.Context, maxTokens int) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.decode",
		trace.WithAttributes(
			attribute.Int(AttrMaxTokens, maxTokens),
		),
	)
}

// StartLayerSpan starts a span for a single layer forward pass.
func StartLayerSpan(ctx context.Context, layerID int, location string) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.layer",
		trace.WithAttributes(
			attribute.Int(AttrLayerID, layerID),
			attribute.String(AttrLocation, location),
		),
	)
}

// StartP2PSpan starts a span for a P2P operation.
func StartP2PSpan(ctx context.Context, operation, peerID string, layerID int) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "p2p."+operation,
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			attribute.String(AttrOperation, operation),
			attribute.String(AttrPeerID, peerID),
			attribute.Int(AttrLayerID, layerID),
		),
	)
}

// StartP2PReceiveSpan starts a span for receiving a P2P message.
func StartP2PReceiveSpan(ctx context.Context, operation, peerID string, layerID int) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "p2p."+operation,
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String(AttrOperation, operation),
			attribute.String(AttrPeerID, peerID),
			attribute.Int(AttrLayerID, layerID),
		),
	)
}

// StartTokenEmbedSpan starts a span for token embedding lookup.
func StartTokenEmbedSpan(ctx context.Context) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.embed_token")
}

// StartLMHeadSpan starts a span for LM head computation.
func StartLMHeadSpan(ctx context.Context) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.lm_head")
}

// StartSampleSpan starts a span for token sampling.
func StartSampleSpan(ctx context.Context) (context.Context, trace.Span) {
	return Tracer().Start(ctx, "inference.sample")
}

// RecordError records an error on the current span.
func RecordError(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
}

// SetInferenceResult sets the result attributes on an inference span.
func SetInferenceResult(span trace.Span, tokensGenerated int, stopReason string) {
	span.SetAttributes(
		attribute.Int(AttrTokensGen, tokensGenerated),
		attribute.String(AttrStopReason, stopReason),
	)
}

// SetLayerResult sets result attributes on a layer span.
func SetLayerResult(span trace.Span, position int, tensorSizeBytes int) {
	span.SetAttributes(
		attribute.Int(AttrPosition, position),
		attribute.Int(AttrTensorSize, tensorSizeBytes),
	)
}

// SpanFromContext extracts the current span from context.
// Returns a no-op span if no span is present.
func SpanFromContext(ctx context.Context) trace.Span {
	return trace.SpanFromContext(ctx)
}

// ContextWithSpan returns a new context with the given span attached.
func ContextWithSpan(ctx context.Context, span trace.Span) context.Context {
	return trace.ContextWithSpan(ctx, span)
}

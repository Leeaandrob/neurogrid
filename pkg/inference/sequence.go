// Package inference provides sequence lifecycle management for continuous batching.
package inference

import (
	"context"
	"sync"
)

// SequenceState represents the lifecycle state of a generation sequence.
type SequenceState int

const (
	SeqWaiting  SequenceState = iota // Queued, not yet started
	SeqPrefill                       // Running prefill phase
	SeqDecode                        // In autoregressive decode loop
	SeqFinished                      // Done (EOS, max_tokens, or error)
)

// Sequence represents a single generation request being processed.
type Sequence struct {
	ID           uint64
	State        SequenceState
	InputTokens  []int
	OutputTokens []int
	Position     int // Current decode position (len(InputTokens) + decode steps)
	MaxTokens    int
	Temperature  float32
	TopP         float32
	StopTokens   []int
	StopStrings  []string

	// Hidden state from prefill (input to first decode step)
	Hidden []byte

	// Result channels
	ResultCh chan *GenerateResponse
	ErrCh    chan error

	// Context for cancellation
	Ctx    context.Context
	Cancel context.CancelFunc

	// Per-sequence conv state snapshots (for context switching)
	ConvStates map[int][]byte // layerID → FP32 conv state bytes

	mu sync.Mutex
}

// NewSequence creates a sequence from a generate request.
func NewSequence(id uint64, req *GenerateRequest, ctx context.Context) *Sequence {
	seqCtx, cancel := context.WithCancel(ctx)
	return &Sequence{
		ID:           id,
		State:        SeqWaiting,
		InputTokens:  nil, // Set after tokenization
		OutputTokens: make([]int, 0, req.MaxTokens),
		MaxTokens:    req.MaxTokens,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
		StopTokens:   req.StopTokens,
		ResultCh:     make(chan *GenerateResponse, 1),
		ErrCh:        make(chan error, 1),
		Ctx:          seqCtx,
		Cancel:       cancel,
		ConvStates:   make(map[int][]byte),
	}
}

// IsActive returns true if the sequence is still being processed.
func (s *Sequence) IsActive() bool {
	return s.State == SeqPrefill || s.State == SeqDecode
}

// IsDone returns true if the sequence has finished.
func (s *Sequence) IsDone() bool {
	return s.State == SeqFinished
}

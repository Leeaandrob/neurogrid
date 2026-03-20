// Package inference provides continuous batching via BatchScheduler.
//
// Instead of processing one request to completion (exclusive mutex),
// the scheduler interleaves prefill and decode across multiple requests.
// Each iteration: prefill new requests, decode one step for active ones.
package inference

import (
	"context"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/neurogrid/engine/gpu/bindings"
)

// BatchScheduler manages concurrent inference requests.
// It runs a continuous loop: admit → prefill → decode → sample → cleanup.
type BatchScheduler struct {
	engine       *Engine
	maxActive    int // Max concurrent sequences
	addCh        chan *Sequence
	active       map[uint64]*Sequence
	seqCounter   uint64
	mu           sync.Mutex
	running      bool
	stopCh       chan struct{}
}

// NewBatchScheduler creates a scheduler with the given max concurrency.
func NewBatchScheduler(engine *Engine, maxActive int) *BatchScheduler {
	if maxActive <= 0 {
		maxActive = 8
	}
	return &BatchScheduler{
		engine:    engine,
		maxActive: maxActive,
		addCh:     make(chan *Sequence, 64),
		active:    make(map[uint64]*Sequence),
		stopCh:    make(chan struct{}),
	}
}

// Submit adds a request to the scheduler queue. Non-blocking.
func (bs *BatchScheduler) Submit(ctx context.Context, req *GenerateRequest) *Sequence {
	id := atomic.AddUint64(&bs.seqCounter, 1)
	seq := NewSequence(id, req, ctx)

	// Tokenize prompt
	tokens, err := bs.engine.tokenizer.Encode(req.Prompt)
	if err != nil {
		seq.ErrCh <- err
		return seq
	}
	seq.InputTokens = tokens
	seq.StopStrings = getModelStopStrings(bs.engine.config.ModelType)

	bs.addCh <- seq
	return seq
}

// getModelStopStrings returns stop strings for the current model.
func getModelStopStrings(modelName string) []string {
	if strings.Contains(strings.ToLower(modelName), "lfm") {
		return []string{"<|im_end|>"}
	}
	return nil
}

// Start begins the scheduler loop in a goroutine.
func (bs *BatchScheduler) Start() {
	bs.mu.Lock()
	if bs.running {
		bs.mu.Unlock()
		return
	}
	bs.running = true
	bs.mu.Unlock()

	go bs.loop()
	log.Printf("[Scheduler] Started (max_active=%d)", bs.maxActive)
}

// Stop halts the scheduler loop.
func (bs *BatchScheduler) Stop() {
	close(bs.stopCh)
}

// loop is the main scheduler loop.
func (bs *BatchScheduler) loop() {
	for {
		select {
		case <-bs.stopCh:
			return
		default:
		}

		// 1. Admit new requests
		bs.admitRequests()

		// 2. If no active sequences, wait for new requests
		if len(bs.active) == 0 {
			select {
			case seq := <-bs.addCh:
				bs.startSequence(seq)
			case <-bs.stopCh:
				return
			}
			continue
		}

		// 3. Process one decode step for each active sequence
		bs.decodeStep()

		// 4. Cleanup finished sequences
		bs.cleanupFinished()
	}
}

// admitRequests drains the add channel and starts prefill for new sequences.
func (bs *BatchScheduler) admitRequests() {
	for {
		select {
		case seq := <-bs.addCh:
			if len(bs.active) >= bs.maxActive {
				// Queue full — wait
				go func() { bs.addCh <- seq }() // Re-queue
				return
			}
			bs.startSequence(seq)
		default:
			return
		}
	}
}

// startSequence runs prefill for a new sequence and transitions to decode.
// Prefill is SYNCHRONOUS (blocks scheduler loop) to avoid GPU contention.
func (bs *BatchScheduler) startSequence(seq *Sequence) {
	seq.State = SeqPrefill
	bs.active[seq.ID] = seq

	start := time.Now()
	hidden, err := bs.runPrefill(seq)
	if err != nil {
		seq.ErrCh <- err
		seq.State = SeqFinished
		return
	}
	seq.Hidden = hidden
	seq.Position = len(seq.InputTokens)
	seq.State = SeqDecode
	log.Printf("[Scheduler] Seq %d prefilled %d tokens in %v",
		seq.ID, len(seq.InputTokens), time.Since(start).Round(time.Millisecond))
}

// runPrefill executes the prefill phase for a sequence.
func (bs *BatchScheduler) runPrefill(seq *Sequence) ([]byte, error) {
	e := bs.engine

	// Allocate paged KV cache
	if pagedAlloc, ok := e.layerExecutor.(PagedCacheAllocator); ok {
		maxTokens := len(seq.InputTokens) + seq.MaxTokens
		if err := pagedAlloc.AllocateSequence(seq.ID, maxTokens); err != nil {
			return nil, err
		}
	}

	// Reset conv state only (don't FreeAll — other sequences' KV blocks must survive)
	if executor, ok := e.layerExecutor.(*CUDALayerExecutor); ok {
		// Zero conv states for fresh prefill
		for layerID, state := range executor.convStates {
			bindings.ConvStateReset(state, 1, e.config.HiddenSize, e.config.ConvKernelSize)
			_ = layerID
		}
		// Invalidate CUDA graph
		if executor.decodeCtx != nil {
			bindings.DecodeInvalidateGraph(executor.decodeCtx)
		}
	}

	// Run prefill
	hidden, err := e.prefill(seq.Ctx, seq.InputTokens, seq.ID)
	if err != nil {
		return nil, err
	}

	// Save conv state for this sequence
	if saver, ok := e.layerExecutor.(interface{ SaveConvStates() map[int][]byte }); ok {
		seq.ConvStates = saver.SaveConvStates()
	}

	return hidden, nil
}

// decodeStep runs one decode iteration for each active decoding sequence.
func (bs *BatchScheduler) decodeStep() {
	e := bs.engine

	for _, seq := range bs.active {
		if seq.State != SeqDecode {
			continue
		}

		select {
		case <-seq.Ctx.Done():
			seq.State = SeqFinished
			continue
		default:
		}

		// Restore conv state for this sequence
		if restorer, ok := e.layerExecutor.(interface{ RestoreConvStates(map[int][]byte) }); ok {
			restorer.RestoreConvStates(seq.ConvStates)
		}

		// Invalidate CUDA graph before each sequence switch
		// (graph captured for one sequence can't be reused for another)
		if executor, ok := e.layerExecutor.(*CUDALayerExecutor); ok && executor.decodeCtx != nil {
			bindings.DecodeInvalidateGraph(executor.decodeCtx)
		}

		// Run one decode step
		token, err := bs.runDecodeStep(seq)
		if err != nil {
			seq.ErrCh <- err
			seq.State = SeqFinished
			continue
		}

		// Save conv state after step
		if saver, ok := e.layerExecutor.(interface{ SaveConvStates() map[int][]byte }); ok {
			seq.ConvStates = saver.SaveConvStates()
		}

		seq.OutputTokens = append(seq.OutputTokens, token)
		seq.Position++

		// Check stopping conditions
		if bs.shouldStop(seq, token) {
			bs.finishSequence(seq)
		}
	}
}

// runDecodeStep processes one token for a sequence.
func (bs *BatchScheduler) runDecodeStep(seq *Sequence) (int, error) {
	e := bs.engine

	// Set hidden state
	gpuDecoder, ok := e.layerExecutor.(GPUResidentDecoder)
	if !ok {
		return 0, nil
	}

	if len(seq.OutputTokens) == 0 {
		// First decode step: set hidden from prefill output
		if err := gpuDecoder.SetHiddenGPU(seq.Hidden); err != nil {
			return 0, err
		}
	}

	// Apply LM head for first token (prefill output already processed through layers)
	if len(seq.OutputTokens) == 0 {
		logits, err := e.applyLMHead(seq.Hidden)
		if err != nil {
			return 0, err
		}
		return e.sampler.Sample(logits, seq.Temperature, seq.TopP), nil
	}

	// For subsequent tokens: embed → layers → LM head
	prevToken := seq.OutputTokens[len(seq.OutputTokens)-1]

	// Embed previous token
	if embedLookup, ok2 := e.gpuInference.(GPUEmbeddingLookup); ok2 {
		embPtr, err := embedLookup.EmbedTokenGPUPtr(prevToken)
		if err != nil {
			return 0, err
		}
		if err := gpuDecoder.SetHiddenFromGPU(embPtr); err != nil {
			return 0, err
		}
	} else {
		hidden, err := e.embedToken(prevToken)
		if err != nil {
			return 0, err
		}
		if err := gpuDecoder.SetHiddenGPU(hidden); err != nil {
			return 0, err
		}
	}

	// Append token to paged cache tracking
	if pagedAlloc, ok2 := e.layerExecutor.(PagedCacheAllocator); ok2 {
		pagedAlloc.AppendToken(seq.ID)

		// Update block table on GPU for THIS sequence
		if mgr := pagedAlloc.GetPagedManager(); mgr != nil {
			blockTable, _, err := mgr.GetBlockTable(seq.ID, 256) // max 256 blocks
			if err == nil {
				executor := e.layerExecutor.(*CUDALayerExecutor)
				tableBytes := uint64(len(blockTable) * 4)
				bindings.CopyToDeviceRaw(executor.blockTableGPU, getBytePointer(blockTableToBytes(blockTable)), tableBytes)
			}
		}
	}

	// Run decode step
	if err := gpuDecoder.DecodeStepGPUResident(seq.Position); err != nil {
		return 0, err
	}

	// Apply LM head from GPU hidden state
	hiddenPtr := gpuDecoder.GetHiddenGPUPtr()
	if lmForwarder, ok2 := e.gpuInference.(GPULMHeadForwarder); ok2 {
		logits, err := lmForwarder.ApplyLMHeadFromGPU(hiddenPtr)
		if err != nil {
			return 0, err
		}
		return e.sampler.Sample(logits, seq.Temperature, seq.TopP), nil
	}

	return 0, nil
}

// shouldStop checks if a sequence should stop generating.
func (bs *BatchScheduler) shouldStop(seq *Sequence, token int) bool {
	// Max tokens
	if len(seq.OutputTokens) >= seq.MaxTokens {
		return true
	}

	// EOS token
	if token == bs.engine.tokenizer.EOSToken() {
		return true
	}

	// Stop tokens
	for _, stop := range seq.StopTokens {
		if token == stop {
			return true
		}
	}

	// Stop strings
	if len(seq.StopStrings) > 0 {
		text, _ := bs.engine.tokenizer.Decode(seq.OutputTokens)
		for _, stopStr := range seq.StopStrings {
			if strings.Contains(text, stopStr) {
				return true
			}
		}
	}

	return false
}

// finishSequence completes a sequence and sends the result.
func (bs *BatchScheduler) finishSequence(seq *Sequence) {
	seq.State = SeqFinished

	// Decode output
	text, _ := bs.engine.tokenizer.Decode(seq.OutputTokens)

	// Clean model output (strip thinking tokens etc.)
	// TODO: move cleanModelOutput from server.go to a shared location

	stopReason := "max_tokens"
	if len(seq.OutputTokens) < seq.MaxTokens {
		stopReason = "eos"
	}

	seq.ResultCh <- &GenerateResponse{
		Text:       text,
		TokenCount: len(seq.OutputTokens),
		StopReason: stopReason,
	}
}

// cleanupFinished removes finished sequences and frees their resources.
func (bs *BatchScheduler) cleanupFinished() {
	for id, seq := range bs.active {
		if seq.State == SeqFinished {
			// Free paged cache blocks
			if pagedAlloc, ok := bs.engine.layerExecutor.(PagedCacheAllocator); ok {
				pagedAlloc.FreeSequence(seq.ID)
			}
			delete(bs.active, id)
		}
	}
}

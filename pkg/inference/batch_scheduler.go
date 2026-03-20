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
	"unsafe"

	"github.com/neurogrid/engine/gpu/bindings"
)

// BatchScheduler manages concurrent inference requests.
// It runs a continuous loop: admit → prefill → decode → sample → cleanup.
type BatchScheduler struct {
	engine       *Engine
	maxActive    int // Max concurrent sequences
	addCh        chan *Sequence
	active       map[uint64]*Sequence
	waiting      []*Sequence // Buffered requests waiting for prefill
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

		// 3. Process decode step (batched when multiple sequences, round-robin for single)
		activeCount := 0
		for _, seq := range bs.active {
			if seq.State == SeqDecode {
				activeCount++
			}
		}
		if activeCount > 1 {
			bs.decodeStepBatched() // Phase 2: N sequences in one CUDA call
		} else {
			bs.decodeStep() // Phase 1: single sequence
		}

		// 4. Cleanup finished sequences
		bs.cleanupFinished()

		// 5. If no more decoding sequences, prefill waiting requests
		hasDecoding := false
		for _, seq := range bs.active {
			if seq.State == SeqDecode {
				hasDecoding = true
				break
			}
		}
		if !hasDecoding {
			bs.mu.Lock()
			pending := bs.waiting
			bs.waiting = nil
			bs.mu.Unlock()
			for _, seq := range pending {
				if len(bs.active) < bs.maxActive {
					bs.startSequence(seq)
				} else {
					bs.mu.Lock()
					bs.waiting = append(bs.waiting, seq)
					bs.mu.Unlock()
				}
			}
		}
	}
}

// admitRequests drains the add channel but only prefills if no sequences are decoding.
// This prevents prefill from corrupting decode state of active sequences.
func (bs *BatchScheduler) admitRequests() {
	// Check if any sequence is actively decoding
	hasDecoding := false
	for _, seq := range bs.active {
		if seq.State == SeqDecode {
			hasDecoding = true
			break
		}
	}

	for {
		select {
		case seq := <-bs.addCh:
			if len(bs.active) >= bs.maxActive || hasDecoding {
				// Can't prefill while decode is active — buffer for later
				bs.mu.Lock()
				bs.waiting = append(bs.waiting, seq)
				bs.mu.Unlock()
				continue
			}
			bs.startSequence(seq)
			hasDecoding = true // After prefill, this seq is decoding
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

// decodeStepBatched processes ALL active sequences in one batched forward pass.
// This is Phase 2: N sequences generate N tokens in one CUDA call.
func (bs *BatchScheduler) decodeStepBatched() {
	e := bs.engine
	executor, ok := e.layerExecutor.(*CUDALayerExecutor)
	if !ok {
		bs.decodeStep() // fallback to round-robin
		return
	}

	// Collect active decoding sequences
	var seqs []*Sequence
	for _, seq := range bs.active {
		if seq.State == SeqDecode && len(seq.OutputTokens) > 0 {
			seqs = append(seqs, seq)
		}
	}

	if len(seqs) == 0 {
		// Handle first-token sequences (i=0 path: LM head only)
		bs.decodeStep()
		return
	}

	N := len(seqs)
	H := e.config.HiddenSize

	// 1. Gather embeddings: embed each sequence's previous token → [N, H]
	dEmbeddings, err := bindings.AllocOnDevice(uint64(N*H*2), 0) // BF16 = 2 bytes
	if err != nil {
		bs.decodeStep()
		return
	}
	defer bindings.FreeOnDevice(dEmbeddings, 0)

	for i, seq := range seqs {
		prevToken := seq.OutputTokens[len(seq.OutputTokens)-1]
		if embedLookup, ok2 := e.gpuInference.(GPUEmbeddingLookup); ok2 {
			embPtr, embErr := embedLookup.EmbedTokenGPUPtr(prevToken)
			if embErr != nil {
				continue
			}
			// Copy FP16 embedding to position i in the batch buffer (D2D)
			off := unsafe.Pointer(uintptr(dEmbeddings) + uintptr(i*H*2))
			bindings.CopyToDeviceRaw(off, embPtr, uint64(H*2))
		}
	}

	// 2. Stack positions, seq_lens, block_tables
	positions := make([]int32, N)
	seqLens := make([]int32, N)
	maxBlocks := executor.maxBlocksPerSeq
	blockTables := make([]int32, N*maxBlocks)

	for i, seq := range seqs {
		positions[i] = int32(seq.Position)
		seqLens[i] = int32(seq.Position + 1)

		// Get block table for this sequence
		if mgr := executor.pagedManager; mgr != nil {
			bt, _, btErr := mgr.GetBlockTable(seq.ID, maxBlocks)
			if btErr == nil {
				copy(blockTables[i*maxBlocks:], bt)
			}
		}

		// Append token to paged cache tracking
		if pagedAlloc, ok2 := e.layerExecutor.(PagedCacheAllocator); ok2 {
			pagedAlloc.AppendToken(seq.ID)
		}
	}

	// Upload to GPU
	dPositions, _ := bindings.AllocOnDevice(uint64(N*4), 0)
	defer bindings.FreeOnDevice(dPositions, 0)
	bindings.CopyToDeviceRaw(dPositions, unsafe.Pointer(&positions[0]), uint64(N*4))

	dSeqLens, _ := bindings.AllocOnDevice(uint64(N*4), 0)
	defer bindings.FreeOnDevice(dSeqLens, 0)
	bindings.CopyToDeviceRaw(dSeqLens, unsafe.Pointer(&seqLens[0]), uint64(N*4))

	dBlockTables, _ := bindings.AllocOnDevice(uint64(len(blockTables)*4), 0)
	defer bindings.FreeOnDevice(dBlockTables, 0)
	bindings.CopyToDeviceRaw(dBlockTables, unsafe.Pointer(&blockTables[0]), uint64(len(blockTables)*4))

	// 3. Output buffer
	dOutput, _ := bindings.AllocOnDevice(uint64(N*H*2), 0) // FP16
	defer bindings.FreeOnDevice(dOutput, 0)

	// 4. Allocate per-sequence conv states on GPU
	// Layout: [num_conv_layers * batch_size] pointers
	// For each conv layer, each sequence has its own state buffer
	numConvLayers := 0
	for layerID := range executor.convStates {
		_ = layerID
		numConvLayers++
	}

	var convStatesGPU unsafe.Pointer
	if numConvLayers > 0 {
		// Allocate per-sequence conv state buffers
		stateSize := e.config.HiddenSize * e.config.ConvKernelSize * 4 // FP32
		convPtrs := make([]unsafe.Pointer, numConvLayers*N)

		convIdx := 0
		for layerID := 0; layerID < e.config.NumLayers; layerID++ {
			if !e.config.IsConvLayer(layerID) {
				continue
			}
			for b := 0; b < N; b++ {
				// Allocate GPU buffer for this sequence's conv state
				ptr, allocErr := bindings.AllocOnDevice(uint64(stateSize), 0)
				if allocErr != nil {
					continue
				}
				// Restore from saved host copy
				if len(seqs[b].ConvStates[layerID]) > 0 {
					bindings.CopyToDeviceRaw(ptr, unsafe.Pointer(&seqs[b].ConvStates[layerID][0]), uint64(stateSize))
				}
				convPtrs[convIdx*N+b] = ptr
			}
			convIdx++
		}

		// Upload pointer array to GPU
		ptrArraySize := uint64(len(convPtrs) * 8) // 8 bytes per pointer
		convStatesGPU, _ = bindings.AllocOnDevice(ptrArraySize, 0)
		bindings.CopyToDeviceRaw(convStatesGPU, unsafe.Pointer(&convPtrs[0]), ptrArraySize)
		defer func() {
			// Free per-sequence conv state buffers
			for _, ptr := range convPtrs {
				if ptr != nil {
					bindings.FreeOnDevice(ptr, 0)
				}
			}
			bindings.FreeOnDevice(convStatesGPU, 0)
		}()
	}

	// 5. Call batched decode with per-sequence conv states
	if err := bindings.DecodeStepBatched(executor.decodeCtx,
		dEmbeddings, dOutput, dPositions, dSeqLens, dBlockTables,
		convStatesGPU,
		N); err != nil {
		log.Printf("[Scheduler] Batched decode failed: %v, falling back to round-robin", err)
		bs.decodeStep()
		return
	}

	// 6. Apply LM head and sample for each sequence
	for i, seq := range seqs {
		// Get hidden state for this sequence from output
		off := unsafe.Pointer(uintptr(dOutput) + uintptr(i*H*2))
		if lmForwarder, ok2 := e.gpuInference.(GPULMHeadForwarder); ok2 {
			logits, lmErr := lmForwarder.ApplyLMHeadFromGPU(off)
			if lmErr != nil {
				continue
			}
			token := e.sampler.Sample(logits, seq.Temperature, seq.TopP)
			seq.OutputTokens = append(seq.OutputTokens, token)
			seq.Position++

			if bs.shouldStop(seq, token) {
				bs.finishSequence(seq)
			}
		}
	}

	// Save conv states back from GPU to each sequence
	if numConvLayers > 0 && convStatesGPU != nil {
		stateSize := e.config.HiddenSize * e.config.ConvKernelSize * 4
		convIdx := 0
		for layerID := 0; layerID < e.config.NumLayers; layerID++ {
			if !e.config.IsConvLayer(layerID) {
				continue
			}
			for b := 0; b < N; b++ {
				// Read pointer from array
				var ptr unsafe.Pointer
				ptrOff := unsafe.Pointer(uintptr(convStatesGPU) + uintptr((convIdx*N+b)*8))
				bindings.CopyFromDeviceRaw(unsafe.Pointer(&ptr), ptrOff, 8)
				if ptr != nil {
					buf := make([]byte, stateSize)
					bindings.CopyFromDeviceRaw(unsafe.Pointer(&buf[0]), ptr, uint64(stateSize))
					seqs[b].ConvStates[layerID] = buf
				}
			}
			convIdx++
		}
	}
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

		// Trace first tokens per sequence
		if len(seq.OutputTokens) <= 3 {
			if decoded, err2 := e.tokenizer.Decode([]int{token}); err2 == nil {
				log.Printf("[Sched] Seq %d step %d: token=%d %q pos=%d",
					seq.ID, len(seq.OutputTokens), token, decoded, seq.Position)
			}
		}

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

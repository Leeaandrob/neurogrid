// Package inference provides distributed inference capabilities for LLM generation.
package inference

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/neurogrid/engine/pkg/metrics"
	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/transport"
	"github.com/neurogrid/engine/pkg/types"
)

// GenerateRequest represents a text generation request.
type GenerateRequest struct {
	Prompt      string
	MaxTokens   int
	Temperature float32
	TopP        float32
	StopTokens  []int
	StopStrings []string // Stop generation when any of these strings appear in output
}

// GenerateResponse represents a text generation response.
type GenerateResponse struct {
	Text       string
	TokenCount int
	StopReason string
}

// StreamToken represents a single token in a streaming response.
type StreamToken struct {
	Token      int
	Text       string
	IsFirst    bool
	IsFinal    bool
	StopReason string
}

// TokenCallback is called for each generated token during streaming.
type TokenCallback func(token StreamToken) error

// Tokenizer interface for encoding/decoding text.
type Tokenizer interface {
	Encode(text string) ([]int, error)
	Decode(tokens []int) (string, error)
	EOSToken() int
	BOSToken() int
	VocabSize() int
}

// LayerExecutor interface for executing a single layer forward pass.
type LayerExecutor interface {
	Forward(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) // returns hidden, k, v
}

// ConvLayerExecutor extends LayerExecutor with conv layer support (LFM2).
type ConvLayerExecutor interface {
	ForwardConv(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, error)
}

// FullDecoder runs all layers in a single call (eliminates per-layer round-trips).
type FullDecoder interface {
	DecodeAll(hidden []byte, position int) ([]byte, error)
}

// GPUResidentDecoder keeps hidden state on GPU between tokens (zero-copy decode).
type GPUResidentDecoder interface {
	SetHiddenGPU(hidden []byte) error
	SetHiddenFromGPU(gpuPtr unsafe.Pointer) error
	DecodeStepGPUResident(position int) error
	GetHiddenGPUPtr() unsafe.Pointer
	GetHiddenGPU(hidden []byte) error
}

// GPULMHeadForwarder applies LM head from a GPU pointer (avoids GPU→Host→GPU copy).
type GPULMHeadForwarder interface {
	ApplyLMHeadFromGPU(hiddenGPUPtr unsafe.Pointer) ([]float32, error)
}

// GPUEmbeddingLookup looks up token embedding directly on GPU.
type GPUEmbeddingLookup interface {
	EmbedTokenGPUPtr(tokenID int) (unsafe.Pointer, error)
}

// PagedCacheAllocator manages paged KV cache block allocation for sequences.
type PagedCacheAllocator interface {
	AllocateSequence(seqID uint64, maxTokens int) error
	AppendToken(seqID uint64) error
	FreeSequence(seqID uint64)
	GetBlockTableGPU(seqID uint64) (unsafe.Pointer, int, error) // returns GPU ptr, seqLen
	GetPagedManager() *PagedKVCacheManager
}

// EngineConfig holds configuration for the inference engine.
type EngineConfig struct {
	ModelConfig      *types.LlamaConfig
	LocalPeerID      string
	EnableHealthCheck bool // Enable tensor health checking (NaN/Inf detection)
}

// GPUInference interface for GPU-accelerated operations.
type GPUInference interface {
	EmbedTokenGPU(token int) ([]byte, error)
	ApplyLMHeadGPU(hidden []byte) ([]float32, error)
}

// Engine orchestrates distributed inference across multiple peers.
type Engine struct {
	config       *types.LlamaConfig
	scheduler    *scheduler.Scheduler
	router       *transport.TransportRouter
	prefetch     *scheduler.PrefetchCoordinator
	tokenizer    Tokenizer
	sampler      *Sampler
	kvCaches     *KVCacheManager
	assignments  []scheduler.LayerAssignment
	localPeerID  string
	specDecoder  *SpeculativeDecoder // Optional: speculative decoding (self-spec mode)
	seqCounter   uint64
	layerExecutor LayerExecutor

	// Remote layer executors for distributed inference
	remoteExecutors map[string]LayerExecutor // Peer ID -> executor for that peer's layers

	// Embedding and output head (stored as raw bytes for CPU fallback)
	embeddings []byte // Token embedding matrix [vocabSize, hiddenSize] in FP16
	lmHead     []byte // Output projection [hiddenSize, vocabSize] in FP16

	// GPU components (nil if running in CPU mode)
	gpuInference GPUInference
	useGPU       bool

	// Health check configuration
	enableHealthCheck bool

	// Prefix caching: reuse KV cache blocks across requests with same prefix
	prefixCache *PrefixCache

	mu sync.RWMutex
}

// NewEngine creates a new distributed inference engine.
func NewEngine(config EngineConfig) *Engine {
	return &Engine{
		config:            config.ModelConfig,
		kvCaches:          NewKVCacheManager(),
		sampler:           NewSampler(42), // Default seed
		localPeerID:       config.LocalPeerID,
		remoteExecutors:   make(map[string]LayerExecutor),
		enableHealthCheck: config.EnableHealthCheck,
		prefixCache:       NewPrefixCache(),
	}
}

// SetHealthCheckEnabled enables or disables tensor health checking.
func (e *Engine) SetHealthCheckEnabled(enabled bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.enableHealthCheck = enabled
}

// RegisterRemoteExecutor registers a remote layer executor for a specific peer.
// The executor will be used for all layers assigned to that peer.
func (e *Engine) RegisterRemoteExecutor(peerID string, exec LayerExecutor) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.remoteExecutors[peerID] = exec
}

// UnregisterRemoteExecutor removes a remote layer executor for a peer.
func (e *Engine) UnregisterRemoteExecutor(peerID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.remoteExecutors, peerID)
}

// SetScheduler sets the layer scheduler.
func (e *Engine) SetScheduler(s *scheduler.Scheduler) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.scheduler = s
}

// SetRouter sets the transport router.
func (e *Engine) SetRouter(r *transport.TransportRouter) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.router = r
}

// SetTokenizer sets the tokenizer.
func (e *Engine) SetTokenizer(t Tokenizer) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.tokenizer = t
}

// SetSampler sets the token sampler with a specific seed.
func (e *Engine) SetSampler(seed int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.sampler = NewSampler(seed)
}

// SetLayerExecutor sets the layer executor for forward passes.
func (e *Engine) SetLayerExecutor(exec LayerExecutor) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.layerExecutor = exec
}

// SetAssignments sets the layer-to-peer assignments.
func (e *Engine) SetAssignments(assignments []scheduler.LayerAssignment) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.assignments = assignments
	e.prefetch = scheduler.NewPrefetchCoordinator(assignments, 16)
	e.prefetch.Start(1) // Start prefetch worker to prevent queue blocking
}

// LoadEmbeddings loads the token embedding matrix.
func (e *Engine) LoadEmbeddings(embeddings []byte) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.embeddings = embeddings
}

// LoadLMHead loads the language model output head.
func (e *Engine) LoadLMHead(lmHead []byte) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.lmHead = lmHead
}

// LoadModel loads a model from the specified path using the model loader.
// It loads embeddings, LM head, and validates weight shapes against config.
// For distributed inference, use LoadModelDistributed instead.
func (e *Engine) LoadModel(ctx context.Context, modelPath string, useMmap bool) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.config == nil {
		return fmt.Errorf("model config not set")
	}

	// Create loader based on strategy
	loader, err := model.NewLoader(modelPath, useMmap)
	if err != nil {
		return fmt.Errorf("failed to create model loader: %w", err)
	}
	defer loader.Close()

	// Validate model layer count matches config
	numLayers := loader.CountLayers()
	if numLayers != e.config.NumLayers {
		log.Printf("[LoadModel] Warning: model has %d layers, config expects %d", numLayers, e.config.NumLayers)
	}

	// Load embeddings
	embData, embInfo, err := loader.LoadTensor("model.embed_tokens.weight")
	if err != nil {
		return fmt.Errorf("failed to load embeddings: %w", err)
	}
	e.embeddings = embData
	log.Printf("[LoadModel] Loaded embeddings: shape=%v, dtype=%s, size=%d bytes",
		embInfo.Shape, embInfo.Dtype, len(embData))

	// Load LM head (may be tied to embeddings)
	lmHeadData, lmHeadInfo, err := loader.LoadTensor("lm_head.weight")
	if err != nil {
		// Fall back to tied embeddings (common in Llama)
		log.Printf("[LoadModel] lm_head.weight not found, using tied embeddings")
		lmHeadData = embData
		lmHeadInfo = embInfo
	}
	e.lmHead = lmHeadData
	log.Printf("[LoadModel] Loaded LM head: shape=%v, dtype=%s, size=%d bytes",
		lmHeadInfo.Shape, lmHeadInfo.Dtype, len(lmHeadData))

	// Validate shapes if we have a WeightLoader
	if wl, ok := loader.(*model.WeightLoader); ok {
		if err := wl.ValidateShapes(e.config); err != nil {
			log.Printf("[LoadModel] Warning: shape validation issues: %v", err)
		}
	}

	log.Printf("[LoadModel] Model loaded successfully from %s", modelPath)
	return nil
}

// LoadModelDistributed loads a model for distributed inference using the scheduler.
// It loads embeddings and LM head locally, and prepares layer weights for distribution.
func (e *Engine) LoadModelDistributed(ctx context.Context, modelPath string) (*model.DistributedModel, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.config == nil {
		return nil, fmt.Errorf("model config not set")
	}
	if e.scheduler == nil {
		return nil, fmt.Errorf("scheduler not set")
	}

	// Create distributed model
	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: e.config,
		ModelPath:   modelPath,
		LocalPeerID: e.localPeerID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create distributed model: %w", err)
	}

	// Set the scheduler
	dm.SetScheduler(e.scheduler)

	// Load to cluster
	if err := dm.LoadToCluster(ctx); err != nil {
		dm.Close()
		return nil, fmt.Errorf("failed to load to cluster: %w", err)
	}

	// Store embeddings and LM head in engine
	e.embeddings = dm.Embeddings()
	e.lmHead = dm.LMHead()

	log.Printf("[LoadModelDistributed] Model loaded for distributed inference from %s", modelPath)
	return dm, nil
}

// InitializeKVCaches creates KV caches for all layers based on assignments.
func (e *Engine) InitializeKVCaches() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.config == nil {
		return fmt.Errorf("model config not set")
	}

	for _, assignment := range e.assignments {
		// Skip embedding (-1) and output layer (numLayers)
		if assignment.LayerID < 0 || assignment.LayerID >= e.config.NumLayers {
			continue
		}

		isLocal := assignment.PeerID == e.localPeerID

		cache := NewDistributedKVCache(
			KVCacheConfig{
				LayerID:    assignment.LayerID,
				NumKVHeads: e.config.NumKVHeads,
				HeadDim:    e.config.HeadDim,
				MaxSeqLen:  e.config.MaxSeqLen,
			},
			assignment.PeerID,
			0, // Device ID determined by peer
			isLocal,
		)

		e.kvCaches.RegisterCache(cache)
	}

	return nil
}

// Generate performs text generation given a prompt.
// Uses exclusive lock to serialize requests (single GPU, shared KV cache state).
func (e *Engine) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.tokenizer == nil {
		return nil, fmt.Errorf("tokenizer not set")
	}

	// Use speculative decoding if available
	if e.specDecoder != nil && e.specDecoder.config.Enabled {
		return e.specDecoder.GenerateSpeculative(ctx, req)
	}

	// Tokenize input
	inputTokens, err := e.tokenizer.Encode(req.Prompt)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	log.Printf("[Generate] Input tokens: %v", inputTokens)

	// Get sequence ID for this request
	seqID := atomic.AddUint64(&e.seqCounter, 1)

	// Reset conv state for new request (but preserve cached KV blocks)
	if resetter, ok := e.layerExecutor.(interface{ ResetKVCache() error }); ok {
		if err := resetter.ResetKVCache(); err != nil {
			log.Printf("[Generate] Warning: KV cache reset failed: %v", err)
		}
	}
	e.kvCaches.ClearAll()

	// Allocate paged KV cache (prefix caching disabled — needs block retention fix)
	var cachedTokens int
	if pagedAlloc, ok := e.layerExecutor.(PagedCacheAllocator); ok {
		maxTokens := len(inputTokens) + req.MaxTokens
		if err := pagedAlloc.AllocateSequence(seqID, maxTokens); err != nil {
			log.Printf("[Generate] Paged cache allocation failed: %v", err)
		} else {
			defer pagedAlloc.FreeSequence(seqID)
		}
	}

	// Prefill: skip already-cached tokens, only process new ones
	prefillTokens := inputTokens[cachedTokens:]
	if cachedTokens > 0 {
		log.Printf("[Generate] Prefix cache: skipping %d/%d tokens (cached)", cachedTokens, len(inputTokens))
	}

	// Restore conv state from prefix cache (if cache hit)
	type convStateSaver interface {
		SaveConvStates() map[int][]byte
		RestoreConvStates(map[int][]byte)
	}
	if cachedTokens > 0 {
		if saver, ok := e.layerExecutor.(convStateSaver); ok && e.prefixCache != nil {
			if cached := e.prefixCache.GetConvState(inputTokens, cachedTokens); cached != nil {
				saver.RestoreConvStates(cached.States)
				log.Printf("[Generate] Restored conv state for %d cached tokens", cachedTokens)
			}
		}
	}

	var hidden []byte
	if len(prefillTokens) == 0 {
		// ALL tokens cached — recompute last token with restored conv state + cached KV
		lastToken := inputTokens[len(inputTokens)-1]
		tokenHidden, embErr := e.embedToken(lastToken)
		if embErr != nil {
			return nil, fmt.Errorf("embed cached last token: %w", embErr)
		}
		hidden, err = e.forwardAllLayersHidden(ctx, tokenHidden, len(inputTokens)-1, seqID)
		if err != nil {
			return nil, fmt.Errorf("forward cached last token: %w", err)
		}
		log.Printf("[Generate] Full prefix cache hit: recomputed last token at pos %d", len(inputTokens)-1)
	} else {
		hidden, err = e.prefillFrom(ctx, inputTokens, prefillTokens, cachedTokens, seqID)
		if err != nil {
			return nil, fmt.Errorf("prefill failed: %w", err)
		}
	}

	// Cache prefix blocks + conv state for future requests
	if pagedAlloc, ok := e.layerExecutor.(PagedCacheAllocator); ok {
		if mgr := pagedAlloc.GetPagedManager(); mgr != nil && e.prefixCache != nil {
			mgr.CacheSequencePrefix(seqID, inputTokens, e.prefixCache)
			// Save conv state snapshot for this prefix
			if saver, ok2 := e.layerExecutor.(convStateSaver); ok2 {
				numCached := len(inputTokens) / BlockSize * BlockSize // only full blocks
				if numCached > 0 {
					e.prefixCache.SaveConvState(inputTokens, numCached, saver.SaveConvStates())
				}
			}
		}
	}

	// Autoregressive decode
	outputTokens := make([]int, 0, req.MaxTokens)
	stopReason := "max_tokens"

	// Try GPU-resident decode path (hidden stays on GPU, eliminates copies)
	gpuDecoder, hasGPUDecoder := e.layerExecutor.(GPUResidentDecoder)
	gpuInf := e.gpuInference

	if hasGPUDecoder && gpuInf != nil {
		log.Printf("[Generate] Using GPU-resident decode path")
	} else {
		log.Printf("[Generate] Using per-layer decode path")
	}

	for i := 0; i < req.MaxTokens; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		var logits []float32

		if hasGPUDecoder && gpuInf != nil {
			if i == 0 {
				// First decode step: prefill output is already layer-processed.
				// Just apply LM head (with final RMSNorm) — don't run layers again.
				logits, err = e.applyLMHead(hidden)
				if err != nil {
					return nil, fmt.Errorf("LM head failed at step 0: %w", err)
				}
			} else {
				// Subsequent steps: hidden_a has the embedding, run through layers
				if err := gpuDecoder.DecodeStepGPUResident(len(inputTokens) + i - 1); err != nil {
					return nil, fmt.Errorf("GPU decode step %d failed: %w", i, err)
				}
				// Apply LM head (with final RMSNorm) from GPU hidden state
				hiddenGPUPtr := gpuDecoder.GetHiddenGPUPtr()
				if lmForwarder, ok := gpuInf.(GPULMHeadForwarder); ok {
					logits, err = lmForwarder.ApplyLMHeadFromGPU(hiddenGPUPtr)
					if err != nil {
						return nil, fmt.Errorf("LM head GPU forward failed at step %d: %w", i, err)
					}
				} else {
					gpuDecoder.GetHiddenGPU(hidden)
					logits, err = e.applyLMHead(hidden)
					if err != nil {
						return nil, fmt.Errorf("LM head failed at step %d: %w", i, err)
					}
				}
			}
		} else {
			// Standard path: forward all layers + LM head (with host copies)
			if i == 0 {
				// First token: hidden is the prefill output (already layer-processed)
				// Just apply LM head, don't run layers again
				logits, err = e.applyLMHead(hidden)
				if err != nil {
					return nil, fmt.Errorf("LM head failed at step %d: %w", i, err)
				}
			} else {
				logits, err = e.forwardAllLayers(ctx, hidden, len(inputTokens)+i-1, seqID)
				if err != nil {
					return nil, fmt.Errorf("forward pass failed at step %d: %w", i, err)
				}
			}
		}

		// Append token to paged KV cache (skip for i=0: no layer forward, just LM head)
		if i > 0 {
			if pagedAlloc, ok := e.layerExecutor.(PagedCacheAllocator); ok {
				if err := pagedAlloc.AppendToken(seqID); err != nil {
					log.Printf("[Generate] Warning: paged cache append failed at decode step %d: %v", i, err)
				}
			}
		}

		// Sample next token
		nextToken := e.sampler.Sample(logits, req.Temperature, req.TopP)
		outputTokens = append(outputTokens, nextToken)

		// Debug: log generated token
		if decoded, err := e.tokenizer.Decode([]int{nextToken}); err == nil {
			log.Printf("[Generate] Step %d: token=%d, decoded=%q, logits_min=%.4f, logits_max=%.4f",
				i, nextToken, decoded, minFloat32(logits), maxFloat32(logits))
		}

		// Check for EOS
		if nextToken == e.tokenizer.EOSToken() {
			stopReason = "eos"
			break
		}

		// Check for stop tokens
		for _, stop := range req.StopTokens {
			if nextToken == stop {
				stopReason = "stop_token"
				break
			}
		}
		if stopReason == "stop_token" {
			break
		}

		// Check for stop strings
		if len(req.StopStrings) > 0 {
			currentText, _ := e.tokenizer.Decode(outputTokens)
			for _, stopStr := range req.StopStrings {
				if idx := strings.Index(currentText, stopStr); idx != -1 {
					log.Printf("[Generate] Stop string %q found at position %d", stopStr, idx)
					truncatedText := currentText[:idx]
					outputTokens, _ = e.tokenizer.Encode(truncatedText)
					if len(outputTokens) > 0 && outputTokens[0] == e.tokenizer.BOSToken() {
						outputTokens = outputTokens[1:]
					}
					stopReason = "stop_string"
					break
				}
			}
			if stopReason == "stop_string" {
				break
			}
		}

		// Get embedding for next token and prepare for next step
		if hasGPUDecoder {
			// GPU-resident: use GPU→GPU embedding copy (zero host copy)
			if embedLookup, ok := gpuInf.(GPUEmbeddingLookup); ok {
				embPtr, embErr := embedLookup.EmbedTokenGPUPtr(nextToken)
				if embErr != nil {
					return nil, fmt.Errorf("GPU embed token failed: %w", embErr)
				}
				if err := gpuDecoder.SetHiddenFromGPU(embPtr); err != nil {
					return nil, fmt.Errorf("set hidden from GPU failed: %w", err)
				}
			} else {
				// Fallback: host copy
				hidden, err = e.embedToken(nextToken)
				if err != nil {
					return nil, fmt.Errorf("embed token failed: %w", err)
				}
				if err := gpuDecoder.SetHiddenGPU(hidden); err != nil {
					return nil, fmt.Errorf("set hidden GPU failed: %w", err)
				}
			}
		} else {
			hidden, err = e.embedToken(nextToken)
			if err != nil {
				return nil, fmt.Errorf("embed token failed: %w", err)
			}
		}
	}

	// Decode output tokens to text
	outputText, err := e.tokenizer.Decode(outputTokens)
	if err != nil {
		return nil, fmt.Errorf("decoding failed: %w", err)
	}

	return &GenerateResponse{
		Text:       outputText,
		TokenCount: len(outputTokens),
		StopReason: stopReason,
	}, nil
}

// GenerateStream performs streaming text generation, calling the callback for each token.
func (e *Engine) GenerateStream(ctx context.Context, req *GenerateRequest, callback TokenCallback) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.tokenizer == nil {
		return fmt.Errorf("tokenizer not set")
	}

	// Tokenize input
	inputTokens, err := e.tokenizer.Encode(req.Prompt)
	if err != nil {
		return fmt.Errorf("tokenization failed: %w", err)
	}
	log.Printf("[GenerateStream] Input tokens: %d", len(inputTokens))

	// Get sequence ID for this request
	seqID := atomic.AddUint64(&e.seqCounter, 1)

	// Reset CUDA KV caches and conv state for new request
	if resetter, ok := e.layerExecutor.(interface{ ResetKVCache() error }); ok {
		if err := resetter.ResetKVCache(); err != nil {
			log.Printf("[GenerateStream] Warning: KV cache reset failed: %v", err)
		}
	}

	// Clear KV caches for new sequence
	e.kvCaches.ClearAll()

	// Allocate paged KV cache blocks for this sequence if available
	if pagedAlloc, ok := e.layerExecutor.(PagedCacheAllocator); ok {
		maxTokens := len(inputTokens) + req.MaxTokens
		if err := pagedAlloc.AllocateSequence(seqID, maxTokens); err != nil {
			log.Printf("[GenerateStream] Paged cache allocation failed: %v (using contiguous fallback)", err)
		} else {
			defer pagedAlloc.FreeSequence(seqID)
		}
	}

	// Prefill phase - process all input tokens
	hidden, err := e.prefill(ctx, inputTokens, seqID)
	if err != nil {
		return fmt.Errorf("prefill failed: %w", err)
	}

	// Autoregressive decode with streaming
	tokenCount := 0
	var accumulatedText strings.Builder // For stop string detection

	for i := 0; i < req.MaxTokens; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Forward pass through all layers
		logits, err := e.forwardAllLayers(ctx, hidden, len(inputTokens)+i, seqID)
		if err != nil {
			return fmt.Errorf("forward pass failed at step %d: %w", i, err)
		}

		// Append token to paged KV cache
		if pagedAlloc, ok := e.layerExecutor.(PagedCacheAllocator); ok {
			if err := pagedAlloc.AppendToken(seqID); err != nil {
				log.Printf("[GenerateStream] Warning: paged cache append failed at step %d: %v", i, err)
			}
		}

		// Sample next token
		nextToken := e.sampler.Sample(logits, req.Temperature, req.TopP)
		tokenCount++

		// Decode single token to text
		tokenText, err := e.tokenizer.Decode([]int{nextToken})
		if err != nil {
			tokenText = "" // Continue even if decode fails
		}

		// Determine if this is final
		isFinal := false
		stopReason := ""

		// Check for EOS
		if nextToken == e.tokenizer.EOSToken() {
			isFinal = true
			stopReason = "eos"
		}

		// Check for stop tokens
		for _, stop := range req.StopTokens {
			if nextToken == stop {
				isFinal = true
				stopReason = "stop_token"
				break
			}
		}

		// Accumulate text and check for stop strings
		accumulatedText.WriteString(tokenText)
		if len(req.StopStrings) > 0 && !isFinal {
			currentText := accumulatedText.String()
			for _, stopStr := range req.StopStrings {
				if idx := strings.Index(currentText, stopStr); idx != -1 {
					// Truncate token text if stop string is in current token
					log.Printf("[GenerateStream] Stop string %q found", stopStr)
					// Calculate how much of the current token to keep
					textBeforeStop := currentText[:idx]
					prevLen := len(currentText) - len(tokenText)
					if idx >= prevLen {
						// Stop string starts within current token
						tokenText = textBeforeStop[prevLen:]
					} else {
						// Stop string was in previous text (shouldn't happen, but handle it)
						tokenText = ""
					}
					isFinal = true
					stopReason = "stop_string"
					break
				}
			}
		}

		// Check max tokens
		if i == req.MaxTokens-1 && !isFinal {
			isFinal = true
			stopReason = "max_tokens"
		}

		// Send token via callback
		streamToken := StreamToken{
			Token:      nextToken,
			Text:       tokenText,
			IsFirst:    i == 0,
			IsFinal:    isFinal,
			StopReason: stopReason,
		}

		if err := callback(streamToken); err != nil {
			return fmt.Errorf("callback failed: %w", err)
		}

		// Stop if final
		if isFinal {
			break
		}

		// Get embedding for next token
		hidden, err = e.embedToken(nextToken)
		if err != nil {
			return fmt.Errorf("embed token failed: %w", err)
		}
	}

	log.Printf("[GenerateStream] Generated %d tokens", tokenCount)
	return nil
}

// BatchPrefiller can process all input tokens in a single batched pass.
type BatchPrefiller interface {
	PrefillBatch(tokens []int, embeddingTable unsafe.Pointer, seqID uint64) ([]byte, error)
}

// prefill processes all input tokens through the model.
func (e *Engine) prefill(ctx context.Context, tokens []int, seqID uint64) ([]byte, error) {
	return e.prefillFrom(ctx, tokens, tokens, 0, seqID)
}

// prefillFrom processes tokens starting from positionOffset.
// allTokens is the full token list (for batch prefill). tokens is the subset to actually process.
// positionOffset is the starting position (non-zero when prefix is cached).
func (e *Engine) prefillFrom(ctx context.Context, allTokens []int, tokens []int, positionOffset int, seqID uint64) ([]byte, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty input tokens")
	}

	// Batch prefill: conv sequential + attention batched + per-token KV cache write
	if true {
		if batcher, ok := e.layerExecutor.(BatchPrefiller); ok && e.useGPU && e.gpuInference != nil {
			if embedLookup, ok2 := e.gpuInference.(GPUEmbeddingLookup); ok2 {
				embTable := embedLookup.(*GPUComponents).Embeddings.ptr
				batchHidden, err := batcher.PrefillBatch(tokens, embTable, seqID)
				if err != nil {
					log.Printf("[prefill] Batch prefill failed: %v", err)
				} else {
					log.Printf("[prefill] Batch prefill: %d tokens (offset=%d)", len(tokens), positionOffset)
					return batchHidden, nil
				}
			}
		}
	}

	// Sequential fallback (one token at a time)
	pagedAlloc, hasPaged := e.layerExecutor.(PagedCacheAllocator)

	var hidden []byte
	for i, token := range tokens {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		tokenHidden, err := e.embedToken(token)
		if err != nil {
			return nil, fmt.Errorf("embed token %d failed: %w", i, err)
		}

		hidden, err = e.forwardAllLayersHidden(ctx, tokenHidden, positionOffset+i, seqID)
		if err != nil {
			return nil, fmt.Errorf("forward at position %d failed: %w", positionOffset+i, err)
		}

		if hasPaged {
			if err := pagedAlloc.AppendToken(seqID); err != nil {
				log.Printf("[prefill] Warning: paged cache append failed at pos %d: %v", i, err)
			}
		}
	}

	return hidden, nil
}

// forwardAllLayers runs the hidden state through all layers and returns logits.
func (e *Engine) forwardAllLayers(ctx context.Context, hidden []byte, position int, seqID uint64) ([]float32, error) {
	// Forward through transformer layers
	output, err := e.forwardAllLayersHidden(ctx, hidden, position, seqID)
	if err != nil {
		return nil, err
	}

	// Apply LM head to get logits
	logits, err := e.applyLMHead(output)
	if err != nil {
		return nil, err
	}

	return logits, nil
}

// forwardAllLayersHidden forwards hidden state through all transformer layers.
func (e *Engine) forwardAllLayersHidden(ctx context.Context, hidden []byte, position int, seqID uint64) ([]byte, error) {
	// Fast path: use DecodeAll for LFM2 (all layers in single CUDA call)
	if decoder, ok := e.layerExecutor.(FullDecoder); ok {
		if out, err := decoder.DecodeAll(hidden, position); err == nil {
			return out, nil
		}
		// Fall through to per-layer path if DecodeAll not available
	}

	current := hidden

	for layerID := 0; layerID < e.config.NumLayers; layerID++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Check if this layer needs prefetch (cross-peer transition)
		if e.prefetch != nil && e.prefetch.NeedsPrefetch(layerID) {
			// Issue prefetch request for activation transfer
			peerID, _ := e.prefetch.GetNextPeer(layerID)
			e.prefetch.RequestPrefetch(scheduler.PrefetchRequest{
				LayerID:  layerID + 1,
				PeerID:   peerID,
				SeqID:    seqID,
				Priority: 1,
			}, nil)
		}

		if e.config.IsConvLayer(layerID) {
			// Conv layer: no KV output, no KV cache update
			output, err := e.forwardConvLayer(ctx, layerID, current, position)
			if err != nil {
				return nil, fmt.Errorf("conv layer %d forward failed: %w", layerID, err)
			}
			current = output
		} else {
			// Attention layer: standard forward with KV cache
			output, k, v, err := e.forwardLayer(ctx, layerID, current, position)
			if err != nil {
				return nil, fmt.Errorf("layer %d forward failed: %w", layerID, err)
			}

			// Update KV cache
			if cache, ok := e.kvCaches.GetCache(layerID); ok && cache.IsLocal() {
				if err := cache.Update(ctx, k, v, position); err != nil {
					return nil, fmt.Errorf("kv cache update failed for layer %d: %w", layerID, err)
				}
			}
			current = output
		}
	}

	return current, nil
}

// forwardConvLayer executes a conv layer forward pass (LFM2).
// Conv layers produce only hidden state output, no K/V.
func (e *Engine) forwardConvLayer(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, error) {
	// Use ConvLayerExecutor if the executor supports it
	if convExec, ok := e.layerExecutor.(ConvLayerExecutor); ok {
		return convExec.ForwardConv(ctx, layerID, hidden, position)
	}
	// Fallback: try standard forward (won't work for conv layers without GPU)
	output, _, _, err := e.forwardLayer(ctx, layerID, hidden, position)
	return output, err
}

// forwardLayer executes a single layer forward pass.
func (e *Engine) forwardLayer(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
	layerStart := time.Now()

	// Check if this is a local or remote layer
	var peerID string
	for _, a := range e.assignments {
		if a.LayerID == layerID {
			peerID = a.PeerID
			break
		}
	}

	isLocal := peerID == e.localPeerID
	location := "local"
	if !isLocal {
		location = "remote"
	}

	// DEBUG: Log layer routing decision (only for first few layers and first token)
	if position == 0 && layerID < 5 {
		log.Printf("[DEBUG] Layer %d: peerID=%s, localPeerID=%s, isLocal=%v", layerID, peerID, e.localPeerID, isLocal)
	}

	var output, k, v []byte
	var err error

	if isLocal {
		// Local execution
		if e.layerExecutor != nil {
			output, k, v, err = e.layerExecutor.Forward(ctx, layerID, hidden, position)
		} else {
			// Mock local execution for testing
			output = hidden
			k = make([]byte, e.config.NumKVHeads*e.config.HeadDim*2)
			v = make([]byte, e.config.NumKVHeads*e.config.HeadDim*2)
		}
	} else {
		// Remote execution - use registered remote executor for this peer
		e.mu.RLock()
		remoteExec, hasRemoteExec := e.remoteExecutors[peerID]
		e.mu.RUnlock()

		// DEBUG: Log remote execution info
		if position == 0 && layerID < 5 {
			log.Printf("[DEBUG] Layer %d: hasRemoteExec=%v, numRemoteExecutors=%d", layerID, hasRemoteExec, len(e.remoteExecutors))
		}

		if hasRemoteExec {
			// Use the remote layer executor to forward to peer
			output, k, v, err = remoteExec.Forward(ctx, layerID, hidden, position)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("remote layer %d forward failed (peer %s): %w", layerID, peerID, err)
			}
		} else {
			// Fallback to router-based activation transfer (legacy path)
			if e.router == nil {
				return nil, nil, nil, fmt.Errorf("no remote executor or router for peer %s", peerID)
			}

			seqID := atomic.LoadUint64(&e.seqCounter)
			if err := e.router.RouteActivation(ctx, layerID, seqID, hidden); err != nil {
				return nil, nil, nil, fmt.Errorf("route activation failed: %w", err)
			}

			// Legacy path: passthrough (caller needs to handle response separately)
			output = hidden
		}
	}

	if err != nil {
		return nil, nil, nil, err
	}

	// Record layer execution duration
	layerDuration := time.Since(layerStart).Seconds()
	metrics.RecordLayerDuration(layerID, location, layerDuration)

	// Perform health check on output tensor if enabled
	if e.enableHealthCheck && output != nil && len(output) > 0 {
		healthStart := time.Now()
		healthResult := metrics.CheckTensorHealthFP16(output)
		metrics.TensorHealthCheckDuration.Observe(time.Since(healthStart).Seconds())
		metrics.RecordTensorHealth(layerID, "hidden_state", healthResult)

		// Log warning if NaN/Inf detected
		if healthResult.NaNCount > 0 || healthResult.InfCount > 0 {
			log.Printf("[WARNING] Layer %d: detected %d NaN, %d Inf in output (sampled)",
				layerID, healthResult.NaNCount, healthResult.InfCount)
		}
	}

	return output, k, v, nil
}

// embedToken looks up the embedding for a token.
func (e *Engine) embedToken(token int) ([]byte, error) {
	// Use GPU embeddings when available
	if e.useGPU && e.gpuInference != nil {
		return e.gpuInference.EmbedTokenGPU(token)
	}

	// CPU fallback
	if e.embeddings == nil {
		// Return mock embedding for testing
		return make([]byte, e.config.HiddenSize*2), nil // FP16
	}

	if token < 0 || token >= e.config.VocabSize {
		return nil, fmt.Errorf("token %d out of vocabulary range [0, %d)", token, e.config.VocabSize)
	}

	// Calculate offset into embedding matrix
	// Embeddings are [vocabSize, hiddenSize] in FP16
	bytesPerEmbedding := e.config.HiddenSize * 2
	offset := token * bytesPerEmbedding

	if offset+bytesPerEmbedding > len(e.embeddings) {
		return nil, fmt.Errorf("embedding offset out of range")
	}

	return e.embeddings[offset : offset+bytesPerEmbedding], nil
}

// applyLMHead applies the output projection to get logits.
func (e *Engine) applyLMHead(hidden []byte) ([]float32, error) {
	// Use GPU LM head when available
	if e.useGPU && e.gpuInference != nil {
		return e.gpuInference.ApplyLMHeadGPU(hidden)
	}

	// CPU fallback - mock logits for testing
	if e.lmHead == nil {
		logits := make([]float32, e.config.VocabSize)
		// Set a peak at position 42 for deterministic testing
		logits[42] = 10.0
		return logits, nil
	}

	// CPU implementation would be matrix multiplication:
	// logits = hidden @ lmHead
	// For now, return mock logits (CPU inference not fully implemented)
	logits := make([]float32, e.config.VocabSize)
	return logits, nil
}

// Config returns the model configuration.
func (e *Engine) Config() *types.LlamaConfig {
	return e.config
}

// EnableSpeculativeDecoding enables self-speculative decoding with K draft tokens.
func (e *Engine) EnableSpeculativeDecoding(numSpecTokens int) {
	config := &SpeculativeConfig{NumSpecTokens: numSpecTokens, Enabled: true}
	e.specDecoder = NewSelfSpeculativeDecoder(e, config)
	log.Printf("[Engine] Speculative decoding enabled (K=%d)", numSpecTokens)
}

// KVCaches returns the KV cache manager.
func (e *Engine) KVCaches() *KVCacheManager {
	return e.kvCaches
}

// Assignments returns the layer assignments.
func (e *Engine) Assignments() []scheduler.LayerAssignment {
	return e.assignments
}

// ForwardLayerPublic is a public wrapper around forwardLayer for testing purposes.
// It executes a single layer forward pass, routing to local or remote executor as needed.
func (e *Engine) ForwardLayerPublic(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.forwardLayer(ctx, layerID, hidden, position)
}

// minFloat32 returns the minimum value in a float32 slice.
func minFloat32(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// maxFloat32 returns the maximum value in a float32 slice.
func maxFloat32(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

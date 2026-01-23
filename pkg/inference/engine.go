// Package inference provides distributed inference capabilities for LLM generation.
package inference

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"

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

// EngineConfig holds configuration for the inference engine.
type EngineConfig struct {
	ModelConfig *types.LlamaConfig
	LocalPeerID string
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

	mu sync.RWMutex
}

// NewEngine creates a new distributed inference engine.
func NewEngine(config EngineConfig) *Engine {
	return &Engine{
		config:          config.ModelConfig,
		kvCaches:        NewKVCacheManager(),
		sampler:         NewSampler(42), // Default seed
		localPeerID:     config.LocalPeerID,
		remoteExecutors: make(map[string]LayerExecutor),
	}
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
func (e *Engine) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.tokenizer == nil {
		return nil, fmt.Errorf("tokenizer not set")
	}

	// Tokenize input
	inputTokens, err := e.tokenizer.Encode(req.Prompt)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	log.Printf("[Generate] Input tokens: %v", inputTokens)

	// Get sequence ID for this request
	seqID := atomic.AddUint64(&e.seqCounter, 1)

	// Clear KV caches for new sequence
	e.kvCaches.ClearAll()

	// Prefill phase - process all input tokens
	hidden, err := e.prefill(ctx, inputTokens, seqID)
	if err != nil {
		return nil, fmt.Errorf("prefill failed: %w", err)
	}

	// Autoregressive decode
	outputTokens := make([]int, 0, req.MaxTokens)
	stopReason := "max_tokens"

	for i := 0; i < req.MaxTokens; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Forward pass through all layers
		logits, err := e.forwardAllLayers(ctx, hidden, len(inputTokens)+i, seqID)
		if err != nil {
			return nil, fmt.Errorf("forward pass failed at step %d: %w", i, err)
		}

		// Sample next token
		nextToken := e.sampler.Sample(logits, req.Temperature, req.TopP)
		outputTokens = append(outputTokens, nextToken)

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

		// Get embedding for next token
		hidden, err = e.embedToken(nextToken)
		if err != nil {
			return nil, fmt.Errorf("embed token failed: %w", err)
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
	e.mu.RLock()
	defer e.mu.RUnlock()

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

	// Clear KV caches for new sequence
	e.kvCaches.ClearAll()

	// Prefill phase - process all input tokens
	hidden, err := e.prefill(ctx, inputTokens, seqID)
	if err != nil {
		return fmt.Errorf("prefill failed: %w", err)
	}

	// Autoregressive decode with streaming
	tokenCount := 0

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

// prefill processes all input tokens through the model.
func (e *Engine) prefill(ctx context.Context, tokens []int, seqID uint64) ([]byte, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty input tokens")
	}

	// Get embeddings for all input tokens
	var hidden []byte
	for i, token := range tokens {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Embed token
		tokenHidden, err := e.embedToken(token)
		if err != nil {
			return nil, fmt.Errorf("embed token %d failed: %w", i, err)
		}

		// Forward through all layers at this position
		hidden, err = e.forwardAllLayersHidden(ctx, tokenHidden, i, seqID)
		if err != nil {
			return nil, fmt.Errorf("forward at position %d failed: %w", i, err)
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

		// Execute layer forward pass
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

	return current, nil
}

// forwardLayer executes a single layer forward pass.
func (e *Engine) forwardLayer(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
	// Check if this is a local or remote layer
	var peerID string
	for _, a := range e.assignments {
		if a.LayerID == layerID {
			peerID = a.PeerID
			break
		}
	}

	isLocal := peerID == e.localPeerID

	// DEBUG: Log layer routing decision (only for first few layers and first token)
	if position == 0 && layerID < 5 {
		log.Printf("[DEBUG] Layer %d: peerID=%s, localPeerID=%s, isLocal=%v", layerID, peerID, e.localPeerID, isLocal)
	}

	if isLocal {
		// Local execution
		if e.layerExecutor != nil {
			return e.layerExecutor.Forward(ctx, layerID, hidden, position)
		}
		// Mock local execution for testing
		return hidden, make([]byte, e.config.NumKVHeads*e.config.HeadDim*2), make([]byte, e.config.NumKVHeads*e.config.HeadDim*2), nil
	}

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
		output, k, v, err := remoteExec.Forward(ctx, layerID, hidden, position)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("remote layer %d forward failed (peer %s): %w", layerID, peerID, err)
		}
		return output, k, v, nil
	}

	// Fallback to router-based activation transfer (legacy path)
	if e.router == nil {
		return nil, nil, nil, fmt.Errorf("no remote executor or router for peer %s", peerID)
	}

	seqID := atomic.LoadUint64(&e.seqCounter)
	if err := e.router.RouteActivation(ctx, layerID, seqID, hidden); err != nil {
		return nil, nil, nil, fmt.Errorf("route activation failed: %w", err)
	}

	// Legacy path: passthrough (caller needs to handle response separately)
	return hidden, nil, nil, nil
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

//go:build cuda

// Package inference provides GPU-accelerated inference capabilities.
package inference

import (
	"fmt"
	"log"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/model"
)

// GPUComponents holds all GPU-accelerated components for the engine.
type GPUComponents struct {
	Embeddings   *GPUEmbeddings
	LMHead       *GPULMHead
	LayerExecutor *CUDALayerExecutor
	DeviceID     int
	Initialized  bool
}

// InitializeGPU sets up the GPU inference pipeline.
// This loads all weights to GPU memory and initializes CUDA execution.
func (e *Engine) InitializeGPU(loader *model.WeightLoader, deviceID int) (*GPUComponents, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Initialize CUDA
	if err := bindings.InitGPU(deviceID); err != nil {
		return nil, fmt.Errorf("CUDA init failed: %w", err)
	}

	// Initialize multi-GPU support (required for AllocOnDevice)
	if err := bindings.InitMultiGPU([]int{deviceID}); err != nil {
		return nil, fmt.Errorf("multi-GPU init failed: %w", err)
	}

	// Get device info for logging
	info, err := bindings.GetDeviceInfo()
	if err == nil {
		log.Printf("GPU: %s (Compute %d.%d, Memory: %.2f GB)",
			info.Name, info.Major, info.Minor,
			float64(info.TotalMemory)/(1024*1024*1024))
	}

	gpu := &GPUComponents{DeviceID: deviceID}

	// 1. Load embeddings to GPU
	log.Printf("Loading embeddings to GPU...")
	embData, _, err := loader.LoadEmbeddings()
	if err != nil {
		return nil, fmt.Errorf("load embeddings: %w", err)
	}

	gpu.Embeddings, err = NewGPUEmbeddings(embData, e.config.VocabSize, e.config.HiddenSize)
	if err != nil {
		return nil, fmt.Errorf("GPU embeddings: %w", err)
	}
	log.Printf("Embeddings loaded to GPU: %d tokens x %d dims", e.config.VocabSize, e.config.HiddenSize)

	// 2. Load LM head and final layernorm to GPU
	log.Printf("Loading LM head to GPU...")
	lmData, _, err := loader.LoadLMHead()
	if err != nil {
		gpu.Embeddings.Close()
		return nil, fmt.Errorf("load lm head: %w", err)
	}

	// Load final layernorm (model.norm.weight or model.embedding_norm.weight for LFM2)
	log.Printf("Loading final layernorm...")
	finalNorm, _, err := loader.LoadTensor("model.norm.weight")
	if err != nil {
		// Try LFM2 naming: model.embedding_norm.weight
		finalNorm, _, err = loader.LoadTensor("model.embedding_norm.weight")
		if err != nil {
			gpu.Embeddings.Close()
			return nil, fmt.Errorf("load final norm: %w", err)
		}
	}

	gpu.LMHead, err = NewGPULMHeadWithNorm(lmData, finalNorm, e.config.HiddenSize, e.config.VocabSize, e.config.RMSNormEps)
	if err != nil {
		gpu.Embeddings.Close()
		return nil, fmt.Errorf("GPU LM head: %w", err)
	}
	log.Printf("LM head loaded to GPU: %d x %d (with final layernorm)", e.config.HiddenSize, e.config.VocabSize)

	// 3. Create CUDA layer executor
	layerExecutor, err := NewCUDALayerExecutor(e.config, deviceID)
	if err != nil {
		gpu.Close()
		return nil, fmt.Errorf("create CUDA layer executor: %w", err)
	}
	gpu.LayerExecutor = layerExecutor

	// 4. Load all transformer layers
	for layerID := 0; layerID < e.config.NumLayers; layerID++ {
		log.Printf("Loading layer %d/%d to GPU...", layerID+1, e.config.NumLayers)

		if e.config.IsConvLayer(layerID) {
			// Load LFM2 conv layer
			convWeights, err := loader.LoadConvLayerWeights(layerID)
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load conv layer %d: %w", layerID, err)
			}

			if err := gpu.LayerExecutor.LoadConvLayer(layerID,
				convWeights.InProjWeight, convWeights.ConvWeight, convWeights.OutProjWeight,
				convWeights.OperatorNorm, convWeights.FFNNorm,
				convWeights.GateWeight, convWeights.UpWeight, convWeights.DownWeight,
				e.config.HiddenSize, e.config.IntermediateSize, e.config.ConvKernelSize, e.config.RMSNormEps,
			); err != nil {
				gpu.Close()
				return nil, fmt.Errorf("GPU conv layer %d: %w", layerID, err)
			}
		} else if e.config.ModelType == "lfm2" {
			// Load LFM2 attention layer (different tensor names: out_proj, feed_forward, operator_norm)
			layerWeights, qLayerNorm, kLayerNorm, err := loader.LoadAttentionLayerWeightsLFM2(layerID)
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load LFM2 attn layer %d: %w", layerID, err)
			}

			weights := &TransformerLayerWeights{
				QProj:      layerWeights.QWeight,
				KProj:      layerWeights.KWeight,
				VProj:      layerWeights.VWeight,
				OProj:      layerWeights.OWeight,
				GateProj:   layerWeights.GateWeight,
				UpProj:     layerWeights.UpWeight,
				DownProj:   layerWeights.DownWeight,
				AttnNorm:   layerWeights.AttnNorm,
				FFNNorm:    layerWeights.FFNNorm,
				QLayerNorm: qLayerNorm,
				KLayerNorm: kLayerNorm,
			}

			if err := gpu.LayerExecutor.LoadLayerFP16(layerID, weights); err != nil {
				gpu.Close()
				return nil, fmt.Errorf("GPU attn layer %d: %w", layerID, err)
			}
		} else {
			// Load standard Llama attention layer
			qProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load q_proj layer %d: %w", layerID, err)
			}

			kProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load k_proj layer %d: %w", layerID, err)
			}

			vProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load v_proj layer %d: %w", layerID, err)
			}

			oProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load o_proj layer %d: %w", layerID, err)
			}

			gateProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load gate_proj layer %d: %w", layerID, err)
			}

			upProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load up_proj layer %d: %w", layerID, err)
			}

			downProj, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load down_proj layer %d: %w", layerID, err)
			}

			attnNorm, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.input_layernorm.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load attn_norm layer %d: %w", layerID, err)
			}

			ffnNorm, _, err := loader.LoadTensor(fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layerID))
			if err != nil {
				gpu.Close()
				return nil, fmt.Errorf("load ffn_norm layer %d: %w", layerID, err)
			}

			// Create TransformerLayerWeights and upload to GPU
			weights := &TransformerLayerWeights{
				QProj:    qProj,
				KProj:    kProj,
				VProj:    vProj,
				OProj:    oProj,
				GateProj: gateProj,
				UpProj:   upProj,
				DownProj: downProj,
				AttnNorm: attnNorm,
				FFNNorm:  ffnNorm,
			}

			if err := gpu.LayerExecutor.LoadLayer(layerID, weights); err != nil {
				gpu.Close()
				return nil, fmt.Errorf("GPU layer %d: %w", layerID, err)
			}
		}
	}
	log.Printf("All %d layers loaded to GPU", e.config.NumLayers)

	// Report GPU memory usage
	used, err := bindings.GetMemoryUsed()
	if err == nil {
		log.Printf("GPU memory used: %.2f GB", float64(used)/(1024*1024*1024))
	}

	// Set the layer executor on the engine
	e.layerExecutor = gpu.LayerExecutor

	// Enable GPU inference mode
	e.gpuInference = gpu
	e.useGPU = true

	gpu.Initialized = true
	log.Printf("GPU inference mode ENABLED")
	return gpu, nil
}

// Close releases all GPU resources.
func (g *GPUComponents) Close() error {
	if g.Embeddings != nil {
		g.Embeddings.Close()
	}
	if g.LMHead != nil {
		g.LMHead.Close()
	}
	if g.LayerExecutor != nil {
		g.LayerExecutor.Close()
	}
	// Shutdown multi-GPU context to allow re-initialization in tests
	bindings.ShutdownMultiGPU()
	g.Initialized = false
	return nil
}

// EmbedTokenGPU looks up the embedding for a token using GPU embeddings.
func (g *GPUComponents) EmbedTokenGPU(token int) ([]byte, error) {
	if g.Embeddings == nil {
		return nil, fmt.Errorf("GPU embeddings not initialized")
	}
	return g.Embeddings.LookupToHost(token)
}

// ApplyLMHeadGPU applies the LM head to hidden state using GPU.
func (g *GPUComponents) ApplyLMHeadGPU(hidden []byte) ([]float32, error) {
	if g.LMHead == nil {
		return nil, fmt.Errorf("GPU LM head not initialized")
	}
	return g.LMHead.Forward(hidden)
}

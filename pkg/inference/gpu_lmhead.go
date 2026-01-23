//go:build cuda

// Package inference provides GPU-accelerated LM head operations.
package inference

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
)

// GPULMHead holds the language model head weights on GPU.
// Used to compute logits from hidden states.
type GPULMHead struct {
	ptr           unsafe.Pointer // GPU memory pointer to LM head weights
	finalNormPtr  unsafe.Pointer // GPU memory pointer to final layernorm weights
	hiddenSize    int            // Input dimension
	vocabSize     int            // Output dimension (vocabulary size)
	byteSize      uint64         // Total bytes on GPU
	rmsNormEps    float32        // RMSNorm epsilon

	// Preallocated GPU buffers for Forward() - avoids per-token cudaMalloc/cudaFree
	hiddenGPU     unsafe.Pointer // GPU buffer for hidden state input
	normalizedGPU unsafe.Pointer // GPU buffer for RMSNorm output
	logitsGPU     unsafe.Pointer // GPU buffer for logits output
}

// NewGPULMHead uploads LM head weights to GPU memory (without final norm).
// data: FP16 weights in row-major order [hiddenSize, vocabSize]
func NewGPULMHead(data []byte, hiddenSize, vocabSize int) (*GPULMHead, error) {
	return NewGPULMHeadWithNorm(data, nil, hiddenSize, vocabSize, 1e-6)
}

// NewGPULMHeadWithNorm uploads LM head weights and final layernorm to GPU memory.
// data: FP16 weights in row-major order [hiddenSize, vocabSize]
// finalNorm: FP16 weights for final RMSNorm [hiddenSize]
func NewGPULMHeadWithNorm(data []byte, finalNorm []byte, hiddenSize, vocabSize int, rmsNormEps float32) (*GPULMHead, error) {
	expectedSize := hiddenSize * vocabSize * 2 // FP16 = 2 bytes per element
	if len(data) != expectedSize {
		return nil, fmt.Errorf("LM head data size mismatch: got %d, expected %d", len(data), expectedSize)
	}

	// Allocate GPU memory for LM head
	ptr, err := bindings.AllocOnDevice(uint64(expectedSize), 0)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU memory for LM head: %w", err)
	}

	// Copy weights to GPU
	if err := bindings.CopyToDeviceRaw(ptr, unsafe.Pointer(&data[0]), uint64(expectedSize)); err != nil {
		bindings.FreeOnDevice(ptr, 0)
		return nil, fmt.Errorf("failed to copy LM head to GPU: %w", err)
	}

	lmHead := &GPULMHead{
		ptr:        ptr,
		hiddenSize: hiddenSize,
		vocabSize:  vocabSize,
		byteSize:   uint64(expectedSize),
		rmsNormEps: rmsNormEps,
	}

	// Preallocate GPU buffers for Forward() to avoid per-token allocations
	hiddenBufSize := uint64(hiddenSize * 2) // FP16
	logitsBufSize := uint64(vocabSize * 2)  // FP16

	lmHead.hiddenGPU, err = bindings.AllocOnDevice(hiddenBufSize, 0)
	if err != nil {
		bindings.FreeOnDevice(ptr, 0)
		return nil, fmt.Errorf("failed to preallocate hidden buffer: %w", err)
	}

	lmHead.normalizedGPU, err = bindings.AllocOnDevice(hiddenBufSize, 0)
	if err != nil {
		bindings.FreeOnDevice(ptr, 0)
		bindings.FreeOnDevice(lmHead.hiddenGPU, 0)
		return nil, fmt.Errorf("failed to preallocate normalized buffer: %w", err)
	}

	lmHead.logitsGPU, err = bindings.AllocOnDevice(logitsBufSize, 0)
	if err != nil {
		bindings.FreeOnDevice(ptr, 0)
		bindings.FreeOnDevice(lmHead.hiddenGPU, 0)
		bindings.FreeOnDevice(lmHead.normalizedGPU, 0)
		return nil, fmt.Errorf("failed to preallocate logits buffer: %w", err)
	}

	// Upload final layernorm weights if provided
	if finalNorm != nil {
		expectedNormSize := hiddenSize * 2 // FP16
		if len(finalNorm) != expectedNormSize {
			lmHead.freePreallocated()
			bindings.FreeOnDevice(ptr, 0)
			return nil, fmt.Errorf("final norm size mismatch: got %d, expected %d", len(finalNorm), expectedNormSize)
		}

		normPtr, err := bindings.AllocOnDevice(uint64(expectedNormSize), 0)
		if err != nil {
			lmHead.freePreallocated()
			bindings.FreeOnDevice(ptr, 0)
			return nil, fmt.Errorf("failed to allocate GPU memory for final norm: %w", err)
		}

		if err := bindings.CopyToDeviceRaw(normPtr, unsafe.Pointer(&finalNorm[0]), uint64(expectedNormSize)); err != nil {
			lmHead.freePreallocated()
			bindings.FreeOnDevice(ptr, 0)
			bindings.FreeOnDevice(normPtr, 0)
			return nil, fmt.Errorf("failed to copy final norm to GPU: %w", err)
		}

		lmHead.finalNormPtr = normPtr
	}

	return lmHead, nil
}

// freePreallocated frees the preallocated GPU buffers.
func (h *GPULMHead) freePreallocated() {
	if h.hiddenGPU != nil {
		bindings.FreeOnDevice(h.hiddenGPU, 0)
		h.hiddenGPU = nil
	}
	if h.normalizedGPU != nil {
		bindings.FreeOnDevice(h.normalizedGPU, 0)
		h.normalizedGPU = nil
	}
	if h.logitsGPU != nil {
		bindings.FreeOnDevice(h.logitsGPU, 0)
		h.logitsGPU = nil
	}
}

// Forward computes logits from hidden state.
// hidden: FP16 hidden state [hiddenSize] in GPU memory (as byte slice)
// returns: FP32 logits [vocabSize] in CPU memory for sampling
//
// OPTIMIZED: Uses preallocated GPU buffers to avoid per-token cudaMalloc/cudaFree.
func (h *GPULMHead) Forward(hidden []byte) ([]float32, error) {
	expectedHiddenSize := h.hiddenSize * 2 // FP16
	if len(hidden) != expectedHiddenSize {
		return nil, fmt.Errorf("hidden size mismatch: got %d bytes, expected %d", len(hidden), expectedHiddenSize)
	}

	// Copy hidden to preallocated GPU buffer (no allocation needed)
	if err := bindings.CopyToDeviceRaw(h.hiddenGPU, unsafe.Pointer(&hidden[0]), uint64(len(hidden))); err != nil {
		return nil, fmt.Errorf("failed to copy hidden to GPU: %w", err)
	}

	// Apply final layernorm if available
	inputForGEMM := h.hiddenGPU
	if h.finalNormPtr != nil {
		// Create tensors for RMSNorm using preallocated buffers
		inputTensor := &types.Tensor{
			Shape:  []int{1, h.hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: 0,
			Data:   h.hiddenGPU,
		}
		outputTensor := &types.Tensor{
			Shape:  []int{1, h.hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: 0,
			Data:   h.normalizedGPU,
		}
		weightTensor := &types.Tensor{
			Shape:  []int{h.hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: 0,
			Data:   h.finalNormPtr,
		}

		// Apply RMSNorm
		if err := bindings.RMSNorm(outputTensor, inputTensor, weightTensor, h.rmsNormEps); err != nil {
			return nil, fmt.Errorf("final RMSNorm failed: %w", err)
		}

		inputForGEMM = h.normalizedGPU
	}

	// Create tensors for GEMM using preallocated logits buffer
	// hidden: [1, hiddenSize]
	// lm_head: [vocabSize, hiddenSize] (stored in row-major, needs transpose)
	// output: [1, vocabSize]
	// GEMM: output = hidden @ lm_head^T = [1, hiddenSize] @ [hiddenSize, vocabSize] = [1, vocabSize]
	hiddenTensor := &types.Tensor{
		Shape:  []int{1, h.hiddenSize},
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   inputForGEMM,
	}

	lmHeadTensor := &types.Tensor{
		Shape:  []int{h.vocabSize, h.hiddenSize}, // Actual shape in memory
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   h.ptr,
	}

	logitsTensor := &types.Tensor{
		Shape:  []int{1, h.vocabSize},
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   h.logitsGPU,
	}

	// Execute GEMM: logits = hidden @ lm_head^T (transB=true to transpose the weight matrix)
	if err := bindings.GEMMFP16(logitsTensor, hiddenTensor, lmHeadTensor, false, true); err != nil {
		return nil, fmt.Errorf("LM head GEMM failed: %w", err)
	}

	// Copy logits to host as FP16
	logitsSize := h.vocabSize * 2 // FP16
	logitsFP16 := make([]byte, logitsSize)
	if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&logitsFP16[0]), h.logitsGPU, uint64(logitsSize)); err != nil {
		return nil, fmt.Errorf("failed to copy logits from GPU: %w", err)
	}

	// Convert FP16 to FP32 on CPU (for sampling)
	logits := make([]float32, h.vocabSize)
	for i := 0; i < h.vocabSize; i++ {
		logits[i] = fp16ToFloat32(logitsFP16[i*2 : i*2+2])
	}

	return logits, nil
}

// ForwardFromGPU computes logits when hidden state is already on GPU.
// hiddenGPU: GPU pointer to FP16 hidden state
// returns: FP32 logits in CPU memory
//
// OPTIMIZED: Uses preallocated logits buffer to avoid per-token cudaMalloc/cudaFree.
func (h *GPULMHead) ForwardFromGPU(hiddenGPUPtr unsafe.Pointer) ([]float32, error) {
	// Create tensors for GEMM using preallocated logits buffer
	// lm_head weights are [vocabSize, hiddenSize], need transB=true
	hiddenTensor := &types.Tensor{
		Shape:  []int{1, h.hiddenSize},
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   hiddenGPUPtr,
	}

	lmHeadTensor := &types.Tensor{
		Shape:  []int{h.vocabSize, h.hiddenSize}, // Actual shape in memory
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   h.ptr,
	}

	logitsTensor := &types.Tensor{
		Shape:  []int{1, h.vocabSize},
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   h.logitsGPU,
	}

	// Execute GEMM with transB=true
	if err := bindings.GEMMFP16(logitsTensor, hiddenTensor, lmHeadTensor, false, true); err != nil {
		return nil, fmt.Errorf("LM head GEMM failed: %w", err)
	}

	// Copy logits to host and convert
	logitsSize := h.vocabSize * 2 // FP16
	logitsFP16 := make([]byte, logitsSize)
	if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&logitsFP16[0]), h.logitsGPU, uint64(logitsSize)); err != nil {
		return nil, fmt.Errorf("failed to copy logits from GPU: %w", err)
	}

	logits := make([]float32, h.vocabSize)
	for i := 0; i < h.vocabSize; i++ {
		logits[i] = fp16ToFloat32(logitsFP16[i*2 : i*2+2])
	}

	return logits, nil
}

// HiddenSize returns the input dimension.
func (h *GPULMHead) HiddenSize() int {
	return h.hiddenSize
}

// VocabSize returns the vocabulary size.
func (h *GPULMHead) VocabSize() int {
	return h.vocabSize
}

// Close frees the GPU memory.
func (h *GPULMHead) Close() error {
	// Free preallocated buffers
	h.freePreallocated()

	// Free LM head weights
	if h.ptr != nil {
		if err := bindings.FreeOnDevice(h.ptr, 0); err != nil {
			return fmt.Errorf("failed to free GPU LM head: %w", err)
		}
		h.ptr = nil
	}

	// Free final layernorm weights
	if h.finalNormPtr != nil {
		if err := bindings.FreeOnDevice(h.finalNormPtr, 0); err != nil {
			return fmt.Errorf("failed to free GPU final norm: %w", err)
		}
		h.finalNormPtr = nil
	}
	return nil
}

// fp16ToFloat32 converts 2 bytes of FP16 to float32.
func fp16ToFloat32(b []byte) float32 {
	if len(b) < 2 {
		return 0
	}

	// FP16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
	bits := uint16(b[0]) | uint16(b[1])<<8

	sign := (bits >> 15) & 1
	exp := (bits >> 10) & 0x1F
	mant := bits & 0x3FF

	var f float32

	if exp == 0 {
		if mant == 0 {
			// Zero
			f = 0
		} else {
			// Denormalized number
			f = float32(mant) / float32(1<<10) * float32(math.Pow(2, -14))
		}
	} else if exp == 31 {
		if mant == 0 {
			// Infinity
			f = float32(math.Inf(1))
		} else {
			// NaN
			f = float32(math.NaN())
		}
	} else {
		// Normalized number
		f = (1 + float32(mant)/float32(1<<10)) * float32(math.Pow(2, float64(exp)-15))
	}

	if sign == 1 {
		f = -f
	}

	return f
}

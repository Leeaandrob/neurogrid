//go:build cuda

// Package inference provides GPU-accelerated LM head operations.
package inference

import (
	"fmt"
	"log"
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

	// Upload final layernorm weights if provided
	if finalNorm != nil {
		expectedNormSize := hiddenSize * 2 // FP16
		if len(finalNorm) != expectedNormSize {
			bindings.FreeOnDevice(ptr, 0)
			return nil, fmt.Errorf("final norm size mismatch: got %d, expected %d", len(finalNorm), expectedNormSize)
		}

		normPtr, err := bindings.AllocOnDevice(uint64(expectedNormSize), 0)
		if err != nil {
			bindings.FreeOnDevice(ptr, 0)
			return nil, fmt.Errorf("failed to allocate GPU memory for final norm: %w", err)
		}

		if err := bindings.CopyToDeviceRaw(normPtr, unsafe.Pointer(&finalNorm[0]), uint64(expectedNormSize)); err != nil {
			bindings.FreeOnDevice(ptr, 0)
			bindings.FreeOnDevice(normPtr, 0)
			return nil, fmt.Errorf("failed to copy final norm to GPU: %w", err)
		}

		lmHead.finalNormPtr = normPtr
	}

	return lmHead, nil
}

// Forward computes logits from hidden state.
// hidden: FP16 hidden state [hiddenSize] in GPU memory (as byte slice)
// returns: FP32 logits [vocabSize] in CPU memory for sampling
func (h *GPULMHead) Forward(hidden []byte) ([]float32, error) {
	expectedHiddenSize := h.hiddenSize * 2 // FP16
	if len(hidden) != expectedHiddenSize {
		return nil, fmt.Errorf("hidden size mismatch: got %d bytes, expected %d", len(hidden), expectedHiddenSize)
	}

	// DEBUG: Print first few hidden values (FP16)
	var hiddenVals []float32
	for i := 0; i < 8 && i*2+1 < len(hidden); i++ {
		hiddenVals = append(hiddenVals, fp16ToFloat32(hidden[i*2:i*2+2]))
	}
	log.Printf("[LMHead] Input hidden (first 8 FP16 values): %v", hiddenVals)

	// Allocate GPU buffer for hidden state
	hiddenGPU, err := bindings.AllocOnDevice(uint64(len(hidden)), 0)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate hidden buffer: %w", err)
	}
	defer bindings.FreeOnDevice(hiddenGPU, 0)

	// Copy hidden to GPU
	if err := bindings.CopyToDeviceRaw(hiddenGPU, unsafe.Pointer(&hidden[0]), uint64(len(hidden))); err != nil {
		return nil, fmt.Errorf("failed to copy hidden to GPU: %w", err)
	}

	// Apply final layernorm if available
	inputForGEMM := hiddenGPU
	if h.finalNormPtr != nil {
		// Allocate buffer for normalized output
		normalizedGPU, err := bindings.AllocOnDevice(uint64(len(hidden)), 0)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate normalized buffer: %w", err)
		}
		defer bindings.FreeOnDevice(normalizedGPU, 0)

		// Create tensors for RMSNorm
		inputTensor := &types.Tensor{
			Shape:  []int{1, h.hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: 0,
			Data:   hiddenGPU,
		}
		outputTensor := &types.Tensor{
			Shape:  []int{1, h.hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: 0,
			Data:   normalizedGPU,
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

		// DEBUG: Copy normalized output back and print
		normDebug := make([]byte, len(hidden))
		if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&normDebug[0]), normalizedGPU, uint64(len(hidden))); err == nil {
			var normVals []float32
			for i := 0; i < 8 && i*2+1 < len(normDebug); i++ {
				normVals = append(normVals, fp16ToFloat32(normDebug[i*2:i*2+2]))
			}
			log.Printf("[LMHead] After RMSNorm (first 8 FP16 values): %v", normVals)
		}

		inputForGEMM = normalizedGPU
	}

	// Allocate GPU buffer for logits (FP16)
	logitsSize := h.vocabSize * 2 // FP16
	logitsGPU, err := bindings.AllocOnDevice(uint64(logitsSize), 0)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate logits buffer: %w", err)
	}
	defer bindings.FreeOnDevice(logitsGPU, 0)

	// Create tensors for GEMM
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
		Data:   logitsGPU,
	}

	// Execute GEMM: logits = hidden @ lm_head^T (transB=true to transpose the weight matrix)
	if err := bindings.GEMMFP16(logitsTensor, hiddenTensor, lmHeadTensor, false, true); err != nil {
		return nil, fmt.Errorf("LM head GEMM failed: %w", err)
	}

	// Copy logits to host as FP16
	logitsFP16 := make([]byte, logitsSize)
	if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&logitsFP16[0]), logitsGPU, uint64(logitsSize)); err != nil {
		return nil, fmt.Errorf("failed to copy logits from GPU: %w", err)
	}

	// Convert FP16 to FP32 on CPU (for sampling)
	logits := make([]float32, h.vocabSize)
	for i := 0; i < h.vocabSize; i++ {
		logits[i] = fp16ToFloat32(logitsFP16[i*2 : i*2+2])
	}

	// DEBUG: Print top 5 logits
	type logitEntry struct {
		idx int
		val float32
	}
	topN := make([]logitEntry, 5)
	for i, v := range logits {
		for j := 0; j < 5; j++ {
			if v > topN[j].val {
				copy(topN[j+1:], topN[j:4])
				topN[j] = logitEntry{i, v}
				break
			}
		}
	}
	log.Printf("[LMHead] Top 5 logits: [%d]=%.3f [%d]=%.3f [%d]=%.3f [%d]=%.3f [%d]=%.3f",
		topN[0].idx, topN[0].val, topN[1].idx, topN[1].val, topN[2].idx, topN[2].val,
		topN[3].idx, topN[3].val, topN[4].idx, topN[4].val)

	return logits, nil
}

// ForwardFromGPU computes logits when hidden state is already on GPU.
// hiddenGPU: GPU pointer to FP16 hidden state
// returns: FP32 logits in CPU memory
func (h *GPULMHead) ForwardFromGPU(hiddenGPU unsafe.Pointer) ([]float32, error) {
	// Allocate GPU buffer for logits (FP16)
	logitsSize := h.vocabSize * 2 // FP16
	logitsGPUPtr, err := bindings.AllocOnDevice(uint64(logitsSize), 0)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate logits buffer: %w", err)
	}
	defer bindings.FreeOnDevice(logitsGPUPtr, 0)

	// Create tensors for GEMM
	// lm_head weights are [vocabSize, hiddenSize], need transB=true
	hiddenTensor := &types.Tensor{
		Shape:  []int{1, h.hiddenSize},
		Dtype:  types.DtypeFP16,
		Device: 0,
		Data:   hiddenGPU,
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
		Data:   logitsGPUPtr,
	}

	// Execute GEMM with transB=true
	if err := bindings.GEMMFP16(logitsTensor, hiddenTensor, lmHeadTensor, false, true); err != nil {
		return nil, fmt.Errorf("LM head GEMM failed: %w", err)
	}

	// Copy logits to host and convert
	logitsFP16 := make([]byte, logitsSize)
	if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&logitsFP16[0]), logitsGPUPtr, uint64(logitsSize)); err != nil {
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
	if h.ptr != nil {
		if err := bindings.FreeOnDevice(h.ptr, 0); err != nil {
			return fmt.Errorf("failed to free GPU LM head: %w", err)
		}
		h.ptr = nil
	}
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

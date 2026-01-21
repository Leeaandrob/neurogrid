//go:build cuda

// Package inference provides GPU-accelerated embedding operations.
package inference

import (
	"fmt"
	"unsafe"

	"github.com/neurogrid/engine/gpu/bindings"
)

// GPUEmbeddings holds token embeddings in GPU memory for fast lookup.
type GPUEmbeddings struct {
	ptr        unsafe.Pointer // GPU memory pointer to embeddings table
	vocabSize  int            // Number of tokens in vocabulary
	hiddenSize int            // Embedding dimension (hidden_size)
	byteSize   uint64         // Total bytes on GPU
}

// NewGPUEmbeddings uploads embedding table to GPU memory.
// data: FP16 embeddings in row-major order [vocabSize, hiddenSize]
func NewGPUEmbeddings(data []byte, vocabSize, hiddenSize int) (*GPUEmbeddings, error) {
	expectedSize := vocabSize * hiddenSize * 2 // FP16 = 2 bytes per element
	if len(data) != expectedSize {
		return nil, fmt.Errorf("embedding data size mismatch: got %d, expected %d", len(data), expectedSize)
	}

	// Allocate GPU memory
	ptr, err := bindings.AllocOnDevice(uint64(expectedSize), 0)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU memory for embeddings: %w", err)
	}

	// Copy embeddings to GPU
	if err := bindings.CopyToDeviceRaw(ptr, unsafe.Pointer(&data[0]), uint64(expectedSize)); err != nil {
		bindings.FreeOnDevice(ptr, 0)
		return nil, fmt.Errorf("failed to copy embeddings to GPU: %w", err)
	}

	return &GPUEmbeddings{
		ptr:        ptr,
		vocabSize:  vocabSize,
		hiddenSize: hiddenSize,
		byteSize:   uint64(expectedSize),
	}, nil
}

// Lookup returns a GPU pointer to the embedding for a token.
// The returned pointer can be used directly in GPU operations.
func (e *GPUEmbeddings) Lookup(tokenID int) (unsafe.Pointer, error) {
	if tokenID < 0 || tokenID >= e.vocabSize {
		return nil, fmt.Errorf("token ID %d out of range [0, %d)", tokenID, e.vocabSize)
	}

	// Calculate offset: tokenID * hiddenSize * sizeof(FP16)
	bytesPerEmbed := e.hiddenSize * 2 // FP16
	offset := uintptr(tokenID * bytesPerEmbed)

	// Return pointer to the specific embedding
	return unsafe.Pointer(uintptr(e.ptr) + offset), nil
}

// LookupToHost copies the embedding for a token to host memory.
// Returns FP16 data as bytes.
func (e *GPUEmbeddings) LookupToHost(tokenID int) ([]byte, error) {
	ptr, err := e.Lookup(tokenID)
	if err != nil {
		return nil, err
	}

	size := e.hiddenSize * 2 // FP16
	data := make([]byte, size)

	if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&data[0]), ptr, uint64(size)); err != nil {
		return nil, fmt.Errorf("failed to copy embedding from GPU: %w", err)
	}

	return data, nil
}

// VocabSize returns the vocabulary size.
func (e *GPUEmbeddings) VocabSize() int {
	return e.vocabSize
}

// HiddenSize returns the embedding dimension.
func (e *GPUEmbeddings) HiddenSize() int {
	return e.hiddenSize
}

// Close frees the GPU memory.
func (e *GPUEmbeddings) Close() error {
	if e.ptr != nil {
		if err := bindings.FreeOnDevice(e.ptr, 0); err != nil {
			return fmt.Errorf("failed to free GPU embeddings: %w", err)
		}
		e.ptr = nil
	}
	return nil
}

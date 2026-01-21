// Package types provides core data types for the NeuroGrid engine.
package types

import (
	"fmt"
	"unsafe"
)

// Dtype represents the data type of tensor elements.
type Dtype int

const (
	DtypeFP32 Dtype = iota
	DtypeFP16
	DtypeINT8
)

// String returns the string representation of the Dtype.
func (d Dtype) String() string {
	switch d {
	case DtypeFP32:
		return "FP32"
	case DtypeFP16:
		return "FP16"
	case DtypeINT8:
		return "INT8"
	default:
		return "UNKNOWN"
	}
}

// ByteSize returns the byte size of one element of this dtype.
func (d Dtype) ByteSize() int {
	switch d {
	case DtypeFP32:
		return 4
	case DtypeFP16:
		return 2
	case DtypeINT8:
		return 1
	default:
		return 0
	}
}

// Tensor represents a GPU tensor with device memory.
type Tensor struct {
	Data   unsafe.Pointer // CUDA device pointer
	Shape  []int          // Dimensions [batch, seq, hidden] etc.
	Dtype  Dtype          // Element data type
	Device int            // GPU device ID
}

// NewTensor creates a new tensor descriptor (does not allocate GPU memory).
func NewTensor(shape []int, dtype Dtype, device int) *Tensor {
	return &Tensor{
		Data:   nil,
		Shape:  shape,
		Dtype:  dtype,
		Device: device,
	}
}

// NumElements returns the total number of elements in the tensor.
func (t *Tensor) NumElements() int {
	n := 1
	for _, dim := range t.Shape {
		n *= dim
	}
	return n
}

// ByteSize returns the total byte size of the tensor data.
func (t *Tensor) ByteSize() int {
	return t.NumElements() * t.Dtype.ByteSize()
}

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int {
	return len(t.Shape)
}

// IsAllocated returns true if GPU memory is allocated.
func (t *Tensor) IsAllocated() bool {
	return t.Data != nil
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%d, allocated=%v)",
		t.Shape, t.Dtype, t.Device, t.IsAllocated())
}

// Validate checks if the tensor configuration is valid.
func (t *Tensor) Validate() error {
	if len(t.Shape) == 0 {
		return fmt.Errorf("tensor shape cannot be empty")
	}
	for i, dim := range t.Shape {
		if dim <= 0 {
			return fmt.Errorf("tensor dimension %d must be positive, got %d", i, dim)
		}
	}
	return nil
}

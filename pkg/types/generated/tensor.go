// Package generated contains types generated from FlatBuffers schemas.
// This file provides manual implementations that match the schema in schemas/tensor.fbs
// until flatc is available to generate proper FlatBuffers code.
package generated

import (
	"encoding/binary"
)

// DType represents the data type of tensor elements.
type DType byte

const (
	DTypeFloat32  DType = 0
	DTypeFloat16  DType = 1
	DTypeInt8     DType = 2
	DTypeBFloat16 DType = 3
	DTypeInt32    DType = 4
	DTypeInt64    DType = 5
)

// String returns the string representation of DType.
func (d DType) String() string {
	switch d {
	case DTypeFloat32:
		return "Float32"
	case DTypeFloat16:
		return "Float16"
	case DTypeInt8:
		return "Int8"
	case DTypeBFloat16:
		return "BFloat16"
	case DTypeInt32:
		return "Int32"
	case DTypeInt64:
		return "Int64"
	default:
		return "Unknown"
	}
}

// TensorMeta contains metadata about a tensor.
type TensorMeta struct {
	Shape    []int64
	Dtype    DType
	DeviceID int32
	LayerID  int32
	Position int32
}

// Activation represents a tensor activation message.
type Activation struct {
	Meta       TensorMeta
	Data       []byte
	SequenceID uint64
}

// KVCacheEntry represents a key-value cache entry for attention layers.
type KVCacheEntry struct {
	LayerID  int32
	Position int32
	Key      []byte
	Value    []byte
	Dtype    DType
	HeadDim  int32
	NumHeads int32
}

// ActivationBatch represents a batch of activations for pipelined transfer.
type ActivationBatch struct {
	Activations []Activation
	BatchID     uint64
}

// Serialize serializes an Activation to bytes.
// Format: [MetaLen:4][Meta][DataLen:4][Data][SequenceID:8]
func (a *Activation) Serialize() []byte {
	// Serialize meta
	metaBytes := a.Meta.Serialize()

	// Calculate total size
	totalSize := 4 + len(metaBytes) + 4 + len(a.Data) + 8
	buf := make([]byte, totalSize)
	offset := 0

	// Write meta length
	binary.BigEndian.PutUint32(buf[offset:offset+4], uint32(len(metaBytes)))
	offset += 4

	// Write meta
	copy(buf[offset:offset+len(metaBytes)], metaBytes)
	offset += len(metaBytes)

	// Write data length
	binary.BigEndian.PutUint32(buf[offset:offset+4], uint32(len(a.Data)))
	offset += 4

	// Write data
	copy(buf[offset:offset+len(a.Data)], a.Data)
	offset += len(a.Data)

	// Write sequence ID
	binary.BigEndian.PutUint64(buf[offset:offset+8], a.SequenceID)

	return buf
}

// Deserialize deserializes bytes into an Activation.
func (a *Activation) Deserialize(buf []byte) error {
	offset := 0

	// Read meta length
	metaLen := binary.BigEndian.Uint32(buf[offset : offset+4])
	offset += 4

	// Read meta
	metaBytes := buf[offset : offset+int(metaLen)]
	a.Meta.Deserialize(metaBytes)
	offset += int(metaLen)

	// Read data length
	dataLen := binary.BigEndian.Uint32(buf[offset : offset+4])
	offset += 4

	// Read data
	a.Data = make([]byte, dataLen)
	copy(a.Data, buf[offset:offset+int(dataLen)])
	offset += int(dataLen)

	// Read sequence ID
	a.SequenceID = binary.BigEndian.Uint64(buf[offset : offset+8])

	return nil
}

// Serialize serializes TensorMeta to bytes.
// Format: [NumDims:4][Shape:NumDims*8][Dtype:1][DeviceID:4][LayerID:4][Position:4]
func (m *TensorMeta) Serialize() []byte {
	// Calculate size
	size := 4 + len(m.Shape)*8 + 1 + 4 + 4 + 4
	buf := make([]byte, size)
	offset := 0

	// Write number of dimensions
	binary.BigEndian.PutUint32(buf[offset:offset+4], uint32(len(m.Shape)))
	offset += 4

	// Write shape
	for _, dim := range m.Shape {
		binary.BigEndian.PutUint64(buf[offset:offset+8], uint64(dim))
		offset += 8
	}

	// Write dtype
	buf[offset] = byte(m.Dtype)
	offset++

	// Write device ID
	binary.BigEndian.PutUint32(buf[offset:offset+4], uint32(m.DeviceID))
	offset += 4

	// Write layer ID
	binary.BigEndian.PutUint32(buf[offset:offset+4], uint32(m.LayerID))
	offset += 4

	// Write position
	binary.BigEndian.PutUint32(buf[offset:offset+4], uint32(m.Position))

	return buf
}

// Deserialize deserializes bytes into TensorMeta.
func (m *TensorMeta) Deserialize(buf []byte) {
	offset := 0

	// Read number of dimensions
	numDims := binary.BigEndian.Uint32(buf[offset : offset+4])
	offset += 4

	// Read shape
	m.Shape = make([]int64, numDims)
	for i := uint32(0); i < numDims; i++ {
		m.Shape[i] = int64(binary.BigEndian.Uint64(buf[offset : offset+8]))
		offset += 8
	}

	// Read dtype
	m.Dtype = DType(buf[offset])
	offset++

	// Read device ID
	m.DeviceID = int32(binary.BigEndian.Uint32(buf[offset : offset+4]))
	offset += 4

	// Read layer ID
	m.LayerID = int32(binary.BigEndian.Uint32(buf[offset : offset+4]))
	offset += 4

	// Read position
	m.Position = int32(binary.BigEndian.Uint32(buf[offset : offset+4]))
}

// ElementSize returns the size in bytes of a single element of the given dtype.
func ElementSize(dtype DType) int {
	switch dtype {
	case DTypeFloat32, DTypeInt32:
		return 4
	case DTypeFloat16, DTypeBFloat16:
		return 2
	case DTypeInt8:
		return 1
	case DTypeInt64:
		return 8
	default:
		return 0
	}
}

// TensorSize returns the total size in bytes of a tensor with the given shape and dtype.
func TensorSize(shape []int64, dtype DType) int64 {
	if len(shape) == 0 {
		return 0
	}

	numElements := int64(1)
	for _, dim := range shape {
		numElements *= dim
	}

	return numElements * int64(ElementSize(dtype))
}

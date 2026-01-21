// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
// Tests for TASK-014: FlatBuffers Schema and Code Generation
package e2e

import (
	"testing"

	"github.com/neurogrid/engine/pkg/types/generated"
)

// =============================================================================
// TASK-014: FlatBuffers Schema Tests
// =============================================================================

// TestDType_Values validates DType enum values
func TestDType_Values(t *testing.T) {
	tests := []struct {
		dtype    generated.DType
		expected byte
		name     string
	}{
		{generated.DTypeFloat32, 0, "Float32"},
		{generated.DTypeFloat16, 1, "Float16"},
		{generated.DTypeInt8, 2, "Int8"},
		{generated.DTypeBFloat16, 3, "BFloat16"},
		{generated.DTypeInt32, 4, "Int32"},
		{generated.DTypeInt64, 5, "Int64"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if byte(tc.dtype) != tc.expected {
				t.Errorf("DType %s: got %d, expected %d", tc.name, byte(tc.dtype), tc.expected)
			}
			if tc.dtype.String() != tc.name {
				t.Errorf("DType.String(): got %s, expected %s", tc.dtype.String(), tc.name)
			}
		})
	}

	t.Log("PASS: DType values are correct")
}

// TestElementSize_AllTypes validates element size calculation
func TestElementSize_AllTypes(t *testing.T) {
	tests := []struct {
		dtype    generated.DType
		expected int
	}{
		{generated.DTypeFloat32, 4},
		{generated.DTypeFloat16, 2},
		{generated.DTypeInt8, 1},
		{generated.DTypeBFloat16, 2},
		{generated.DTypeInt32, 4},
		{generated.DTypeInt64, 8},
	}

	for _, tc := range tests {
		size := generated.ElementSize(tc.dtype)
		if size != tc.expected {
			t.Errorf("ElementSize(%s): got %d, expected %d", tc.dtype, size, tc.expected)
		}
	}

	t.Log("PASS: ElementSize works correctly")
}

// TestTensorSize_Calculation validates tensor size calculation
func TestTensorSize_Calculation(t *testing.T) {
	tests := []struct {
		shape    []int64
		dtype    generated.DType
		expected int64
		name     string
	}{
		{[]int64{1024}, generated.DTypeFloat32, 4096, "1D float32"},
		{[]int64{8, 1024}, generated.DTypeFloat16, 16384, "2D float16"},
		{[]int64{2, 512, 4096}, generated.DTypeFloat32, 16777216, "3D float32"},
		{[]int64{1, 1, 1}, generated.DTypeInt8, 1, "scalar int8"},
		{[]int64{}, generated.DTypeFloat32, 0, "empty shape"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			size := generated.TensorSize(tc.shape, tc.dtype)
			if size != tc.expected {
				t.Errorf("TensorSize(%v, %s): got %d, expected %d", tc.shape, tc.dtype, size, tc.expected)
			}
		})
	}

	t.Log("PASS: TensorSize calculation works correctly")
}

// TestTensorMeta_SerializeDeserialize validates TensorMeta round-trip
func TestTensorMeta_SerializeDeserialize(t *testing.T) {
	original := generated.TensorMeta{
		Shape:    []int64{8, 512, 4096},
		Dtype:    generated.DTypeFloat16,
		DeviceID: 0,
		LayerID:  5,
		Position: 100,
	}

	// Serialize
	data := original.Serialize()
	if len(data) == 0 {
		t.Fatal("Serialization returned empty data")
	}

	// Deserialize
	var restored generated.TensorMeta
	restored.Deserialize(data)

	// Verify
	if len(restored.Shape) != len(original.Shape) {
		t.Errorf("Shape length mismatch: got %d, expected %d", len(restored.Shape), len(original.Shape))
	}

	for i, dim := range original.Shape {
		if restored.Shape[i] != dim {
			t.Errorf("Shape[%d] mismatch: got %d, expected %d", i, restored.Shape[i], dim)
		}
	}

	if restored.Dtype != original.Dtype {
		t.Errorf("Dtype mismatch: got %d, expected %d", restored.Dtype, original.Dtype)
	}

	if restored.DeviceID != original.DeviceID {
		t.Errorf("DeviceID mismatch: got %d, expected %d", restored.DeviceID, original.DeviceID)
	}

	if restored.LayerID != original.LayerID {
		t.Errorf("LayerID mismatch: got %d, expected %d", restored.LayerID, original.LayerID)
	}

	if restored.Position != original.Position {
		t.Errorf("Position mismatch: got %d, expected %d", restored.Position, original.Position)
	}

	t.Log("PASS: TensorMeta serialization round-trip works")
}

// TestActivation_SerializeDeserialize validates Activation round-trip
func TestActivation_SerializeDeserialize(t *testing.T) {
	// Create test data
	testData := make([]byte, 1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	original := generated.Activation{
		Meta: generated.TensorMeta{
			Shape:    []int64{1, 256},
			Dtype:    generated.DTypeFloat32,
			DeviceID: 1,
			LayerID:  10,
			Position: 50,
		},
		Data:       testData,
		SequenceID: 12345,
	}

	// Serialize
	data := original.Serialize()
	if len(data) == 0 {
		t.Fatal("Serialization returned empty data")
	}

	// Deserialize
	var restored generated.Activation
	err := restored.Deserialize(data)
	if err != nil {
		t.Fatalf("Deserialization failed: %v", err)
	}

	// Verify metadata
	if restored.Meta.LayerID != original.Meta.LayerID {
		t.Errorf("Meta.LayerID mismatch: got %d, expected %d", restored.Meta.LayerID, original.Meta.LayerID)
	}

	// Verify data
	if len(restored.Data) != len(original.Data) {
		t.Errorf("Data length mismatch: got %d, expected %d", len(restored.Data), len(original.Data))
	}

	for i := 0; i < 100; i++ {
		if restored.Data[i] != original.Data[i] {
			t.Errorf("Data[%d] mismatch: got %d, expected %d", i, restored.Data[i], original.Data[i])
			break
		}
	}

	// Verify sequence ID
	if restored.SequenceID != original.SequenceID {
		t.Errorf("SequenceID mismatch: got %d, expected %d", restored.SequenceID, original.SequenceID)
	}

	t.Log("PASS: Activation serialization round-trip works")
}

// TestActivation_LargeTensor validates large tensor serialization
func TestActivation_LargeTensor(t *testing.T) {
	// 1MB tensor
	testData := make([]byte, 1024*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	original := generated.Activation{
		Meta: generated.TensorMeta{
			Shape:    []int64{256, 1024},
			Dtype:    generated.DTypeFloat32,
			DeviceID: 0,
			LayerID:  20,
			Position: 0,
		},
		Data:       testData,
		SequenceID: 99999,
	}

	// Serialize
	data := original.Serialize()

	// Deserialize
	var restored generated.Activation
	err := restored.Deserialize(data)
	if err != nil {
		t.Fatalf("Deserialization failed: %v", err)
	}

	// Verify data integrity
	if len(restored.Data) != len(original.Data) {
		t.Errorf("Data length mismatch: got %d, expected %d", len(restored.Data), len(original.Data))
	}

	// Sample check
	for i := 0; i < 1000; i += 100 {
		if restored.Data[i] != original.Data[i] {
			t.Errorf("Data[%d] mismatch: got %d, expected %d", i, restored.Data[i], original.Data[i])
		}
	}

	t.Log("PASS: Large tensor serialization works")
}

// TestKVCacheEntry_Fields validates KVCacheEntry structure
func TestKVCacheEntry_Fields(t *testing.T) {
	entry := generated.KVCacheEntry{
		LayerID:  5,
		Position: 100,
		Key:      make([]byte, 2048),
		Value:    make([]byte, 2048),
		Dtype:    generated.DTypeFloat16,
		HeadDim:  64,
		NumHeads: 32,
	}

	if entry.LayerID != 5 {
		t.Errorf("LayerID mismatch: got %d, expected 5", entry.LayerID)
	}

	if entry.Position != 100 {
		t.Errorf("Position mismatch: got %d, expected 100", entry.Position)
	}

	if entry.HeadDim != 64 {
		t.Errorf("HeadDim mismatch: got %d, expected 64", entry.HeadDim)
	}

	if entry.NumHeads != 32 {
		t.Errorf("NumHeads mismatch: got %d, expected 32", entry.NumHeads)
	}

	t.Log("PASS: KVCacheEntry fields work correctly")
}

// TestActivationBatch_Fields validates ActivationBatch structure
func TestActivationBatch_Fields(t *testing.T) {
	batch := generated.ActivationBatch{
		Activations: []generated.Activation{
			{
				Meta: generated.TensorMeta{
					Shape:   []int64{1, 512},
					Dtype:   generated.DTypeFloat32,
					LayerID: 0,
				},
				Data:       make([]byte, 2048),
				SequenceID: 1,
			},
			{
				Meta: generated.TensorMeta{
					Shape:   []int64{1, 512},
					Dtype:   generated.DTypeFloat32,
					LayerID: 1,
				},
				Data:       make([]byte, 2048),
				SequenceID: 2,
			},
		},
		BatchID: 42,
	}

	if len(batch.Activations) != 2 {
		t.Errorf("Activations length mismatch: got %d, expected 2", len(batch.Activations))
	}

	if batch.BatchID != 42 {
		t.Errorf("BatchID mismatch: got %d, expected 42", batch.BatchID)
	}

	t.Log("PASS: ActivationBatch fields work correctly")
}

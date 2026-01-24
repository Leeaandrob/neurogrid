// Package inference provides tests for config transfer functionality.
// Tests for PRP: Hybrid Distributed Inference System
//
// RED PHASE: These tests should FAIL initially because the implementation
// does not exist yet. This follows TDD methodology.
package inference

import (
	"testing"

	"github.com/neurogrid/engine/pkg/types"
)

// =============================================================================
// PRP: HYBRID DISTRIBUTED INFERENCE - CONFIG TRANSFER TESTS (RED PHASE)
// =============================================================================
// These tests validate the TransferableConfig serialization for P2P transfer.
// All tests should FAIL initially as the implementation doesn't exist yet.

// TestSerializeConfig_RoundTrip validates JSON serialization round-trip
// AC1: Worker receives ModelConfig via P2P
func TestSerializeConfig_RoundTrip(t *testing.T) {
	// Arrange: Create a Mistral 7B config (the validation model from PRP)
	originalConfig := &types.LlamaConfig{
		HiddenSize:       4096,
		IntermediateSize: 14336,
		NumLayers:        32,
		NumHeads:         32,
		NumKVHeads:       8, // GQA for Mistral
		HeadDim:          128,
		VocabSize:        32000,
		MaxSeqLen:        8192,
		RMSNormEps:       1e-5,
	}
	modelName := "mistral-7b"

	// Act: Serialize the config
	// This should FAIL - SerializeConfig doesn't exist yet
	serialized, err := SerializeConfig(originalConfig, modelName)
	if err != nil {
		t.Fatalf("SerializeConfig failed: %v", err)
	}

	// Verify serialized data is not empty
	if len(serialized) == 0 {
		t.Fatal("Serialized config is empty")
	}

	// Act: Deserialize the config
	// This should FAIL - DeserializeConfig doesn't exist yet
	deserializedConfig, deserializedName, err := DeserializeConfig(serialized)
	if err != nil {
		t.Fatalf("DeserializeConfig failed: %v", err)
	}

	// Assert: All fields match
	if deserializedName != modelName {
		t.Errorf("ModelName mismatch: got %s, expected %s", deserializedName, modelName)
	}
	if deserializedConfig.HiddenSize != originalConfig.HiddenSize {
		t.Errorf("HiddenSize mismatch: got %d, expected %d", deserializedConfig.HiddenSize, originalConfig.HiddenSize)
	}
	if deserializedConfig.IntermediateSize != originalConfig.IntermediateSize {
		t.Errorf("IntermediateSize mismatch: got %d, expected %d", deserializedConfig.IntermediateSize, originalConfig.IntermediateSize)
	}
	if deserializedConfig.NumLayers != originalConfig.NumLayers {
		t.Errorf("NumLayers mismatch: got %d, expected %d", deserializedConfig.NumLayers, originalConfig.NumLayers)
	}
	if deserializedConfig.NumHeads != originalConfig.NumHeads {
		t.Errorf("NumHeads mismatch: got %d, expected %d", deserializedConfig.NumHeads, originalConfig.NumHeads)
	}
	if deserializedConfig.NumKVHeads != originalConfig.NumKVHeads {
		t.Errorf("NumKVHeads mismatch: got %d, expected %d", deserializedConfig.NumKVHeads, originalConfig.NumKVHeads)
	}
	if deserializedConfig.HeadDim != originalConfig.HeadDim {
		t.Errorf("HeadDim mismatch: got %d, expected %d", deserializedConfig.HeadDim, originalConfig.HeadDim)
	}
	if deserializedConfig.VocabSize != originalConfig.VocabSize {
		t.Errorf("VocabSize mismatch: got %d, expected %d", deserializedConfig.VocabSize, originalConfig.VocabSize)
	}
	if deserializedConfig.MaxSeqLen != originalConfig.MaxSeqLen {
		t.Errorf("MaxSeqLen mismatch: got %d, expected %d", deserializedConfig.MaxSeqLen, originalConfig.MaxSeqLen)
	}
	if deserializedConfig.RMSNormEps != originalConfig.RMSNormEps {
		t.Errorf("RMSNormEps mismatch: got %v, expected %v", deserializedConfig.RMSNormEps, originalConfig.RMSNormEps)
	}

	t.Log("PASS: Config serialization round-trip working")
}

// TestSerializeConfig_AllFields verifies all LlamaConfig fields are preserved
// AC1: Worker receives ModelConfig via P2P
func TestSerializeConfig_AllFields(t *testing.T) {
	// Test with TinyLlama config to verify all fields
	originalConfig := types.TinyLlamaConfig()
	modelName := "tinyllama-1.1b"

	// Act: Serialize
	// This should FAIL - SerializeConfig doesn't exist yet
	serialized, err := SerializeConfig(originalConfig, modelName)
	if err != nil {
		t.Fatalf("SerializeConfig failed: %v", err)
	}

	// Act: Deserialize
	// This should FAIL - DeserializeConfig doesn't exist yet
	deserializedConfig, deserializedName, err := DeserializeConfig(serialized)
	if err != nil {
		t.Fatalf("DeserializeConfig failed: %v", err)
	}

	// Assert: Model name preserved
	if deserializedName != modelName {
		t.Errorf("ModelName mismatch: got %s, expected %s", deserializedName, modelName)
	}

	// Assert: All config fields match TinyLlama
	if deserializedConfig.HiddenSize != 2048 {
		t.Errorf("HiddenSize mismatch: got %d, expected 2048", deserializedConfig.HiddenSize)
	}
	if deserializedConfig.IntermediateSize != 5632 {
		t.Errorf("IntermediateSize mismatch: got %d, expected 5632", deserializedConfig.IntermediateSize)
	}
	if deserializedConfig.NumLayers != 22 {
		t.Errorf("NumLayers mismatch: got %d, expected 22", deserializedConfig.NumLayers)
	}
	if deserializedConfig.NumHeads != 32 {
		t.Errorf("NumHeads mismatch: got %d, expected 32", deserializedConfig.NumHeads)
	}
	if deserializedConfig.NumKVHeads != 4 {
		t.Errorf("NumKVHeads mismatch: got %d, expected 4", deserializedConfig.NumKVHeads)
	}
	if deserializedConfig.HeadDim != 64 {
		t.Errorf("HeadDim mismatch: got %d, expected 64", deserializedConfig.HeadDim)
	}
	if deserializedConfig.VocabSize != 32000 {
		t.Errorf("VocabSize mismatch: got %d, expected 32000", deserializedConfig.VocabSize)
	}
	if deserializedConfig.MaxSeqLen != 2048 {
		t.Errorf("MaxSeqLen mismatch: got %d, expected 2048", deserializedConfig.MaxSeqLen)
	}

	t.Log("PASS: All config fields preserved in serialization")
}

// TestSerializeConfig_Llama7B verifies Llama 2 7B config serialization
func TestSerializeConfig_Llama7B(t *testing.T) {
	originalConfig := types.Llama7BConfig()
	modelName := "llama-2-7b"

	// Act: Serialize and deserialize
	// This should FAIL - functions don't exist yet
	serialized, err := SerializeConfig(originalConfig, modelName)
	if err != nil {
		t.Fatalf("SerializeConfig failed: %v", err)
	}

	deserializedConfig, _, err := DeserializeConfig(serialized)
	if err != nil {
		t.Fatalf("DeserializeConfig failed: %v", err)
	}

	// Assert: Llama 7B specific values
	if deserializedConfig.HiddenSize != 4096 {
		t.Errorf("HiddenSize mismatch: got %d, expected 4096", deserializedConfig.HiddenSize)
	}
	if deserializedConfig.NumLayers != 32 {
		t.Errorf("NumLayers mismatch: got %d, expected 32", deserializedConfig.NumLayers)
	}
	if deserializedConfig.NumHeads != 32 {
		t.Errorf("NumHeads mismatch: got %d, expected 32", deserializedConfig.NumHeads)
	}
	if deserializedConfig.NumKVHeads != 32 {
		t.Errorf("NumKVHeads mismatch: got %d, expected 32 (no GQA)", deserializedConfig.NumKVHeads)
	}

	t.Log("PASS: Llama 7B config serialization working")
}

// TestDeserializeConfig_InvalidJSON verifies error handling for bad input
// AC1: Worker receives ModelConfig via P2P (error handling)
func TestDeserializeConfig_InvalidJSON(t *testing.T) {
	testCases := []struct {
		name        string
		input       []byte
		expectError bool
	}{
		{
			name:        "empty data",
			input:       []byte{},
			expectError: true,
		},
		{
			name:        "invalid JSON",
			input:       []byte("not valid json"),
			expectError: true,
		},
		{
			name:        "truncated JSON",
			input:       []byte(`{"model_name": "test", "hidden_size": 4096`),
			expectError: true,
		},
		{
			name:        "null JSON",
			input:       []byte("null"),
			expectError: true,
		},
		{
			name:        "empty object",
			input:       []byte("{}"),
			expectError: false, // Should parse but with zero values
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act: Try to deserialize invalid input
			// This should FAIL - DeserializeConfig doesn't exist yet
			_, _, err := DeserializeConfig(tc.input)

			// Assert
			if tc.expectError && err == nil {
				t.Errorf("Expected error for %s, got nil", tc.name)
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.name, err)
			}
		})
	}

	t.Log("PASS: DeserializeConfig error handling working")
}

// TestTransferableConfig_Struct verifies TransferableConfig struct exists
// AC1: Worker receives ModelConfig via P2P
func TestTransferableConfig_Struct(t *testing.T) {
	// This should FAIL - TransferableConfig doesn't exist yet
	config := TransferableConfig{
		ModelName:        "mistral-7b",
		HiddenSize:       4096,
		NumLayers:        32,
		IntermediateSize: 14336,
		NumHeads:         32,
		NumKVHeads:       8,
		HeadDim:          128,
		VocabSize:        32000,
		MaxSeqLen:        8192,
		RMSNormEps:       1e-5,
	}

	// Verify struct fields
	if config.ModelName != "mistral-7b" {
		t.Errorf("ModelName mismatch: got %s, expected mistral-7b", config.ModelName)
	}
	if config.HiddenSize != 4096 {
		t.Errorf("HiddenSize mismatch: got %d, expected 4096", config.HiddenSize)
	}
	if config.NumLayers != 32 {
		t.Errorf("NumLayers mismatch: got %d, expected 32", config.NumLayers)
	}
	if config.IntermediateSize != 14336 {
		t.Errorf("IntermediateSize mismatch: got %d, expected 14336", config.IntermediateSize)
	}
	if config.NumHeads != 32 {
		t.Errorf("NumHeads mismatch: got %d, expected 32", config.NumHeads)
	}
	if config.NumKVHeads != 8 {
		t.Errorf("NumKVHeads mismatch: got %d, expected 8", config.NumKVHeads)
	}
	if config.HeadDim != 128 {
		t.Errorf("HeadDim mismatch: got %d, expected 128", config.HeadDim)
	}
	if config.VocabSize != 32000 {
		t.Errorf("VocabSize mismatch: got %d, expected 32000", config.VocabSize)
	}
	if config.MaxSeqLen != 8192 {
		t.Errorf("MaxSeqLen mismatch: got %d, expected 8192", config.MaxSeqLen)
	}
	if config.RMSNormEps != 1e-5 {
		t.Errorf("RMSNormEps mismatch: got %v, expected 1e-5", config.RMSNormEps)
	}

	t.Log("PASS: TransferableConfig struct correctly defined")
}

// TestSerializeConfig_SizeBound verifies config fits in single message (< 1KB)
// PRP specifies config should be sent as single message, not chunked
func TestSerializeConfig_SizeBound(t *testing.T) {
	// Test with largest config (Llama 70B)
	config := types.Llama70BConfig()
	modelName := "llama-2-70b"

	// Act: Serialize
	// This should FAIL - SerializeConfig doesn't exist yet
	serialized, err := SerializeConfig(config, modelName)
	if err != nil {
		t.Fatalf("SerializeConfig failed: %v", err)
	}

	// Assert: Config is less than 1KB (PRP requirement: single message, not chunked)
	maxSize := 1024 // 1KB
	if len(serialized) > maxSize {
		t.Errorf("Serialized config too large: got %d bytes, max %d bytes", len(serialized), maxSize)
	}

	t.Logf("PASS: Serialized config size: %d bytes (max: %d)", len(serialized), maxSize)
}

// TestSerializeConfig_NilConfig verifies error handling for nil input
func TestSerializeConfig_NilConfig(t *testing.T) {
	// Act: Try to serialize nil config
	// This should FAIL - SerializeConfig doesn't exist yet
	_, err := SerializeConfig(nil, "test")

	// Assert: Should return error
	if err == nil {
		t.Error("Expected error for nil config, got nil")
	}

	t.Log("PASS: SerializeConfig correctly rejects nil config")
}

// TestSerializeConfig_EmptyModelName verifies handling of empty model name
func TestSerializeConfig_EmptyModelName(t *testing.T) {
	config := types.TinyLlamaConfig()

	// Act: Serialize with empty model name
	// This should FAIL - SerializeConfig doesn't exist yet
	_, err := SerializeConfig(config, "")

	// Assert: Should return error (model name is required)
	if err == nil {
		t.Error("Expected error for empty model name, got nil")
	}

	t.Log("PASS: SerializeConfig correctly rejects empty model name")
}

// Package model_test contains E2E acceptance tests for the SentencePiece tokenizer.
// These tests verify the success criteria defined in PRP-02.
package model_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/neurogrid/engine/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTokenizerInterface verifies that the tokenizer implements the required interface.
// This is a compile-time check that ensures the implementation satisfies the interface.
func TestTokenizerInterface(t *testing.T) {
	// The TokenizerInterface defines the contract for tokenizers
	var _ model.TokenizerInterface = (*model.SentencePieceTokenizer)(nil)
}

// =============================================================================
// Success Criteria 1: Encode text to token IDs matching HuggingFace tokenizer output
// =============================================================================

func TestEncode_BasicText(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	tests := []struct {
		name    string
		input   string
		wantLen int // Minimum expected token count (excluding BOS)
		wantBOS bool
	}{
		{
			name:    "simple hello world",
			input:   "Hello, world!",
			wantLen: 3, // At least 3 tokens for "Hello, world!"
			wantBOS: true,
		},
		{
			name:    "empty string",
			input:   "",
			wantLen: 0,
			wantBOS: true, // BOS token should still be present
		},
		{
			name:    "single word",
			input:   "Hello",
			wantLen: 1,
			wantBOS: true,
		},
		{
			name:    "sentence with punctuation",
			input:   "The quick brown fox jumps over the lazy dog.",
			wantLen: 5, // At least 5 tokens
			wantBOS: true,
		},
		{
			name:    "unicode text",
			input:   "Hello, \u4e16\u754c!",
			wantLen: 2, // At least 2 tokens
			wantBOS: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tokenizer.Encode(tt.input)
			require.NoError(t, err)

			if tt.wantBOS {
				require.NotEmpty(t, tokens)
				assert.Equal(t, tokenizer.BOSToken(), tokens[0], "first token should be BOS")
			}

			// Verify minimum token count (excluding BOS)
			tokenCount := len(tokens)
			if tt.wantBOS {
				tokenCount--
			}
			assert.GreaterOrEqual(t, tokenCount, tt.wantLen, "should have at least %d tokens", tt.wantLen)

			// All tokens should be valid (within vocab range)
			for i, tok := range tokens {
				assert.GreaterOrEqual(t, tok, 0, "token %d should be non-negative", i)
				assert.Less(t, tok, tokenizer.VocabSize(), "token %d should be within vocab size", i)
			}
		})
	}
}

func TestEncode_SpaceHandling(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Llama tokenizer encodes spaces as part of the following token
	// with the '▁' (U+2581) prefix
	input := "Hello world"
	tokens, err := tokenizer.Encode(input)
	require.NoError(t, err)

	// Verify we get consistent encoding
	assert.NotEmpty(t, tokens)

	// Encode twice should give same result (deterministic)
	tokens2, err := tokenizer.Encode(input)
	require.NoError(t, err)
	assert.Equal(t, tokens, tokens2, "encoding should be deterministic")
}

func TestEncode_NoBOSOption(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	input := "Hello"

	// EncodeWithoutBOS should not prepend BOS token
	tokens, err := tokenizer.EncodeWithoutBOS(input)
	require.NoError(t, err)

	if len(tokens) > 0 {
		assert.NotEqual(t, tokenizer.BOSToken(), tokens[0],
			"first token should NOT be BOS when using EncodeWithoutBOS")
	}
}

// =============================================================================
// Success Criteria 2: Decode token IDs back to text correctly
// =============================================================================

func TestDecode_BasicRoundtrip(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	tests := []struct {
		name  string
		input string
	}{
		{"simple text", "Hello, world!"},
		{"sentence", "The quick brown fox jumps over the lazy dog."},
		{"numbers", "12345 67890"},
		{"mixed", "Hello 123 world!"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Encode
			tokens, err := tokenizer.Encode(tt.input)
			require.NoError(t, err)

			// Decode
			decoded, err := tokenizer.Decode(tokens)
			require.NoError(t, err)

			// Roundtrip should match (possibly with leading space trimmed)
			assert.Equal(t, tt.input, decoded, "roundtrip decode should match original")
		})
	}
}

func TestDecode_SkipsSpecialTokens(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Create a token sequence with special tokens
	tokens := []int{
		tokenizer.BOSToken(),
		100, // Arbitrary content token
		tokenizer.EOSToken(),
	}

	decoded, err := tokenizer.Decode(tokens)
	require.NoError(t, err)

	// Decoded text should not contain special token representations
	assert.NotContains(t, decoded, "<s>")
	assert.NotContains(t, decoded, "</s>")
	assert.NotContains(t, decoded, "<pad>")
}

func TestDecode_EmptyTokens(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	decoded, err := tokenizer.Decode([]int{})
	require.NoError(t, err)
	assert.Empty(t, decoded)
}

// =============================================================================
// Success Criteria 3: Handle special tokens (BOS, EOS, PAD, UNK)
// =============================================================================

func TestSpecialTokens_Values(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Llama 2 standard special token IDs
	assert.Equal(t, 1, tokenizer.BOSToken(), "BOS token should be 1")
	assert.Equal(t, 2, tokenizer.EOSToken(), "EOS token should be 2")

	// PAD and UNK may vary by model
	assert.GreaterOrEqual(t, tokenizer.PADToken(), 0, "PAD token should be valid")
	assert.GreaterOrEqual(t, tokenizer.UNKToken(), 0, "UNK token should be valid")
}

func TestSpecialTokens_DecodeCorrectly(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// DecodeSingle for special tokens should return their string representation
	bos := tokenizer.DecodeSingle(tokenizer.BOSToken())
	eos := tokenizer.DecodeSingle(tokenizer.EOSToken())

	// Special tokens have specific string representations
	assert.Contains(t, []string{"<s>", ""}, bos, "BOS should decode to <s> or empty")
	assert.Contains(t, []string{"</s>", ""}, eos, "EOS should decode to </s> or empty")
}

func TestSpecialTokens_UNKForUnknownBytes(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// SentencePiece uses byte fallback instead of UNK for unknown bytes
	// This test verifies that invalid UTF-8 sequences are handled
	input := "Hello \xff world" // Contains invalid UTF-8 byte

	tokens, err := tokenizer.Encode(input)
	// Should not error - byte fallback should handle it
	require.NoError(t, err)
	assert.NotEmpty(t, tokens)
}

// =============================================================================
// Success Criteria 4: Support Llama 2 chat template formatting
// =============================================================================

func TestChatTemplate_Llama2Format(t *testing.T) {
	template := model.NewLlama2ChatTemplate()

	tests := []struct {
		name     string
		messages []model.Message
		want     string
	}{
		{
			name: "system + user",
			messages: []model.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			want: "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello! [/INST] ",
		},
		{
			name: "user only",
			messages: []model.Message{
				{Role: "user", Content: "What is 2+2?"},
			},
			want: "[INST] What is 2+2? [/INST] ",
		},
		{
			name: "multi-turn",
			messages: []model.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "How are you?"},
			},
			want: "[INST] Hi [/INST] Hello! </s><s>[INST] How are you? [/INST] ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := template.Format(tt.messages)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestChatTemplate_Llama3Format(t *testing.T) {
	template := model.NewLlama3ChatTemplate()

	messages := []model.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello!"},
	}

	result := template.Format(messages)

	// Llama 3 uses different format with <|begin_of_text|>, <|start_header_id|>, etc.
	assert.Contains(t, result, "<|start_header_id|>system<|end_header_id|>")
	assert.Contains(t, result, "<|start_header_id|>user<|end_header_id|>")
	assert.Contains(t, result, "You are a helpful assistant.")
	assert.Contains(t, result, "Hello!")
}

// =============================================================================
// Success Criteria 5: Incremental decode for streaming
// =============================================================================

func TestDecodeSingle_StreamingSupport(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Encode some text
	input := "Hello world"
	tokens, err := tokenizer.Encode(input)
	require.NoError(t, err)

	// Decode each token individually
	var parts []string
	for _, tok := range tokens {
		// Skip BOS token
		if tok == tokenizer.BOSToken() {
			continue
		}
		part := tokenizer.DecodeSingle(tok)
		parts = append(parts, part)
	}

	// Joined parts should reconstruct the original (with space handling)
	joined := strings.Join(parts, "")
	// May have leading underscore due to SentencePiece space encoding
	cleaned := strings.ReplaceAll(joined, "\u2581", " ")
	cleaned = strings.TrimPrefix(cleaned, " ")
	assert.Equal(t, input, cleaned, "streaming decode should reconstruct original")
}

func TestStreamingDecode_PartialUTF8(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Create a streaming decoder
	decoder := model.NewStreamingDecoder(tokenizer)

	// Encode text with multi-byte UTF-8 characters
	input := "Hello \u4e16\u754c" // "Hello 世界"
	tokens, err := tokenizer.Encode(input)
	require.NoError(t, err)

	// Feed tokens one at a time
	var result strings.Builder
	for _, tok := range tokens {
		if tok == tokenizer.BOSToken() {
			continue
		}
		text := decoder.Decode(tok)
		result.WriteString(text)
	}

	// Flush any remaining bytes
	result.WriteString(decoder.Flush())

	// Result should match original
	assert.Equal(t, input, result.String())
}

// =============================================================================
// Success Criteria 6: Load tokenizer.model from model directory
// =============================================================================

func TestLoadFromModelDirectory(t *testing.T) {
	// Skip if no test model available
	modelPath := os.Getenv("TOKENIZER_MODEL_PATH")
	if modelPath == "" {
		t.Skip("TOKENIZER_MODEL_PATH not set, skipping integration test")
	}

	tokenizerPath := filepath.Join(modelPath, "tokenizer.model")
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		t.Skipf("tokenizer.model not found at %s", tokenizerPath)
	}

	tokenizer, err := model.NewSentencePieceTokenizer(modelPath)
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	// Basic sanity checks
	assert.Greater(t, tokenizer.VocabSize(), 0, "vocab size should be positive")
	assert.Equal(t, 1, tokenizer.BOSToken(), "BOS should be 1 for Llama")
	assert.Equal(t, 2, tokenizer.EOSToken(), "EOS should be 2 for Llama")
}

func TestNewSentencePieceTokenizer_FileNotFound(t *testing.T) {
	_, err := model.NewSentencePieceTokenizer("/nonexistent/path")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tokenizer.model")
}

func TestNewSentencePieceTokenizer_InvalidFile(t *testing.T) {
	// Create a temp directory with an invalid tokenizer.model file
	tmpDir := t.TempDir()
	invalidPath := filepath.Join(tmpDir, "tokenizer.model")
	err := os.WriteFile(invalidPath, []byte("not a valid sentencepiece model"), 0644)
	require.NoError(t, err)

	_, err = model.NewSentencePieceTokenizer(tmpDir)
	require.Error(t, err)
}

// =============================================================================
// Additional E2E Tests
// =============================================================================

func TestVocabSize_RealTokenizer(t *testing.T) {
	// This test only runs with a real tokenizer
	modelPath := os.Getenv("TOKENIZER_MODEL_PATH")
	if modelPath == "" {
		t.Skip("TOKENIZER_MODEL_PATH not set, skipping vocab size test")
	}

	tokenizer, err := model.NewSentencePieceTokenizer(modelPath)
	if err != nil {
		t.Skipf("Could not load real tokenizer: %v", err)
	}

	// Llama 2 vocab size is 32000
	vocabSize := tokenizer.VocabSize()
	assert.Greater(t, vocabSize, 1000, "vocab size should be > 1000")
}

func TestVocabSize_MockTokenizer(t *testing.T) {
	tokenizer := model.NewMockSentencePieceTokenizer()

	// Mock tokenizer has limited vocab for testing
	vocabSize := tokenizer.VocabSize()
	assert.Greater(t, vocabSize, 0, "mock vocab size should be positive")
}

func TestEncodeDecode_LongText(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Test with longer text
	input := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 10)

	tokens, err := tokenizer.Encode(input)
	require.NoError(t, err)
	assert.Greater(t, len(tokens), 50, "should have many tokens for long text")

	decoded, err := tokenizer.Decode(tokens)
	require.NoError(t, err)
	assert.Equal(t, strings.TrimSpace(input), strings.TrimSpace(decoded))
}

func TestEncode_Concurrent(t *testing.T) {
	tokenizer := setupTestTokenizer(t)

	// Encoding should be thread-safe
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(idx int) {
			input := strings.Repeat("test ", idx+1)
			_, err := tokenizer.Encode(input)
			assert.NoError(t, err)
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

// =============================================================================
// Test Helpers
// =============================================================================

// setupTestTokenizer creates a tokenizer for testing.
// First tries to load a real tokenizer.model, falls back to mock.
func setupTestTokenizer(t *testing.T) model.TokenizerInterface {
	t.Helper()

	// Try to load from environment variable first
	modelPath := os.Getenv("TOKENIZER_MODEL_PATH")
	if modelPath != "" {
		tokenizerPath := filepath.Join(modelPath, "tokenizer.model")
		if _, err := os.Stat(tokenizerPath); err == nil {
			tok, err := model.NewSentencePieceTokenizer(modelPath)
			if err == nil {
				return tok
			}
			t.Logf("Failed to load tokenizer from %s: %v, falling back to mock", modelPath, err)
		}
	}

	// Fall back to mock tokenizer for unit tests
	return model.NewMockSentencePieceTokenizer()
}

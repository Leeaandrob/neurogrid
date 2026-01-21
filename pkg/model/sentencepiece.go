// Package model provides model loading and weight management for LLM inference.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// TokenizerInterface defines the contract for tokenizers.
// This interface is implemented by SentencePieceTokenizer.
type TokenizerInterface interface {
	// Encode converts text to token IDs, prepending BOS token.
	Encode(text string) ([]int, error)

	// EncodeWithoutBOS converts text to token IDs without BOS token.
	EncodeWithoutBOS(text string) ([]int, error)

	// Decode converts token IDs to text, skipping special tokens.
	Decode(tokens []int) (string, error)

	// DecodeSingle decodes a single token ID to its string representation.
	DecodeSingle(token int) string

	// VocabSize returns the vocabulary size.
	VocabSize() int

	// BOSToken returns the beginning-of-sequence token ID.
	BOSToken() int

	// EOSToken returns the end-of-sequence token ID.
	EOSToken() int

	// PADToken returns the padding token ID.
	PADToken() int

	// UNKToken returns the unknown token ID.
	UNKToken() int
}

// SentencePieceTokenizer implements tokenization using SentencePiece models.
// This tokenizer loads .model files from Llama model directories.
type SentencePieceTokenizer struct {
	vocab         map[string]int
	vocabByID     map[int]string
	vocabScores   []float32
	vocabSize     int
	bosToken      int
	eosToken      int
	padToken      int
	unkToken      int
	byteTokens    map[byte]int    // Byte fallback tokens
	specialTokens map[string]int  // Special tokens (</s>, <s>, etc.)
	isMock        bool            // true for mock tokenizer
}

// NewSentencePieceTokenizer creates a new tokenizer from a model directory.
// It expects to find tokenizer.model in the given directory.
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error) {
	tokenizerPath := filepath.Join(modelPath, "tokenizer.model")

	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer.model not found at %s", tokenizerPath)
	}

	// Parse the SentencePiece model
	model, err := ParseSentencePieceModel(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer.model: %w", err)
	}

	if len(model.Pieces) == 0 {
		return nil, errors.New("tokenizer.model contains no vocabulary pieces")
	}

	t := &SentencePieceTokenizer{
		vocab:         make(map[string]int),
		vocabByID:     make(map[int]string),
		vocabScores:   make([]float32, len(model.Pieces)),
		byteTokens:    make(map[byte]int),
		specialTokens: make(map[string]int),
		bosToken:      int(model.TrainerSpec.BosID),
		eosToken:      int(model.TrainerSpec.EosID),
		padToken:      int(model.TrainerSpec.PadID),
		unkToken:      int(model.TrainerSpec.UnkID),
		vocabSize:     len(model.Pieces),
	}

	// Build vocabulary maps
	for i, piece := range model.Pieces {
		t.vocab[piece.Piece] = i
		t.vocabByID[i] = piece.Piece
		t.vocabScores[i] = piece.Score

		// Track byte fallback tokens
		if piece.Type == PieceTypeByte && len(piece.Piece) == 6 {
			// Byte tokens are in format <0xXX>
			if strings.HasPrefix(piece.Piece, "<0x") && strings.HasSuffix(piece.Piece, ">") {
				hexStr := piece.Piece[3:5]
				var b byte
				if _, err := fmt.Sscanf(hexStr, "%02X", &b); err == nil {
					t.byteTokens[b] = i
				}
			}
		}
	}

	// Load special tokens from tokenizer.json if it exists
	tokenizerJSONPath := filepath.Join(modelPath, "tokenizer.json")
	if data, err := os.ReadFile(tokenizerJSONPath); err == nil {
		var config struct {
			AddedTokens []struct {
				ID      int    `json:"id"`
				Content string `json:"content"`
				Special bool   `json:"special"`
			} `json:"added_tokens"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			for _, added := range config.AddedTokens {
				if added.Special {
					t.specialTokens[added.Content] = added.ID
					// Also add to vocab maps if not already present
					if _, exists := t.vocab[added.Content]; !exists {
						t.vocab[added.Content] = added.ID
						t.vocabByID[added.ID] = added.Content
					}
				}
			}
			log.Printf("[SentencePieceTokenizer] Loaded %d special tokens from tokenizer.json", len(t.specialTokens))
			for tok, id := range t.specialTokens {
				log.Printf("[SentencePieceTokenizer]   %q -> %d", tok, id)
			}
		}
	}

	return t, nil
}

// Encode converts text to token IDs, prepending BOS token.
func (t *SentencePieceTokenizer) Encode(text string) ([]int, error) {
	tokens := []int{t.bosToken}
	if text == "" {
		return tokens, nil
	}
	encoded, err := t.encode(text)
	if err != nil {
		return nil, err
	}
	return append(tokens, encoded...), nil
}

// EncodeWithoutBOS converts text to token IDs without BOS token.
func (t *SentencePieceTokenizer) EncodeWithoutBOS(text string) ([]int, error) {
	return t.encode(text)
}

// encode is the internal encoding method using BPE algorithm.
// It handles special tokens by splitting them out before BPE encoding.
func (t *SentencePieceTokenizer) encode(text string) ([]int, error) {
	if t.isMock {
		return t.encodeMock(text)
	}

	if text == "" {
		return nil, nil
	}

	// Split text by special tokens and encode each segment
	var tokens []int
	remaining := text

	for len(remaining) > 0 {
		// Find the earliest occurring special token
		foundPos := -1
		var foundToken string
		var foundID int

		for tok, id := range t.specialTokens {
			pos := strings.Index(remaining, tok)
			if pos != -1 && (foundPos == -1 || pos < foundPos) {
				foundPos = pos
				foundToken = tok
				foundID = id
			}
		}

		if foundPos != -1 {
			// Encode text before the special token
			if foundPos > 0 {
				beforeText := remaining[:foundPos]
				beforeTokens := t.encodeBPE(beforeText)
				tokens = append(tokens, beforeTokens...)
			}
			// Add the special token
			tokens = append(tokens, foundID)
			// Move past the special token
			remaining = remaining[foundPos+len(foundToken):]
		} else {
			// No more special tokens, encode the rest
			restTokens := t.encodeBPE(remaining)
			tokens = append(tokens, restTokens...)
			break
		}
	}

	return tokens, nil
}

// encodeBPE encodes a text segment using BPE algorithm (no special token handling).
func (t *SentencePieceTokenizer) encodeBPE(text string) []int {
	if text == "" {
		return nil
	}

	// Normalize: add space prefix (SentencePiece adds dummy prefix)
	// Convert spaces to the SentencePiece space marker
	normalized := "\u2581" + strings.ReplaceAll(text, " ", "\u2581")

	// Start with character-level tokens
	symbols := t.textToSymbols(normalized)
	if len(symbols) == 0 {
		return nil
	}

	// Apply BPE merges
	t.applyBPEMerges(&symbols)

	// Convert symbols to token IDs
	tokens := make([]int, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			tokens = append(tokens, id)
		} else {
			// Byte fallback
			for _, b := range []byte(sym) {
				if byteID, ok := t.byteTokens[b]; ok {
					tokens = append(tokens, byteID)
				} else {
					tokens = append(tokens, t.unkToken)
				}
			}
		}
	}

	return tokens
}

// textToSymbols converts text to initial character/byte symbols.
func (t *SentencePieceTokenizer) textToSymbols(text string) []string {
	var symbols []string

	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		if r == utf8.RuneError {
			// Invalid UTF-8, use byte fallback
			symbols = append(symbols, string(text[i]))
			i++
		} else {
			symbols = append(symbols, text[i:i+size])
			i += size
		}
	}

	return symbols
}

// applyBPEMerges iteratively merges the highest-scoring pairs.
func (t *SentencePieceTokenizer) applyBPEMerges(symbols *[]string) {
	for len(*symbols) > 1 {
		// Find best merge
		bestScore := float32(-1e9)
		bestIdx := -1
		bestMerged := ""

		for i := 0; i < len(*symbols)-1; i++ {
			merged := (*symbols)[i] + (*symbols)[i+1]
			if score, exists := t.getScore(merged); exists {
				if score > bestScore {
					bestScore = score
					bestIdx = i
					bestMerged = merged
				}
			}
		}

		if bestIdx == -1 {
			// No more merges possible
			break
		}

		// Apply the merge
		newSymbols := make([]string, 0, len(*symbols)-1)
		for i := 0; i < len(*symbols); i++ {
			if i == bestIdx {
				newSymbols = append(newSymbols, bestMerged)
				i++ // Skip next symbol (merged)
			} else {
				newSymbols = append(newSymbols, (*symbols)[i])
			}
		}
		*symbols = newSymbols
	}
}

// getScore returns the vocabulary score for a token.
func (t *SentencePieceTokenizer) getScore(token string) (float32, bool) {
	if id, ok := t.vocab[token]; ok {
		return t.vocabScores[id], true
	}
	return 0, false
}

// encodeMock is the mock implementation for testing.
func (t *SentencePieceTokenizer) encodeMock(text string) ([]int, error) {
	var tokens []int
	remaining := text

	for len(remaining) > 0 {
		found := false

		// Handle space at the beginning
		if remaining[0] == ' ' {
			remaining = remaining[1:]
			if len(remaining) > 0 {
				// Try to match underscore+word tokens
				for length := min(len(remaining), 20); length > 0; length-- {
					candidate := "\u2581" + remaining[:length]
					if id, ok := t.vocab[candidate]; ok {
						tokens = append(tokens, id)
						remaining = remaining[length:]
						found = true
						break
					}
				}
				if !found {
					if id, ok := t.vocab["\u2581"]; ok {
						tokens = append(tokens, id)
					}
				}
			}
			continue
		}

		// Try to find longest matching token
		for length := min(len(remaining), 20); length > 0; length-- {
			candidate := remaining[:length]
			if id, ok := t.vocab[candidate]; ok {
				tokens = append(tokens, id)
				remaining = remaining[length:]
				found = true
				break
			}
			if id, ok := t.vocab["\u2581"+candidate]; ok {
				tokens = append(tokens, id)
				remaining = remaining[length:]
				found = true
				break
			}
		}

		if !found {
			char := string(remaining[0])
			if id, ok := t.vocab[char]; ok {
				tokens = append(tokens, id)
			} else {
				tokens = append(tokens, t.unkToken)
			}
			remaining = remaining[1:]
		}
	}

	return tokens, nil
}

// Decode converts token IDs to text, skipping special tokens.
func (t *SentencePieceTokenizer) Decode(tokens []int) (string, error) {
	var result strings.Builder
	for _, tok := range tokens {
		// Skip special tokens
		if tok == t.bosToken || tok == t.eosToken || tok == t.padToken {
			continue
		}
		if s, ok := t.vocabByID[tok]; ok {
			// Replace SentencePiece space marker with actual space
			s = strings.ReplaceAll(s, "\u2581", " ")
			result.WriteString(s)
		}
	}
	text := result.String()
	return strings.TrimPrefix(text, " "), nil
}

// DecodeSingle decodes a single token ID to its string representation.
func (t *SentencePieceTokenizer) DecodeSingle(token int) string {
	if s, ok := t.vocabByID[token]; ok {
		return s
	}
	return ""
}

// VocabSize returns the vocabulary size.
func (t *SentencePieceTokenizer) VocabSize() int {
	return t.vocabSize
}

// BOSToken returns the beginning-of-sequence token ID.
func (t *SentencePieceTokenizer) BOSToken() int {
	return t.bosToken
}

// EOSToken returns the end-of-sequence token ID.
func (t *SentencePieceTokenizer) EOSToken() int {
	return t.eosToken
}

// PADToken returns the padding token ID.
func (t *SentencePieceTokenizer) PADToken() int {
	return t.padToken
}

// UNKToken returns the unknown token ID.
func (t *SentencePieceTokenizer) UNKToken() int {
	return t.unkToken
}

// NewMockSentencePieceTokenizer creates a mock tokenizer for testing.
func NewMockSentencePieceTokenizer() *SentencePieceTokenizer {
	vocab := map[string]int{
		"<unk>":       0,
		"<s>":         1,
		"</s>":        2,
		"\u2581":      3,
		"H":           4,
		"e":           5,
		"l":           6,
		"o":           7,
		",":           8,
		"w":           9,
		"r":           10,
		"d":           11,
		"!":           12,
		"\u2581Hello": 13,
		"\u2581world": 14,
		"\u2581The":   15,
		"\u2581quick": 16,
		"\u2581brown": 17,
		"\u2581fox":   18,
		"\u2581jumps": 19,
		"\u2581over":  20,
		"\u2581the":   21,
		"\u2581lazy":  22,
		"\u2581dog":   23,
		".":           24,
		"1":           25,
		"2":           26,
		"3":           27,
		"4":           28,
		"5":           29,
		"6":           30,
		"7":           31,
		"8":           32,
		"9":           33,
		"0":           34,
		"\u4e16":      35, // Chinese char
		"\u754c":      36, // Chinese char
		"Hello":       37,
		"t":           38,
		"s":           39,
		"\u2581test":  40,
	}

	vocabByID := make(map[int]string, len(vocab))
	for k, v := range vocab {
		vocabByID[v] = k
	}

	// Build scores (not used in mock, but needed for interface)
	scores := make([]float32, len(vocab))
	// Sort by ID to assign scores
	type kv struct {
		k string
		v int
	}
	var sorted []kv
	for k, v := range vocab {
		sorted = append(sorted, kv{k, v})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].v < sorted[j].v
	})
	for i, item := range sorted {
		scores[item.v] = float32(-i) // Higher score for earlier tokens
	}

	return &SentencePieceTokenizer{
		vocab:         vocab,
		vocabByID:     vocabByID,
		vocabScores:   scores,
		vocabSize:     len(vocab),
		bosToken:      1,
		eosToken:      2,
		padToken:      0,
		unkToken:      0,
		byteTokens:    make(map[byte]int),
		specialTokens: make(map[string]int),
		isMock:        true,
	}
}

// StreamingDecoder handles incremental token decoding for streaming output.
// It buffers partial UTF-8 sequences that may span multiple tokens.
type StreamingDecoder struct {
	tokenizer TokenizerInterface
	buffer    []byte
}

// NewStreamingDecoder creates a new streaming decoder.
func NewStreamingDecoder(tokenizer TokenizerInterface) *StreamingDecoder {
	return &StreamingDecoder{
		tokenizer: tokenizer,
	}
}

// Decode decodes a single token and returns any complete UTF-8 text.
// Incomplete UTF-8 sequences are buffered for the next call.
func (d *StreamingDecoder) Decode(token int) string {
	piece := d.tokenizer.DecodeSingle(token)
	if piece == "" {
		return ""
	}

	// Replace SentencePiece space marker
	piece = strings.ReplaceAll(piece, "\u2581", " ")

	// Append to buffer
	d.buffer = append(d.buffer, []byte(piece)...)

	// Find the last valid UTF-8 boundary
	validLen := 0
	for i := 0; i < len(d.buffer); {
		r, size := decodeRune(d.buffer[i:])
		if r == -1 {
			// Incomplete UTF-8 sequence
			break
		}
		i += size
		validLen = i
	}

	if validLen == 0 {
		return ""
	}

	result := string(d.buffer[:validLen])
	d.buffer = d.buffer[validLen:]
	return result
}

// Flush returns any remaining buffered bytes as a string.
// Called at the end of streaming to get any partial content.
func (d *StreamingDecoder) Flush() string {
	if len(d.buffer) == 0 {
		return ""
	}
	result := string(d.buffer)
	d.buffer = nil
	return result
}

// decodeRune decodes the first UTF-8 rune from bytes.
// Returns -1 and 0 if the sequence is incomplete.
func decodeRune(b []byte) (r rune, size int) {
	if len(b) == 0 {
		return -1, 0
	}

	// Single-byte (ASCII)
	if b[0] < 0x80 {
		return rune(b[0]), 1
	}

	// Multi-byte sequence
	var expected int
	switch {
	case b[0]&0xE0 == 0xC0:
		expected = 2
	case b[0]&0xF0 == 0xE0:
		expected = 3
	case b[0]&0xF8 == 0xF0:
		expected = 4
	default:
		// Invalid leading byte
		return -1, 0
	}

	if len(b) < expected {
		// Incomplete sequence
		return -1, 0
	}

	// Validate continuation bytes
	for i := 1; i < expected; i++ {
		if b[i]&0xC0 != 0x80 {
			return -1, 0
		}
	}

	// Decode the rune
	switch expected {
	case 2:
		r = rune(b[0]&0x1F)<<6 | rune(b[1]&0x3F)
	case 3:
		r = rune(b[0]&0x0F)<<12 | rune(b[1]&0x3F)<<6 | rune(b[2]&0x3F)
	case 4:
		r = rune(b[0]&0x07)<<18 | rune(b[1]&0x3F)<<12 | rune(b[2]&0x3F)<<6 | rune(b[3]&0x3F)
	}

	return r, expected
}

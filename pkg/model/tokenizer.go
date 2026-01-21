// Package model provides model loading and weight management for LLM inference.
package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer implements text tokenization for Llama models.
// This is a simplified BPE tokenizer implementation.
type Tokenizer struct {
	vocab         map[string]int // token string -> id
	merges        []BPEMerge     // ordered merge rules
	vocabByID     map[int]string // id -> token string
	specialTokens map[string]int // special token string -> id
	bosToken      int
	eosToken      int
	padToken      int
	unkToken      int
	vocabSize     int
}

// BPEMerge represents a BPE merge rule.
type BPEMerge struct {
	A        string
	B        string
	Priority int
}

// NewTokenizer creates a new tokenizer from a tokenizer.json file.
func NewTokenizer(modelPath string) (*Tokenizer, error) {
	tokenizerPath := filepath.Join(modelPath, "tokenizer.json")

	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer.json: %w", err)
	}

	return NewTokenizerFromJSON(data)
}

// NewTokenizerFromJSON creates a tokenizer from JSON data.
func NewTokenizerFromJSON(data []byte) (*Tokenizer, error) {
	var config struct {
		Model struct {
			Type   string         `json:"type"`
			Vocab  map[string]int `json:"vocab"`
			Merges []string       `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer: %w", err)
	}

	t := &Tokenizer{
		vocab:         config.Model.Vocab,
		vocabByID:     make(map[int]string),
		specialTokens: make(map[string]int),
		bosToken:      1, // Default for Llama
		eosToken:      2, // Default for Llama
		padToken:      0, // Default for Llama
		unkToken:      0, // Default for Llama
	}

	// Build reverse vocab
	for token, id := range t.vocab {
		t.vocabByID[id] = token
	}

	// Parse merges
	for i, merge := range config.Model.Merges {
		parts := strings.SplitN(merge, " ", 2)
		if len(parts) == 2 {
			t.merges = append(t.merges, BPEMerge{
				A:        parts[0],
				B:        parts[1],
				Priority: i,
			})
		}
	}

	// Look for special tokens
	for _, added := range config.AddedTokens {
		t.vocabByID[added.ID] = added.Content
		t.vocab[added.Content] = added.ID
		// Store all special tokens for encoding
		if added.Special {
			t.specialTokens[added.Content] = added.ID
		}
		switch added.Content {
		case "<s>":
			t.bosToken = added.ID
		case "</s>":
			t.eosToken = added.ID
		case "<pad>":
			t.padToken = added.ID
		case "<unk>":
			t.unkToken = added.ID
		}
	}

	t.vocabSize = len(t.vocab)

	// DEBUG: Print loaded special tokens
	log.Printf("[Tokenizer] Loaded %d special tokens:", len(t.specialTokens))
	for tok, id := range t.specialTokens {
		log.Printf("[Tokenizer]   %q -> %d", tok, id)
	}

	return t, nil
}

// Encode tokenizes text into token IDs.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{t.bosToken}, nil
	}

	// DEBUG: Log input and special tokens count
	log.Printf("[Tokenizer.Encode] Input text (first 100 chars): %q", truncateString(text, 100))
	log.Printf("[Tokenizer.Encode] specialTokens map has %d entries", len(t.specialTokens))

	// Split text by special tokens and encode each segment
	var tokens []int
	tokens = append(tokens, t.bosToken)

	// Find and process special tokens
	remaining := text
	for len(remaining) > 0 {
		found := false
		foundPos := -1
		var foundToken string
		var foundID int

		// Find the earliest occurring special token
		for tok, id := range t.specialTokens {
			pos := strings.Index(remaining, tok)
			if pos != -1 && (foundPos == -1 || pos < foundPos) {
				foundPos = pos
				foundToken = tok
				foundID = id
				found = true
			}
		}

		if found {
			log.Printf("[Tokenizer.Encode] Found special token %q (id=%d) at pos %d", foundToken, foundID, foundPos)
			// Process text before the special token
			if foundPos > 0 {
				beforeText := remaining[:foundPos]
				words := t.preTokenize(beforeText)
				for _, word := range words {
					wordTokens := t.bpeEncode(word)
					tokens = append(tokens, wordTokens...)
				}
			}
			// Add the special token
			tokens = append(tokens, foundID)
			// Move past the special token
			remaining = remaining[foundPos+len(foundToken):]
		} else {
			// No more special tokens, encode the rest
			words := t.preTokenize(remaining)
			for _, word := range words {
				wordTokens := t.bpeEncode(word)
				tokens = append(tokens, wordTokens...)
			}
			break
		}
	}

	log.Printf("[Tokenizer.Encode] Final tokens: %v", tokens)
	return tokens, nil
}

// truncateString truncates a string to maxLen characters for logging.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// preTokenize splits text into preliminary tokens (words).
func (t *Tokenizer) preTokenize(text string) []string {
	var words []string
	var current strings.Builder

	for _, r := range text {
		if r == ' ' {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			// Llama tokenizer includes space as prefix
			current.WriteRune('▁') // Unicode character U+2581
		} else {
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

// bpeEncode applies BPE to a single word.
func (t *Tokenizer) bpeEncode(word string) []int {
	if len(word) == 0 {
		return nil
	}

	// Split into characters
	chars := t.splitToChars(word)
	if len(chars) == 0 {
		return nil
	}

	// If single char, look up directly
	if len(chars) == 1 {
		if id, ok := t.vocab[chars[0]]; ok {
			return []int{id}
		}
		return []int{t.unkToken}
	}

	// Apply merges iteratively
	pieces := make([]string, len(chars))
	copy(pieces, chars)

	for len(pieces) > 1 {
		bestMerge := -1
		bestPriority := len(t.merges)
		bestIdx := -1

		// Find the best merge
		for i := 0; i < len(pieces)-1; i++ {
			merged := pieces[i] + pieces[i+1]
			if _, ok := t.vocab[merged]; ok {
				// Find priority in merges
				for j, m := range t.merges {
					if m.A == pieces[i] && m.B == pieces[i+1] {
						if j < bestPriority {
							bestPriority = j
							bestMerge = j
							bestIdx = i
						}
						break
					}
				}
			}
		}

		if bestMerge == -1 {
			break
		}

		// Apply the merge
		newPieces := make([]string, 0, len(pieces)-1)
		for i := 0; i < len(pieces); i++ {
			if i == bestIdx {
				newPieces = append(newPieces, pieces[i]+pieces[i+1])
				i++ // Skip next
			} else {
				newPieces = append(newPieces, pieces[i])
			}
		}
		pieces = newPieces
	}

	// Convert pieces to IDs
	ids := make([]int, 0, len(pieces))
	for _, piece := range pieces {
		if id, ok := t.vocab[piece]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.unkToken)
		}
	}

	return ids
}

// splitToChars splits a string into UTF-8 characters as strings.
func (t *Tokenizer) splitToChars(s string) []string {
	var chars []string
	for len(s) > 0 {
		r, size := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			// Invalid UTF-8, skip byte
			s = s[1:]
			continue
		}
		chars = append(chars, string(r))
		s = s[size:]
	}
	return chars
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(tokens []int) (string, error) {
	var result strings.Builder

	for _, id := range tokens {
		// Skip special tokens
		if id == t.bosToken || id == t.eosToken || id == t.padToken {
			continue
		}

		if token, ok := t.vocabByID[id]; ok {
			// Replace sentencepiece space marker with actual space
			token = strings.ReplaceAll(token, "▁", " ")
			result.WriteString(token)
		}
	}

	text := result.String()
	// Trim leading space that might be from first token
	text = strings.TrimPrefix(text, " ")

	return text, nil
}

// DecodeToken decodes a single token ID to its string representation.
func (t *Tokenizer) DecodeToken(id int) string {
	if token, ok := t.vocabByID[id]; ok {
		return token
	}
	return "<unk>"
}

// EOSToken returns the end-of-sequence token ID.
func (t *Tokenizer) EOSToken() int {
	return t.eosToken
}

// BOSToken returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOSToken() int {
	return t.bosToken
}

// PADToken returns the padding token ID.
func (t *Tokenizer) PADToken() int {
	return t.padToken
}

// UNKToken returns the unknown token ID.
func (t *Tokenizer) UNKToken() int {
	return t.unkToken
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}

// GetVocab returns a copy of the vocabulary map.
func (t *Tokenizer) GetVocab() map[string]int {
	vocab := make(map[string]int, len(t.vocab))
	for k, v := range t.vocab {
		vocab[k] = v
	}
	return vocab
}

// TopTokens returns the most common tokens (by ID).
func (t *Tokenizer) TopTokens(n int) []string {
	type idToken struct {
		id    int
		token string
	}

	pairs := make([]idToken, 0, len(t.vocabByID))
	for id, token := range t.vocabByID {
		pairs = append(pairs, idToken{id, token})
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].id < pairs[j].id
	})

	if n > len(pairs) {
		n = len(pairs)
	}

	result := make([]string, n)
	for i := 0; i < n; i++ {
		result[i] = pairs[i].token
	}
	return result
}

// CreateMockTokenizer creates a mock tokenizer for testing.
func CreateMockTokenizer() *Tokenizer {
	vocab := map[string]int{
		"<unk>":  0,
		"<s>":    1,
		"</s>":   2,
		"▁Hello": 3,
		"▁world": 4,
		"!":      5,
		"▁":      6,
		"▁The":   7,
		"▁quick": 8,
		"▁brown": 9,
		"▁fox":   10,
	}

	vocabByID := make(map[int]string)
	for k, v := range vocab {
		vocabByID[v] = k
	}

	return &Tokenizer{
		vocab:     vocab,
		vocabByID: vocabByID,
		bosToken:  1,
		eosToken:  2,
		padToken:  0,
		unkToken:  0,
		vocabSize: len(vocab),
	}
}

// CreateMockTokenizerJSON creates mock tokenizer.json content.
func CreateMockTokenizerJSON() []byte {
	config := map[string]interface{}{
		"model": map[string]interface{}{
			"type": "BPE",
			"vocab": map[string]int{
				"<unk>":  0,
				"<s>":    1,
				"</s>":   2,
				"▁Hello": 3,
				"▁world": 4,
				"!":      5,
				"▁":      6,
			},
			"merges": []string{},
		},
		"added_tokens": []map[string]interface{}{
			{"id": 0, "content": "<unk>", "special": true},
			{"id": 1, "content": "<s>", "special": true},
			{"id": 2, "content": "</s>", "special": true},
		},
	}

	data, _ := json.Marshal(config)
	return data
}

// SerializeTokenizer serializes vocab to binary format for fast loading.
func (t *Tokenizer) SerializeBinary() []byte {
	var buf []byte

	// Write vocab size
	sizeBuf := make([]byte, 4)
	binary.LittleEndian.PutUint32(sizeBuf, uint32(len(t.vocab)))
	buf = append(buf, sizeBuf...)

	// Write special tokens
	binary.LittleEndian.PutUint32(sizeBuf, uint32(t.bosToken))
	buf = append(buf, sizeBuf...)
	binary.LittleEndian.PutUint32(sizeBuf, uint32(t.eosToken))
	buf = append(buf, sizeBuf...)
	binary.LittleEndian.PutUint32(sizeBuf, uint32(t.padToken))
	buf = append(buf, sizeBuf...)
	binary.LittleEndian.PutUint32(sizeBuf, uint32(t.unkToken))
	buf = append(buf, sizeBuf...)

	// Write vocab entries
	for token, id := range t.vocab {
		// Write ID
		binary.LittleEndian.PutUint32(sizeBuf, uint32(id))
		buf = append(buf, sizeBuf...)

		// Write token length and bytes
		tokenBytes := []byte(token)
		binary.LittleEndian.PutUint32(sizeBuf, uint32(len(tokenBytes)))
		buf = append(buf, sizeBuf...)
		buf = append(buf, tokenBytes...)
	}

	return buf
}

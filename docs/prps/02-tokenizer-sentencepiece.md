# PRP: Tokenizer (SentencePiece Integration)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | SentencePiece Tokenizer |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Short-Term (Critical Path) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 9/10 |

---

## Discovery Summary

### Initial Task Analysis

Implement tokenization using SentencePiece to convert text to tokens and back. Llama models use SentencePiece BPE tokenizer with a 32000 vocabulary.

### User Clarifications Received

- **Question**: Use CGO bindings or pure Go?
- **Answer**: Pure Go preferred for portability, CGO as fallback
- **Impact**: Will use go-sentencepiece library

### Missing Requirements Identified

- Chat template formatting (Llama 2 chat format)
- Special token handling (BOS, EOS, PAD)
- Streaming decode for SSE

---

## Goal

Implement `pkg/model/tokenizer.go` that can encode text to token IDs and decode token IDs back to text using Llama's SentencePiece model.

## Why

- **Critical path**: Cannot process text input without tokenization
- **Chat format**: Need proper formatting for instruction-tuned models
- **Streaming**: Decode must work incrementally for SSE streaming

## What

### Success Criteria

- [ ] Encode text to token IDs matching HuggingFace tokenizer output
- [ ] Decode token IDs back to text correctly
- [ ] Handle special tokens (BOS, EOS, PAD, UNK)
- [ ] Support Llama 2 chat template formatting
- [ ] Incremental decode for streaming
- [ ] Load tokenizer.model from model directory

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: None - new component
- **External research needed**: Yes - SentencePiece format, Go libraries
- **Knowledge gaps identified**: BPE algorithm details, chat template format

### Documentation & References

```yaml
- url: https://github.com/google/sentencepiece
  why: SentencePiece format and algorithm

- url: https://github.com/pkoukk/go-sentencepiece
  why: Pure Go SentencePiece implementation

- url: https://huggingface.co/docs/transformers/main/en/chat_templating
  why: Llama chat template format

- file: pkg/inference/engine.go
  why: Integration point for tokenization
```

### Current Codebase tree

```
pkg/
├── model/
│   ├── loader.go        # Model weights (from PRP-01)
│   └── (tokenizer.go)   # NEW - to be created
└── inference/
    └── engine.go        # Uses tokenizer
```

### Desired Codebase tree

```
pkg/model/
├── loader.go           # Model weights
├── tokenizer.go        # Tokenizer interface and implementation
├── chat_template.go    # Chat formatting (Llama 2 format)
└── vocab.go            # Vocabulary management
```

### Known Gotchas

```go
// CRITICAL: Llama uses BOS token at start, EOS at end
// CRITICAL: Byte-fallback for unknown characters (not UNK token)
// CRITICAL: Spaces are encoded as '▁' (U+2581)
// CRITICAL: Chat template varies between Llama 2 and Llama 3
```

---

## Implementation Blueprint

### Data Models

```go
// pkg/model/tokenizer.go

type Tokenizer interface {
    // Encode converts text to token IDs
    Encode(text string) ([]int, error)

    // Decode converts token IDs to text
    Decode(tokens []int) (string, error)

    // DecodeSingle decodes a single token (for streaming)
    DecodeSingle(token int) string

    // Vocab returns vocabulary size
    VocabSize() int

    // Special tokens
    BOSToken() int
    EOSToken() int
    PADToken() int
}

type SentencePieceTokenizer struct {
    model       *sentencepiece.Model
    vocabSize   int
    bosToken    int
    eosToken    int
    padToken    int
    unkToken    int
    byteTokens  map[byte]int  // Byte fallback tokens
}

// pkg/model/chat_template.go

type ChatTemplate interface {
    Format(messages []Message) string
}

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type Llama2ChatTemplate struct{}
type Llama3ChatTemplate struct{}
```

### Task List

```yaml
Task 1: Add go-sentencepiece dependency
  MODIFY go.mod:
    - ADD: github.com/pkoukk/go-sentencepiece v0.0.0-latest
  RUN: go mod tidy

Task 2: Create tokenizer interface
  CREATE pkg/model/tokenizer.go:
    - Tokenizer interface definition
    - NewTokenizer factory function
    - Error types for tokenization

Task 3: Implement SentencePiece tokenizer
  MODIFY pkg/model/tokenizer.go:
    - SentencePieceTokenizer struct
    - Load from tokenizer.model file
    - Encode with BOS prefix
    - Decode with proper handling

Task 4: Implement streaming decode
  MODIFY pkg/model/tokenizer.go:
    - DecodeSingle for incremental output
    - Handle partial UTF-8 sequences
    - Buffer incomplete bytes

Task 5: Create chat template
  CREATE pkg/model/chat_template.go:
    - Llama2ChatTemplate implementation
    - Llama3ChatTemplate implementation
    - Template selection based on model config

Task 6: Integrate with inference engine
  MODIFY pkg/inference/engine.go:
    - Add tokenizer field
    - Use in Generate method
    - Apply chat template in prefill

Task 7: Add tests
  CREATE tests/model/tokenizer_test.go:
    - Test encode/decode roundtrip
    - Test special tokens
    - Test chat template formatting
    - Compare with HuggingFace output

Task 8: Add vocab management
  CREATE pkg/model/vocab.go:
    - Token ID to string mapping
    - Special token constants
    - Byte fallback handling
```

### Per-Task Pseudocode

```go
// Task 3: SentencePiece implementation
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error) {
    // PATTERN: Load .model file
    model, err := sentencepiece.NewModelFromFile(modelPath)
    if err != nil {
        return nil, fmt.Errorf("load tokenizer: %w", err)
    }

    return &SentencePieceTokenizer{
        model:    model,
        vocabSize: model.VocabSize(),
        bosToken: 1,  // <s>
        eosToken: 2,  // </s>
        padToken: 0,  // <pad> or <unk>
    }, nil
}

func (t *SentencePieceTokenizer) Encode(text string) ([]int, error) {
    // PATTERN: Prepend BOS, encode, no EOS (added by sampler)
    tokens := []int{t.bosToken}
    encoded := t.model.Encode(text)
    tokens = append(tokens, encoded...)
    return tokens, nil
}

// Task 4: Streaming decode
func (t *SentencePieceTokenizer) DecodeSingle(token int) string {
    // PATTERN: Decode single token, handle byte fallback
    // GOTCHA: May return partial UTF-8, caller must buffer
    piece := t.model.IdToPiece(token)
    // Replace ▁ with space
    return strings.ReplaceAll(piece, "▁", " ")
}

// Task 5: Chat template
func (t *Llama2ChatTemplate) Format(messages []Message) string {
    // PATTERN: Llama 2 chat format
    // <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>
    var b strings.Builder

    for i, msg := range messages {
        switch msg.Role {
        case "system":
            b.WriteString("[INST] <<SYS>>\n")
            b.WriteString(msg.Content)
            b.WriteString("\n<</SYS>>\n\n")
        case "user":
            if i > 0 && messages[i-1].Role != "system" {
                b.WriteString("<s>[INST] ")
            }
            b.WriteString(msg.Content)
            b.WriteString(" [/INST] ")
        case "assistant":
            b.WriteString(msg.Content)
            b.WriteString(" </s>")
        }
    }

    return b.String()
}
```

### Integration Points

```yaml
INFERENCE:
  - add to: pkg/inference/engine.go
  - field: tokenizer Tokenizer
  - usage: Encode prompt, decode output tokens

API:
  - add to: api/handlers.go
  - usage: Apply chat template before inference

CONFIG:
  - add to: cmd/neurogrid/main.go
  - flag: --tokenizer-path (default: model_path/tokenizer.model)
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./pkg/model/...
go vet ./pkg/model/...
golangci-lint run ./pkg/model/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/model/tokenizer_test.go

# Expected: All tests pass
```

### Level 3: Comparison Test

```python
# scripts/compare_tokenizer.py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # Compare with Go output
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./pkg/model/...`
- [ ] No linting errors: `golangci-lint run ./pkg/model/...`
- [ ] Encode output matches HuggingFace tokenizer
- [ ] Decode roundtrip is lossless
- [ ] Special tokens handled correctly
- [ ] Chat template produces correct format
- [ ] Streaming decode works incrementally

---

## Anti-Patterns to Avoid

- ❌ Don't hardcode vocab size (varies by model)
- ❌ Don't forget BOS token at start
- ❌ Don't add EOS token in encode (sampler does this)
- ❌ Don't assume all text is ASCII
- ❌ Don't ignore byte fallback tokens

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Encode throughput | > 100k tokens/sec |
| Decode throughput | > 1M tokens/sec |
| Memory usage | < 50 MB for vocab |

---

## Dependencies

```go
// go.mod additions
require (
    github.com/pkoukk/go-sentencepiece v0.0.0-20240101000000-abcdef123456
)
```

---

**PRP Confidence Score: 9/10**

**Rationale**:
- +3: Well-documented SentencePiece format
- +2: Existing Go library available
- +2: Clear integration points
- +2: Chat template format documented
- -1: Byte fallback edge cases

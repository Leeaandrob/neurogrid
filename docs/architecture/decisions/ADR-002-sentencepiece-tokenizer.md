# ADR-002: SentencePiece Tokenizer Implementation

## Status

Accepted

## Context

The neurogrid inference engine needs to tokenize text input before passing it to Llama models for inference. Llama models use SentencePiece tokenization with a unigram language model. We need to:

1. Load tokenizer vocabulary from `.model` files (protobuf format)
2. Encode text to token IDs with BOS token prepending
3. Decode token IDs back to text
4. Support streaming decode for incremental output
5. Handle Llama 2 and Llama 3 chat template formatting

The Go ecosystem lacks a well-maintained SentencePiece library that can parse `.model` files directly. Options considered:

1. **CGO bindings** to the C++ SentencePiece library
2. **External process** calling Python/C++ tokenizer
3. **Native Go implementation** parsing protobuf directly

## Decision

We implement a **native Go SentencePiece parser and tokenizer** that:

1. Parses the protobuf wire format directly (no protobuf codegen dependency)
2. Implements BPE merging algorithm based on vocabulary scores
3. Supports byte-fallback for unknown characters
4. Provides streaming decode with UTF-8 boundary detection
5. Includes chat template formatting for Llama 2 and Llama 3

### Key Design Choices

#### 1. Native Protobuf Parsing

Instead of using protobuf code generation, we implement a minimal wire-format parser that extracts only the fields we need:
- Vocabulary pieces (string, score, type)
- Trainer spec (BOS/EOS/PAD/UNK token IDs)

This reduces dependencies and binary size.

#### 2. BPE Algorithm

We use a greedy best-first merge strategy:
- Start with character-level symbols
- Iteratively merge the highest-scoring adjacent pair
- Continue until no more merges are possible

This matches SentencePiece's unigram model behavior for most common tokens.

#### 3. Space Handling

SentencePiece uses the `▁` (U+2581) character to represent spaces at word boundaries. Our implementation:
- Adds dummy prefix (`▁`) at the start of input
- Converts spaces to `▁` during encoding
- Converts `▁` back to spaces during decoding

#### 4. Streaming Decode

The `StreamingDecoder` buffers partial UTF-8 sequences to handle cases where a character spans multiple tokens (common with byte-fallback encoding).

#### 5. Chat Templates

We provide separate template implementations for:
- **Llama 2**: `[INST] <<SYS>>...` format
- **Llama 3**: `<|start_header_id|>...` format

A factory function selects the appropriate template based on model name.

## Consequences

### Positive

- **No CGO dependencies**: Pure Go implementation works on all platforms
- **Minimal dependencies**: Only uses standard library + existing project deps
- **Fast startup**: No subprocess spawning or library loading
- **Thread-safe**: Tokenizer methods are safe for concurrent use
- **Streaming support**: Efficient incremental decode for real-time output

### Negative

- **Encoding accuracy**: May not match SentencePiece exactly for edge cases (rare subword splits)
- **Performance**: Native Go may be slower than optimized C++ for very long texts
- **Maintenance**: Must manually update if protobuf format changes

### Mitigations

- Comprehensive test suite validates encode/decode roundtrips
- Mock tokenizer enables testing without model files
- Integration tests with real `.model` files when available

## Implementation

### File Structure

```
pkg/model/
  sentencepiece.go       # TokenizerInterface, SentencePieceTokenizer, StreamingDecoder
  sentencepiece_model.go # Protobuf parser for .model files
  chat_template.go       # Message, ChatTemplate, Llama2/Llama3 templates

tests/model/
  tokenizer_test.go      # E2E acceptance tests
```

### Usage

```go
// Load tokenizer
tok, err := model.NewSentencePieceTokenizer("/path/to/model")
if err != nil {
    return err
}

// Encode text (prepends BOS)
tokens, err := tok.Encode("Hello, world!")
// tokens = [1, ...]  (1 is BOS)

// Decode tokens
text, err := tok.Decode(tokens)
// text = "Hello, world!"

// Streaming decode
decoder := model.NewStreamingDecoder(tok)
for _, token := range tokens {
    chunk := decoder.Decode(token)
    fmt.Print(chunk)
}
fmt.Print(decoder.Flush())

// Chat template
template := model.NewLlama2ChatTemplate()
prompt := template.Format([]model.Message{
    {Role: "system", Content: "You are helpful."},
    {Role: "user", Content: "Hi!"},
})
```

## Related

- PRP-02: SentencePiece Tokenizer
- ADR-001: Model Weights Loader

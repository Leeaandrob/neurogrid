# ADR-004: SSE Token Streaming Implementation

## Status

Accepted

## Date

2026-01-21

## Context

The NeuroGrid inference engine needs to support streaming token delivery for the `/v1/chat/completions` endpoint to match OpenAI's API behavior when `stream: true` is set. Users expect to see tokens as they are generated rather than waiting for the entire response.

### Requirements

1. Stream tokens via Server-Sent Events (SSE) when `stream: true`
2. Match OpenAI's SSE message format exactly
3. Send `[DONE]` marker at end of stream
4. Handle client disconnect gracefully
5. Include usage stats in final chunk (optional)
6. Achieve time-to-first-token < 500ms (excluding model latency)

### Alternatives Considered

1. **WebSockets**: More complex, bidirectional (not needed), requires different client code
2. **Long Polling**: Higher latency, more server overhead
3. **Server-Sent Events**: Matches OpenAI standard, simple HTTP/1.1 compatible

## Decision

We implement SSE streaming following OpenAI's exact format with dedicated streaming utilities.

### Architecture

```
api/
├── server.go       # HTTP handlers, routes to streaming handler
├── stream.go       # NEW: SSEWriter and StreamState utilities
├── types.go        # Existing types + ChatCompletionChunk
```

### Key Components

1. **SSEWriter**: Wraps `http.ResponseWriter` for SSE protocol compliance
   - Sets required headers (Content-Type, Cache-Control, Connection, X-Accel-Buffering)
   - Formats events as `data: {json}\n\n`
   - Flushes after each event
   - Handles `[DONE]` termination

2. **StreamState**: Maintains consistent state across chunks
   - Single ID for all chunks in a stream
   - Single timestamp for all chunks
   - Factory methods for role, content, and final chunks

### SSE Format (OpenAI Compatible)

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":123,"model":"llama-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":123,"model":"llama-7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":123,"model":"llama-7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

### Critical Implementation Details

1. **Headers before first write**: SSE headers must be set before any data is written
2. **Flush after each event**: Ensures immediate delivery to client
3. **Space after "data:"**: OpenAI uses `data: ` not `data:`
4. **Double newline**: Events separated by `\n\n`
5. **[DONE] is literal**: Not JSON, literal string `[DONE]`
6. **Consistent IDs**: All chunks share the same ID and timestamp

## Consequences

### Positive

- OpenAI API compatibility for streaming responses
- Better user experience with visible token generation
- Low time-to-first-token for perceived performance
- Clean separation of SSE utilities from business logic
- Testable streaming with dedicated E2E tests

### Negative

- Simulated streaming (generates full response, then streams words)
- Real token-by-token streaming requires engine changes (future work)
- Additional complexity in error handling mid-stream

### Neutral

- 10ms delay between chunks for simulation (remove when real streaming added)
- Usage stats not included in streaming (optional per OpenAI spec)

## Implementation Notes

### Files Modified

- `api/server.go`: Refactored to use SSEWriter and StreamState
- `api/stream.go`: New file with SSE utilities
- `api/types.go`: Already had ChatCompletionChunk types

### Tests Added

- `tests/e2e/stream_test.go`: 22 comprehensive E2E tests covering:
  - SSE format compliance
  - OpenAI format matching
  - [DONE] marker
  - Client disconnect handling
  - Concurrent streams
  - Time-to-first-token
  - Error handling

### Future Work

1. **Real streaming**: Modify inference engine to support `GenerateStream` with callback
2. **Usage in streaming**: Add token counting during stream
3. **Backpressure**: Handle slow clients

## References

- [OpenAI Streaming API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream)
- [Server-Sent Events Spec](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [PRP-04: SSE Streaming](../prps/04-sse-streaming.md)

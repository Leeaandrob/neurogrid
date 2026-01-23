# Task Breakdown: PRP-004 Performance Optimization & Distributed Inference Fix

**PRP Reference**: `docs/prps/PRP-004-performance-optimization-distributed-fix.md`
**Created**: 2026-01-23
**Estimated Total Effort**: Medium (2-3 days)

---

## Phase 1: Performance Optimization (Priority: HIGH)

### Task 1.1: Remove Debug Logging from gpu_lmhead.go
**Effort**: Quick (30 minutes)
**Priority**: P0 - Immediate impact
**Dependencies**: None

**Description**:
Remove all debug logging that performs GPU→CPU copies in production code.

**Acceptance Criteria**:
- Given: `gpu_lmhead.go` contains debug logging with GPU→CPU copies
- When: Debug lines are removed or guarded
- Then: No extra GPU→CPU copies occur during Forward()

**Files to Modify**:
- `pkg/inference/gpu_lmhead.go`

**Changes**:
1. Remove lines 97-102 (input hidden debug)
2. Remove lines 151-159 (normalized output debug)
3. Remove lines 216-232 (top-5 logits calculation)

**Validation**:
```bash
go build -tags cuda ./pkg/inference/...
go test -tags cuda ./pkg/inference/... -run TestLMHead
```

---

### Task 1.2: Preallocate GPU Buffers
**Effort**: Short (2-4 hours)
**Priority**: P0 - Largest performance impact
**Dependencies**: Task 1.1

**Description**:
Preallocate reusable GPU buffers in `GPULMHead` struct instead of allocating per-token.

**Acceptance Criteria**:
- Given: GPULMHead allocates GPU memory every Forward() call
- When: Buffers are preallocated at construction
- Then: Forward() reuses existing buffers, no cudaMalloc/cudaFree per token

**Files to Modify**:
- `pkg/inference/gpu_lmhead.go`

**Changes**:
1. Add buffer fields to GPULMHead struct:
   ```go
   hiddenBuffer     unsafe.Pointer
   logitsBuffer     unsafe.Pointer
   normalizedBuffer unsafe.Pointer
   ```

2. Allocate in NewGPULMHeadWithNorm:
   ```go
   lmHead.hiddenBuffer, _ = bindings.AllocOnDevice(uint64(hiddenSize*2), 0)
   lmHead.logitsBuffer, _ = bindings.AllocOnDevice(uint64(vocabSize*2), 0)
   lmHead.normalizedBuffer, _ = bindings.AllocOnDevice(uint64(hiddenSize*2), 0)
   ```

3. Modify Forward() to use preallocated buffers

4. Add cleanup in Close():
   ```go
   bindings.FreeOnDevice(h.hiddenBuffer, 0)
   bindings.FreeOnDevice(h.logitsBuffer, 0)
   bindings.FreeOnDevice(h.normalizedBuffer, 0)
   ```

**Validation**:
```bash
go build -tags cuda ./pkg/inference/...
go test -tags cuda ./pkg/inference/... -run TestLMHead
# Performance test - should show <100ms for LMHead.Forward
```

---

### Task 1.3: GPU-Based FP16→FP32 Conversion
**Effort**: Medium (4-6 hours)
**Priority**: P1 - Significant performance impact
**Dependencies**: Task 1.2

**Description**:
Move FP16→FP32 conversion from CPU loop to GPU kernel.

**Acceptance Criteria**:
- Given: Forward() converts 32K FP16 values to FP32 in CPU loop
- When: GPU kernel performs conversion
- Then: Conversion completes in <5ms instead of 30-80ms

**Files to Create**:
- `gpu/kernels/convert.cu` (CUDA kernel)

**Files to Modify**:
- `gpu/bindings/gpu.go` (Go binding)
- `pkg/inference/gpu_lmhead.go` (use new binding)

**Changes**:

1. Create CUDA kernel (`gpu/kernels/convert.cu`):
   ```cuda
   #include <cuda_fp16.h>

   extern "C" __global__ void fp16_to_fp32_kernel(
       const half* input,
       float* output,
       int size
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < size) {
           output[idx] = __half2float(input[idx]);
       }
   }
   ```

2. Add Go binding:
   ```go
   func ConvertFP16ToFP32(output, input unsafe.Pointer, size int) error
   ```

3. Replace CPU loop in Forward():
   ```go
   // Allocate FP32 output buffer (preallocated)
   bindings.ConvertFP16ToFP32(logitsGPUFP32, logitsGPU, h.vocabSize)
   bindings.CopyFromDeviceRaw(unsafe.Pointer(&logits[0]), logitsGPUFP32, uint64(h.vocabSize*4))
   ```

**Validation**:
```bash
go build -tags cuda ./gpu/bindings/...
go test -tags cuda ./gpu/bindings/... -run TestConvert
go test -tags cuda ./pkg/inference/... -run TestLMHead
```

---

## Phase 2: Distributed Inference Fix (Priority: HIGH)

### Task 2.1: Add Protocol Debug Logging
**Effort**: Quick (30 minutes)
**Priority**: P0 - Required for diagnosis
**Dependencies**: None

**Description**:
Add detailed logging to trace activation/response flow between coordinator and worker.

**Acceptance Criteria**:
- Given: Distributed inference times out with no visibility
- When: Logging is added to protocol handlers
- Then: Can see exactly where message flow breaks

**Files to Modify**:
- `pkg/inference/remote_executor.go`
- `cmd/worker/main.go`

**Changes**:

1. In `remote_executor.go` RemoteLayerExecutor.Forward():
   ```go
   log.Printf("[COORD→] Sending activation: layer=%d, reqID=%d, peer=%s, bytes=%d",
       layerID, requestID, rle.targetPeerID, len(hidden))

   // after send
   log.Printf("[COORD] Waiting for response: reqID=%d, timeout=%s",
       requestID, rle.defaultTimeout)

   // after receive
   log.Printf("[COORD←] Received response: reqID=%d, bytes=%d",
       requestID, len(response.Data))
   ```

2. In `cmd/worker/main.go` handleActivation():
   ```go
   log.Printf("[WORKER←] Activation received: layer=%d, reqID=%d, from=%s, bytes=%d",
       msg.LayerID, msg.RequestID, msg.From, len(msg.Data))
   ```

3. In `cmd/worker/main.go` sendResponse():
   ```go
   log.Printf("[WORKER→] Sending response: reqID=%d, layer=%d, to=%s, bytes=%d",
       requestID, layerID, peerID, len(data))
   ```

**Validation**:
Run distributed test and examine logs on both servers.

---

### Task 2.2: Verify Protocol Response Routing
**Effort**: Short (1-2 hours)
**Priority**: P0 - Critical for distributed
**Dependencies**: Task 2.1

**Description**:
Verify and fix protocol response routing from worker back to coordinator.

**Acceptance Criteria**:
- Given: Worker sends response via SendResponse()
- When: Coordinator's protocol receives response message
- Then: Response is routed to WaitForResponse() via pendingResponses channel

**Files to Analyze**:
- `p2p/protocol.go`

**Analysis Steps**:
1. Verify `handleExtendedMessage` has case for `MsgTypeResponse`
2. Verify `handleResponse` routes to `pendingResponses[requestID]` channel
3. Verify `WaitForResponse` creates channel in `pendingResponses` map
4. Verify channel is cleaned up after response received or timeout

**Potential Fixes**:
- If MsgTypeResponse case missing in handleExtendedMessage: Add it
- If pendingResponses map not initialized: Initialize in NewProtocol
- If channel cleanup missing: Add defer cleanup

**Validation**:
```bash
go test -v ./p2p/... -run TestProtocol
# Distributed test with logging from Task 2.1
```

---

### Task 2.3: End-to-End Distributed Test
**Effort**: Medium (2-3 hours)
**Priority**: P1 - Validates fixes
**Dependencies**: Tasks 2.1, 2.2

**Description**:
Run full distributed inference test across two GH200 servers.

**Acceptance Criteria**:
- Given: Coordinator on Server 1, Worker on Server 2
- When: Inference request is made
- Then: Both servers process layers, response returned without timeout

**Test Steps**:

1. **Server 2 (Worker)**:
   ```bash
   ./worker --model models/llama-2-13b --port 9000 --gpu 0 \
     --model-name llama-2-13b
   # Note peer ID from output
   ```

2. **Server 1 (Coordinator)**:
   ```bash
   ./neurogrid serve --model models/llama-2-13b --min-peers 1 --http 8089 \
     --p2p 9001 --skip-weight-transfer \
     --bootstrap /ip4/192.222.58.78/tcp/9000/p2p/<WORKER_PEER_ID>
   ```

3. **Test Request**:
   ```bash
   curl localhost:8089/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"llama-2-13b","messages":[{"role":"user","content":"Hello"}],"max_tokens":5}'
   ```

4. **Verify Logs**:
   - Server 1: `[COORD→]`, `[COORD]`, `[COORD←]` messages
   - Server 2: `[WORKER←]`, `[WORKER→]` messages
   - No timeout errors

**Validation**:
Response received within 30s, both GPUs utilized.

---

## Phase 3: Performance Validation (Priority: MEDIUM)

### Task 3.1: Benchmark Single-Node Performance
**Effort**: Short (1 hour)
**Priority**: P2 - Validates Phase 1
**Dependencies**: Tasks 1.1, 1.2, 1.3

**Description**:
Measure and document single-node performance improvements.

**Acceptance Criteria**:
- Given: All performance optimizations applied
- When: Single token generation benchmarked
- Then: Latency < 1s/token on GH200

**Benchmark Script**:
```bash
#!/bin/bash
# benchmark_single_node.sh

echo "Starting coordinator in local mode..."
./neurogrid serve --model models/llama-2-13b --min-peers 0 --http 8089 &
PID=$!
sleep 10

echo "Warming up..."
curl -s localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-2-13b","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}' > /dev/null

echo "Benchmarking 10 tokens..."
for i in {1..10}; do
  START=$(date +%s%N)
  curl -s localhost:8089/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"llama-2-13b","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}' > /dev/null
  END=$(date +%s%N)
  ELAPSED=$(( (END - START) / 1000000 ))
  echo "Token $i: ${ELAPSED}ms"
done

kill $PID
```

**Expected Results**:
- Before: ~10,000ms/token
- After: <500ms/token

---

## Task Summary

| Task | Effort | Priority | Dependencies | Status |
|------|--------|----------|--------------|--------|
| 1.1 Remove Debug Logging | Quick | P0 | None | Pending |
| 1.2 Preallocate GPU Buffers | Short | P0 | 1.1 | Pending |
| 1.3 GPU FP16→FP32 Conversion | Medium | P1 | 1.2 | Pending |
| 2.1 Add Protocol Debug Logging | Quick | P0 | None | Pending |
| 2.2 Verify Protocol Response Routing | Short | P0 | 2.1 | Pending |
| 2.3 End-to-End Distributed Test | Medium | P1 | 2.1, 2.2 | Pending |
| 3.1 Benchmark Single-Node | Short | P2 | 1.1-1.3 | Pending |

---

## Execution Order

**Day 1 (Focus: Quick Wins)**:
1. Task 1.1 - Remove debug logging (immediate 50-100ms savings)
2. Task 2.1 - Add protocol debug logging (enable diagnosis)
3. Task 2.2 - Verify protocol routing (fix distributed)

**Day 2 (Focus: Major Performance)**:
4. Task 1.2 - Preallocate GPU buffers (200-500ms savings)
5. Task 2.3 - End-to-end distributed test (validate fixes)

**Day 3 (Focus: Polish)**:
6. Task 1.3 - GPU FP16→FP32 conversion (30-80ms savings)
7. Task 3.1 - Benchmark and document results

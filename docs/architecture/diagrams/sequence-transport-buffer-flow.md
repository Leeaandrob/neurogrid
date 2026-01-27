# Sequence Diagram: Transport Buffer Flow (Phase 2)

## Overview

This document describes the end-to-end flow of activation data through the transport layer using buffer pools, from Coordinator to Worker to GPU. This represents the Phase 2 integration of CUDA pinned memory optimization.

## End-to-End Flow: Coordinator to Worker GPU

```mermaid
sequenceDiagram
    autonumber
    participant C as Coordinator
    participant Net as libp2p Network
    participant PR as Protocol (Worker)
    participant A as pinnedPoolAdapter
    participant PP as PinnedBufferPool
    participant W as Worker
    participant E as CUDALayerExecutor
    participant GPU as GPU Device

    Note over C,GPU: Forward Pass - Distributed Inference

    C->>Net: SendActivation(layerID, seqID, data)
    Net-->>PR: Incoming stream

    Note over PR,PP: Buffer Pool Allocation

    PR->>PR: Read header (msgType, layerID, seqID, dataLen)
    PR->>A: Get(dataLen)
    A->>PP: Get()

    alt Pool has buffer
        PP-->>A: PinnedBuffer
        A->>A: AsSlice()[:dataLen]
        A-->>PR: pinnedBuffer
    else Pool exhausted
        A->>A: make([]byte, dataLen)
        A-->>PR: regularBuffer (fallback)
    end

    PR->>PR: io.ReadFull(stream, buffer)

    Note over PR,W: Copy and Return Buffer

    PR->>PR: msgData = make([]byte, len(data))
    PR->>PR: copy(msgData, data)
    PR->>A: Put(buffer)
    Note right of A: No-op for adapter

    PR->>W: handleActivation(TensorMessage)

    Note over W,GPU: GPU Execution

    W->>E: Forward(layerID, hidden, position)
    E->>E: Check preallocated buffers

    alt Has preallocated buffers
        E->>GPU: CopyToDeviceRaw(inputGPU, hidden)
        Note right of GPU: DMA if pinned memory
        E->>GPU: LayerForward(outputGPU, inputGPU, weights)
        E->>E: CopyFromDeviceRaw(output, outputGPU)
    else Dynamic allocation
        E->>GPU: AllocateTensor(inputTensor)
        E->>GPU: CopyToDeviceRaw(inputTensor.Data, hidden)
        E->>GPU: LayerForward(outputTensor, inputTensor)
        E->>GPU: CopyFromDeviceRaw(output, outputTensor.Data)
        E->>GPU: FreeTensor(inputTensor, outputTensor)
    end

    E-->>W: output

    Note over W,C: Send Response

    W->>PR: SendResponse(peerID, layerID, seqID, output)
    PR->>Net: Write header + data
    Net-->>C: Response stream
```

## Worker Initialization Flow

```mermaid
sequenceDiagram
    autonumber
    participant Main as main()
    participant W as Worker
    participant GPU as GPU Bindings
    participant PP as PinnedBufferPool
    participant A as pinnedPoolAdapter
    participant PR as Protocol
    participant CUDA as CUDA Runtime

    Main->>W: NewWorker(config)
    W-->>Main: worker

    Main->>W: Start()

    Note over W,CUDA: GPU Initialization

    W->>GPU: SetDevice(gpuID)
    W->>GPU: GetDeviceInfo()
    GPU-->>W: deviceInfo (name, VRAM)

    Note over W,CUDA: Pinned Pool Initialization

    W->>W: initPinnedPool()
    W->>PP: NewPinnedBufferPoolWithDefaults()
    PP->>CUDA: cudaHostAlloc(16KB) x 32
    CUDA-->>PP: pinnedPtrs
    PP-->>W: pool

    Note over W,PR: Protocol Configuration

    W->>PR: NewProtocol(host)
    W->>A: Create pinnedPoolAdapter{pool}
    W->>PR: SetBufferPool(adapter)
    PR->>PR: p.bufferPool = adapter

    Note over W,Main: Ready for Requests

    W-->>Main: nil (success)
```

## Protocol Message Handling Detail

```mermaid
sequenceDiagram
    autonumber
    participant S as libp2p Stream
    participant PR as Protocol
    participant A as pinnedPoolAdapter
    participant PP as PinnedBufferPool
    participant H as TensorHandler

    Note over S,H: handleExtendedMessage Flow

    S->>PR: handleStream(stream)
    PR->>S: io.ReadFull(firstByte)

    alt Extended message (0x01-0x09)
        PR->>S: io.ReadFull(restHeader, 24B)
        PR->>PR: DecodeExtendedHeader()
    else Traced message (0x11-0x12)
        PR->>S: io.ReadFull(restHeader, 49B)
        PR->>PR: DecodeExtendedHeader + TraceContext
    end

    Note over PR,PP: Buffer Pool Usage

    PR->>PR: mu.RLock() - get pool reference
    PR->>A: Get(dataLen)

    A->>PP: Get()
    alt Buffer available
        PP->>PP: <-buffers channel
        PP-->>A: PinnedBuffer
        A-->>PR: buffer.AsSlice()[:dataLen]
    else Pool empty
        A-->>PR: make([]byte, dataLen)
    end

    PR->>S: io.ReadFull(stream, buffer)

    Note over PR,H: Copy and Dispatch

    PR->>PR: msgData = make([]byte, len(data))
    PR->>PR: copy(msgData, data)
    PR->>A: Put(buffer)

    PR->>PR: Create TensorMessage{msgData, ...}

    alt MsgTypeActivation
        PR->>H: activationHandler(msg)
    else MsgTypeResponse
        PR->>PR: pendingResponses[requestID] <- msg
        PR->>H: responseHandler(msg)
    else MsgTypeWeights
        PR->>H: weightsHandler(layerID, chunk, ...)
    end
```

## CUDALayerExecutor Forward Flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant E as CUDALayerExecutor
    participant B as GPU Bindings
    participant GPU as GPU Device

    C->>E: Forward(layerID, hidden, position)

    E->>E: mu.RLock()
    E->>E: weights = layerWeights[layerID]
    E->>E: cache = kvCaches[layerID]
    E->>E: mu.RUnlock()

    alt No weights loaded
        E-->>C: error("layer not loaded")
    end

    Note over E,GPU: Buffer Strategy Decision

    E->>E: expectedSize = hiddenSize * 2

    alt inputGPU != nil AND len(hidden) <= bufferSize
        Note right of E: Preallocated path (Phase 2)

        E->>E: inputTensor.Data = inputGPU
        E->>B: CopyToDeviceRaw(inputGPU, hidden)
        B->>GPU: cudaMemcpy H2D
        Note right of GPU: DMA (fast)

        E->>E: outputTensor.Data = outputGPU
        E->>B: LayerForward(output, input, weights, cache)
        B->>GPU: CUDA kernel execution

        E->>B: CopyFromDeviceRaw(output, outputGPU)
        B->>GPU: cudaMemcpy D2H

    else Dynamic allocation (fallback)
        Note right of E: Legacy path

        E->>B: AllocateTensor(inputTensor)
        B->>GPU: cudaMalloc

        E->>B: CopyToDeviceRaw(input.Data, hidden)
        B->>GPU: cudaMemcpy H2D

        E->>B: AllocateTensor(outputTensor)
        B->>GPU: cudaMalloc

        E->>B: LayerForward(...)
        B->>GPU: CUDA kernel

        E->>B: CopyFromDeviceRaw(output, output.Data)
        B->>GPU: cudaMemcpy D2H

        E->>B: FreeTensor(inputTensor)
        E->>B: FreeTensor(outputTensor)
        B->>GPU: cudaFree x 2
    end

    E-->>C: output, k, v, nil
```

## Shutdown Flow

```mermaid
sequenceDiagram
    autonumber
    participant Main as main()
    participant W as Worker
    participant PP as PinnedBufferPool
    participant E as CUDALayerExecutor
    participant CUDA as CUDA Runtime

    Main->>W: Shutdown()
    W->>W: cancel() - stop context

    Note over W,CUDA: Free Pinned Memory

    W->>PP: Close()
    PP->>PP: close(buffers channel)

    loop Drain channel
        PP->>PP: buf = <-buffers
        PP->>CUDA: cudaFreeHost(buf.Ptr)
    end

    W->>W: pinnedPool = nil

    Note over W,CUDA: Free GPU Resources

    W->>E: Close() (if exists)
    E->>CUDA: FreeOnDevice(inputGPU)
    E->>CUDA: FreeOnDevice(outputGPU)
    E->>CUDA: FreeLayerWeights(...)
    E->>CUDA: FreeKVCache(...)

    W->>W: host.Close()
    W-->>Main: nil
```

## Performance Comparison

```mermaid
gantt
    title Transfer Path Latency Comparison
    dateFormat X
    axisFormat %L ms

    section Without Pools
    make([]byte, dataLen)     :a1, 0, 30
    io.ReadFull               :a2, after a1, 100
    copy(msgData, data)       :a3, after a2, 20
    cudaMemcpy H2D (staged)   :a4, after a3, 150
    cudaMalloc (input)        :a5, after a4, 80
    cudaMalloc (output)       :a6, after a5, 80
    LayerForward              :a7, after a6, 200
    cudaFree x2               :a8, after a7, 60
    GC pressure               :a9, after a8, 50

    section With Pools (Phase 2)
    pool.Get()                :b1, 0, 5
    io.ReadFull (pinned)      :b2, after b1, 100
    copy(msgData, data)       :b3, after b2, 20
    cudaMemcpy H2D (DMA)      :b4, after b3, 120
    Use preallocated GPU      :b5, after b4, 5
    LayerForward              :b6, after b5, 200
    pool.Put() (no-op)        :b7, after b6, 1
```

## Key Integration Points

| Component | Buffer Pool Integration | Notes |
|-----------|------------------------|-------|
| Protocol.handleExtendedMessage | Uses p.bufferPool.Get/Put | Copies data before Put |
| Protocol.handleTracedMessage | Uses p.bufferPool.Get/Put | Same pattern as extended |
| P2PTransport.handleExtendedStream | Uses t.bufferPool.Get/Put | transport package version |
| P2PTransport.handleLegacyStream | Uses t.bufferPool.Get/Put | Backward compatible |
| CUDALayerExecutor.Forward | Reuses inputGPU/outputGPU | GPU-side optimization |
| Worker.initPinnedPool | Creates PinnedBufferPool | Called during GPU init |
| Worker.Shutdown | Closes PinnedBufferPool | Frees CUDA pinned memory |

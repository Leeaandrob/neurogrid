# Sequence Diagram: Pinned Memory Activation Transfer

## Overview

This document describes the sequence of operations for activation transfers using CUDA pinned memory in distributed LLM inference.

## Main Flow: Coordinator to Worker Transfer

```mermaid
sequenceDiagram
    autonumber
    participant C as Coordinator
    participant CP as Coordinator Pool
    participant CT as P2PTransport
    participant Net as libp2p Network
    participant WT as Worker Transport
    participant WP as Worker Pool
    participant W as Worker
    participant GPU as GPU Device

    Note over C,GPU: Forward Pass - Layer N on Coordinator

    C->>C: LayerForward() completes
    C->>C: Activation ready (FP16)

    Note over C,CP: Get Buffer from Pool

    C->>CP: Get(activationSize)
    alt Pool has buffer
        CP-->>C: pinnedBuffer
    else Pool exhausted
        CP->>CP: AllocPinnedMemory(size)
        CP-->>C: newBuffer (fallback)
    end

    C->>C: Copy activation to buffer

    Note over C,Net: Network Transfer

    C->>CT: SendActivation(layerID, seqID, data)
    CT->>Net: Write header (layerID, seqID, len)
    CT->>Net: Write data (pinnedBuffer)
    Net-->>WT: TCP delivery

    Note over C,CP: Return Buffer to Pool

    C->>CP: Put(pinnedBuffer)
    alt Pool not full
        CP->>CP: Return to channel
    else Pool full
        CP->>CP: FreePinnedMemory()
    end

    Note over WT,GPU: Worker Receives Activation

    WT->>WT: Read header
    WT->>WP: Get(dataLen)
    WP-->>WT: pinnedBuffer

    WT->>WT: io.ReadFull(stream, pinnedBuffer)

    WT->>W: handleActivation(layerID, pinnedBuffer)

    Note over W,GPU: GPU Transfer (DMA)

    W->>GPU: CopyToDeviceRaw(gpuPtr, pinnedBuffer)
    Note right of GPU: DMA transfer<br/>(no CPU involvement)

    W->>WP: Put(pinnedBuffer)

    Note over W,GPU: Continue Inference

    W->>GPU: LayerForward(N+1)
    GPU-->>W: Result activation
```

## Buffer Pool Initialization

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant Pool as PinnedBufferPool
    participant CUDA as CUDA Runtime

    App->>Pool: NewPinnedBufferPool(16KB, 32)

    Pool->>Pool: Create buffered channel(32)

    loop Preallocate 32 buffers
        Pool->>CUDA: cudaHostAlloc(16KB, PORTABLE)
        CUDA-->>Pool: pinnedPtr
        Pool->>Pool: PinnedBuffer{Ptr, Size}
        Pool->>Pool: buffers <- buffer
    end

    Pool-->>App: *PinnedBufferPool

    Note over App,Pool: Pool ready with 32 preallocated buffers
```

## Buffer Get/Put Cycle

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Code
    participant Pool as PinnedBufferPool
    participant Chan as buffers channel
    participant CUDA as CUDA Runtime

    Note over Client,Pool: Get Buffer

    Client->>Pool: Get()
    Pool->>Chan: select receive

    alt Buffer available
        Chan-->>Pool: PinnedBuffer
        Pool-->>Client: buffer
    else Channel empty (pool exhausted)
        Pool->>CUDA: AllocPinnedMemory(bufSize)
        CUDA-->>Pool: pinnedPtr
        Pool->>Pool: Create PinnedBuffer
        Pool-->>Client: newBuffer
        Note right of Pool: Fallback allocation
    end

    Note over Client,Pool: Use Buffer

    Client->>Client: Use buffer for transfer

    Note over Client,Pool: Return Buffer

    Client->>Pool: Put(buffer)
    Pool->>Chan: select send

    alt Channel not full
        Chan->>Chan: Receive buffer
        Note right of Chan: Buffer returned to pool
    else Channel full
        Pool->>CUDA: FreePinnedMemory(ptr)
        Note right of CUDA: Overflow freed
    end
```

## Error Handling: Allocation Failure

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant Pool as Transport Pool
    participant Bindings as GPU Bindings
    participant CUDA as CUDA Runtime

    App->>Pool: NewPinnedBufferPool(size, count)

    Pool->>Bindings: NewPinnedBufferPool(size, count)

    loop Preallocate buffers
        Bindings->>CUDA: cudaHostAlloc()
        alt Success
            CUDA-->>Bindings: ptr
            Bindings->>Bindings: Add to pool
        else Failure (OOM, no CUDA)
            CUDA-->>Bindings: error
            Bindings->>Bindings: Close() - free allocated
            Bindings-->>Pool: error
            Pool-->>App: error
            Note over App: Fallback to DefaultBufferPool
        end
    end
```

## Multi-GPU Transfer with Pinned Memory

```mermaid
sequenceDiagram
    autonumber
    participant GPU0 as GPU 0 (4090)
    participant Host as Host (Pinned)
    participant GPU1 as GPU 1 (2080Ti)

    Note over GPU0,GPU1: cudaHostAllocPortable ensures visibility

    GPU0->>GPU0: LayerForward completes
    GPU0->>Host: cudaMemcpy D2H
    Note right of Host: Activation in pinned memory

    Host->>Host: Network transfer (simulated)

    Host->>GPU1: cudaMemcpy H2D
    Note right of GPU1: DMA transfer<br/>Pinned memory visible<br/>from GPU1

    GPU1->>GPU1: LayerForward continues
```

## Pool Shutdown Sequence

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant Pool as PinnedBufferPool
    participant Chan as buffers channel
    participant CUDA as CUDA Runtime

    App->>Pool: Close()

    Pool->>Pool: Set closed = true

    Pool->>Chan: close(buffers)

    loop Drain channel
        Chan-->>Pool: PinnedBuffer
        Pool->>CUDA: FreePinnedMemory(ptr)
    end

    Pool-->>App: nil (success)

    Note over App,Pool: Subsequent Get() returns nil
```

## Concurrent Request Handling

```mermaid
sequenceDiagram
    autonumber
    participant R1 as Request 1
    participant R2 as Request 2
    participant Pool as BufferPool
    participant Chan as buffers channel

    par Request 1
        R1->>Pool: Get()
        Pool->>Chan: <- buffers
        Chan-->>Pool: buffer1
        Pool-->>R1: buffer1
    and Request 2
        R2->>Pool: Get()
        Pool->>Chan: <- buffers
        Chan-->>Pool: buffer2
        Pool-->>R2: buffer2
    end

    Note over R1,R2: Both requests have<br/>independent buffers

    par Process
        R1->>R1: Process with buffer1
    and
        R2->>R2: Process with buffer2
    end

    par Return
        R1->>Pool: Put(buffer1)
        Pool->>Chan: buffers <- buffer1
    and
        R2->>Pool: Put(buffer2)
        Pool->>Chan: buffers <- buffer2
    end
```

## Performance Timeline Comparison

```mermaid
gantt
    title Transfer Latency Comparison (64KB)
    dateFormat X
    axisFormat %L ms

    section Regular Memory
    Allocate buffer     :a1, 0, 50
    Copy to buffer      :a2, after a1, 30
    CPU staging copy    :a3, after a2, 100
    PCIe transfer       :a4, after a3, 150
    Free buffer (GC)    :a5, after a4, 20

    section Pinned Memory
    Pool Get            :b1, 0, 5
    Copy to buffer      :b2, after b1, 30
    DMA transfer        :b3, after b2, 120
    Pool Put            :b4, after b3, 5
```

## Key Observations

1. **Lock-free Pool Access**: Channel-based design enables concurrent Get/Put without locks
2. **DMA Benefit**: Pinned memory bypasses CPU staging buffer
3. **Graceful Fallback**: Pool exhaustion triggers allocation, not failure
4. **Multi-GPU Visibility**: `cudaHostAllocPortable` flag enables cross-device DMA
5. **Clean Shutdown**: Close() drains and frees all buffers safely

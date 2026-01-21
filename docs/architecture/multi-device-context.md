# Multi-Device Context Manager - Architecture Document

## Overview

The Multi-Device Context Manager is responsible for managing multiple GPU devices in the NeuroGrid distributed inference engine. It provides initialization, P2P access management, and cross-device memory operations.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Go Application Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│                     gpu/bindings/gpu.go                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │ InitMultiGPU() │  │ CrossDeviceCopy│  │ AllocOnDevice()│        │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │
│          │                   │                   │                  │
├──────────┼───────────────────┼───────────────────┼──────────────────┤
│          │         CGO Interface                 │                  │
├──────────┼───────────────────┼───────────────────┼──────────────────┤
│          │    gpu/cuda/multi_device.cu           │                  │
│  ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐        │
│  │cuda_multi_init │  │cuda_cross_     │  │cuda_alloc_on_  │        │
│  │                │  │device_copy     │  │device          │        │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │
│          │                   │                   │                  │
├──────────┼───────────────────┼───────────────────┼──────────────────┤
│          │         CUDA Runtime API              │                  │
│  ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐        │
│  │cudaSetDevice   │  │cudaMemcpyPeer  │  │cudaMalloc      │        │
│  │cudaDeviceCan-  │  │cudaMemcpy      │  │cudaFree        │        │
│  │AccessPeer      │  │                │  │                │        │
│  └────────────────┘  └────────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Structures

### DeviceContext

Holds per-GPU state for multi-device operations:

```go
type DeviceContext struct {
    DeviceID       int            // CUDA device ID
    TotalMemory    uint64         // Total GPU memory in bytes
    UsedMemory     uint64         // Currently used memory
    ComputeStream  unsafe.Pointer // CUDA stream for compute operations
    TransferStream unsafe.Pointer // CUDA stream for data transfers
    PeerAccess     []bool         // P2P access matrix row for this device
}
```

### MultiDeviceManagerInfo

Provides global multi-device manager state:

```go
type MultiDeviceManagerInfo struct {
    NumDevices        int      // Number of initialized devices
    DeviceIDs         []int    // List of CUDA device IDs
    StagingBufferSize uint64   // Size of host staging buffer for non-P2P transfers
    Initialized       bool     // Whether the manager is initialized
}
```

## API Reference

### Initialization

| Function | Description |
|----------|-------------|
| `InitMultiGPU(deviceIDs []int)` | Initialize multi-GPU context with specified devices |
| `ShutdownMultiGPU()` | Release all multi-GPU resources |

### Device Context

| Function | Description |
|----------|-------------|
| `GetDeviceContext(deviceID int)` | Get context for a specific device |
| `GetMultiDeviceManagerInfo()` | Get global manager information |

### P2P Access

| Function | Description |
|----------|-------------|
| `CanAccessPeer(src, dst int)` | Check if P2P access is available |
| `GetP2PAccessMatrix()` | Get full P2P access matrix |

### Memory Operations

| Function | Description |
|----------|-------------|
| `AllocOnDevice(size, deviceID)` | Allocate memory on specific device |
| `FreeOnDevice(ptr, deviceID)` | Free memory on specific device |
| `CrossDeviceCopy(dst, dstDev, src, srcDev, size)` | Copy data between devices |

## P2P Transfer Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cross-Device Copy Decision                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ CrossDeviceCopy │                                            │
│  │    called       │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐    Yes    ┌─────────────────────────┐  │
│  │ Same device?       ├──────────►│ cudaMemcpy D2D          │  │
│  └────────┬───────────┘           └─────────────────────────┘  │
│           │ No                                                   │
│           ▼                                                      │
│  ┌────────────────────┐    Yes    ┌─────────────────────────┐  │
│  │ P2P available?     ├──────────►│ cudaMemcpyPeer          │  │
│  │ (p2p_matrix[s][d]) │           │ (Direct GPU-to-GPU)     │  │
│  └────────┬───────────┘           └─────────────────────────┘  │
│           │ No                                                   │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Staged Transfer via Host:                                │   │
│  │ 1. cudaMemcpy(staging_buf, src, D2H)                    │   │
│  │ 2. cudaMemcpy(dst, staging_buf, H2D)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling

All functions return errors following Go conventions:
- `nil` on success
- Descriptive error on failure

Common error conditions:
- Multi-device context not initialized
- Invalid device ID
- Device not in context
- Memory allocation failure
- P2P access not available (falls back to staging)

## Thread Safety

The Go stub implementation uses `sync.RWMutex` for thread-safe access:
- Read operations use `RLock()`/`RUnlock()`
- Write operations use `Lock()`/`Unlock()`

The CUDA implementation relies on CUDA's internal synchronization for:
- Stream operations
- Memory allocations
- P2P transfers

## Performance Considerations

1. **P2P Access**: When available, provides highest throughput for cross-device transfers
2. **Staging Buffer**: 64 MB default size, chunked transfers for larger data
3. **Dedicated Streams**: Separate streams for compute and transfer enable overlap
4. **Memory Pool**: Future optimization for frequent allocations

## Files

| File | Description |
|------|-------------|
| `gpu/bindings/gpu.go` | Go bindings with CGO calls (cuda build tag) |
| `gpu/bindings/gpu_stub.go` | Stub implementation for testing (!cuda build tag) |
| `gpu/bindings/gpu.h` | C header with function declarations |
| `gpu/cuda/multi_device.cu` | CUDA implementation |
| `gpu/cuda/multi_device.h` | CUDA header with types |
| `tests/e2e/multi_device_test.go` | E2E test suite |

## Related Tasks

- TASK-001: Multi-Device Context Manager (this document)
- TASK-002: P2P Topology Detection
- TASK-003: Cross-Device Memory Operations
- TASK-004: Device-Specific Allocation
- TASK-005: Go Multi-Device Bindings

---

*Generated by TDD E2E Workflow - TASK-001 Multi-Device Context Manager*

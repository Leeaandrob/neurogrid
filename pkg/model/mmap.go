// Package model provides model loading and weight management for LLM inference.
package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// MmapLoader provides memory-mapped access to SafeTensors model weights.
// This is more efficient than WeightLoader for large models that exceed
// available RAM, as it maps files directly into virtual memory without
// copying data.
type MmapLoader struct {
	basePath  string
	index     map[string]TensorInfo
	mmapFiles map[string]*MmapFile
	mu        sync.RWMutex
}

// MmapFile represents a memory-mapped file.
type MmapFile struct {
	Path       string
	File       *os.File
	Data       []byte
	Size       int64
	DataOffset int64 // Offset where tensor data starts (after header)
}

// NewMmapLoader creates a new memory-mapped weight loader.
func NewMmapLoader(basePath string) (*MmapLoader, error) {
	loader := &MmapLoader{
		basePath:  basePath,
		index:     make(map[string]TensorInfo),
		mmapFiles: make(map[string]*MmapFile),
	}

	// Try to load model.safetensors.index.json for sharded models
	indexPath := filepath.Join(basePath, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); err == nil {
		if err := loader.loadShardedIndex(indexPath); err != nil {
			return nil, fmt.Errorf("failed to load sharded index: %w", err)
		}
		return loader, nil
	}

	// Try to load single model.safetensors
	singlePath := filepath.Join(basePath, "model.safetensors")
	if _, err := os.Stat(singlePath); err == nil {
		if err := loader.loadSingleFile(singlePath); err != nil {
			return nil, fmt.Errorf("failed to load single file: %w", err)
		}
		return loader, nil
	}

	return nil, fmt.Errorf("no SafeTensors model found at %s", basePath)
}

// loadShardedIndex loads the index for a sharded SafeTensors model.
func (l *MmapLoader) loadShardedIndex(indexPath string) error {
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return err
	}

	var shardIndex struct {
		Metadata  map[string]interface{} `json:"metadata"`
		WeightMap map[string]string      `json:"weight_map"`
	}

	if err := json.Unmarshal(data, &shardIndex); err != nil {
		return fmt.Errorf("failed to parse shard index: %w", err)
	}

	// Group tensors by file
	fileToTensors := make(map[string][]string)
	for tensor, file := range shardIndex.WeightMap {
		fileToTensors[file] = append(fileToTensors[file], tensor)
	}

	// Load and mmap each shard file
	for file := range fileToTensors {
		shardPath := filepath.Join(l.basePath, file)
		if err := l.loadSingleFile(shardPath); err != nil {
			return fmt.Errorf("failed to load shard %s: %w", file, err)
		}
	}

	return nil
}

// loadSingleFile loads and memory-maps a single SafeTensors file.
func (l *MmapLoader) loadSingleFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}

	// Get file size
	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	fileSize := stat.Size()

	// Memory-map the file
	data, err := unix.Mmap(int(f.Fd()), 0, int(fileSize),
		unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		f.Close()
		return fmt.Errorf("mmap failed: %w", err)
	}

	// Read header size from first 8 bytes
	if len(data) < 8 {
		unix.Munmap(data)
		f.Close()
		return fmt.Errorf("file too small")
	}

	headerSize := binary.LittleEndian.Uint64(data[:8])
	if headerSize > 100*1024*1024 {
		unix.Munmap(data)
		f.Close()
		return fmt.Errorf("header size too large: %d", headerSize)
	}

	// Parse header JSON
	headerEnd := 8 + int(headerSize)
	if len(data) < headerEnd {
		unix.Munmap(data)
		f.Close()
		return fmt.Errorf("file too small for header")
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data[8:headerEnd], &header); err != nil {
		unix.Munmap(data)
		f.Close()
		return fmt.Errorf("failed to parse header: %w", err)
	}

	// Store mmap info
	filename := filepath.Base(path)
	mmapFile := &MmapFile{
		Path:       path,
		File:       f,
		Data:       data,
		Size:       fileSize,
		DataOffset: int64(headerEnd),
	}

	l.mu.Lock()
	l.mmapFiles[filename] = mmapFile

	// Extract tensor information
	dataOffset := int64(headerEnd)
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}

		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			continue
		}

		info.File = filename
		// Adjust offsets relative to data section
		info.Offsets[0] += dataOffset
		info.Offsets[1] += dataOffset

		l.index[name] = info
	}
	l.mu.Unlock()

	return nil
}

// GetTensorInfo returns information about a tensor.
func (l *MmapLoader) GetTensorInfo(name string) (*TensorInfo, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	info, ok := l.index[name]
	if !ok {
		return nil, false
	}
	return &info, true
}

// MmapTensor returns a memory-mapped slice for a tensor's data.
// The returned slice is directly backed by the mmap'd file, so no copy is made.
func (l *MmapLoader) MmapTensor(name string) ([]byte, error) {
	l.mu.RLock()
	info, ok := l.index[name]
	if !ok {
		l.mu.RUnlock()
		return nil, fmt.Errorf("tensor not found: %s", name)
	}

	mmapFile, ok := l.mmapFiles[info.File]
	if !ok {
		l.mu.RUnlock()
		return nil, fmt.Errorf("mmap file not found: %s", info.File)
	}
	l.mu.RUnlock()

	// Return slice of mmap'd data
	start := info.Offsets[0]
	end := info.Offsets[1]

	if start < 0 || end > int64(len(mmapFile.Data)) || start > end {
		return nil, fmt.Errorf("invalid tensor offsets: [%d, %d]", start, end)
	}

	return mmapFile.Data[start:end], nil
}

// MmapTensorAligned returns tensor data with guaranteed memory alignment.
// This is useful for GPU operations that require specific alignment (e.g., 256-byte).
// Note: If the data is not naturally aligned, a copy will be made to an aligned buffer.
func (l *MmapLoader) MmapTensorAligned(name string, alignment int) ([]byte, error) {
	data, err := l.MmapTensor(name)
	if err != nil {
		return nil, err
	}

	// Check if already aligned
	ptr := uintptr(unsafe.Pointer(&data[0]))
	if ptr%uintptr(alignment) == 0 {
		return data, nil
	}

	// Need to copy to aligned memory
	alignedBuf := AlignedAlloc(len(data), alignment)
	copy(alignedBuf, data)
	return alignedBuf, nil
}

// AlignedAlloc allocates a byte slice with specified alignment.
// This is useful for GPU memory copies that require alignment.
func AlignedAlloc(size, alignment int) []byte {
	// Allocate extra space for alignment
	buf := make([]byte, size+alignment-1)

	// Find the aligned start position
	ptr := uintptr(unsafe.Pointer(&buf[0]))
	offset := (alignment - int(ptr%uintptr(alignment))) % alignment

	return buf[offset : offset+size]
}

// LoadTensor loads a tensor by name (copies data unlike MmapTensor).
func (l *MmapLoader) LoadTensor(name string) ([]byte, *TensorInfo, error) {
	l.mu.RLock()
	info, ok := l.index[name]
	if !ok {
		l.mu.RUnlock()
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}

	mmapFile, ok := l.mmapFiles[info.File]
	if !ok {
		l.mu.RUnlock()
		return nil, nil, fmt.Errorf("mmap file not found: %s", info.File)
	}
	l.mu.RUnlock()

	// Copy data from mmap
	start := info.Offsets[0]
	end := info.Offsets[1]
	size := end - start

	data := make([]byte, size)
	copy(data, mmapFile.Data[start:end])

	return data, &info, nil
}

// ListTensors returns all tensor names.
func (l *MmapLoader) ListTensors() []string {
	l.mu.RLock()
	defer l.mu.RUnlock()

	names := make([]string, 0, len(l.index))
	for name := range l.index {
		names = append(names, name)
	}
	return names
}

// CountLayers returns the number of transformer layers in the model.
func (l *MmapLoader) CountLayers() int {
	l.mu.RLock()
	defer l.mu.RUnlock()

	maxLayer := -1
	for name := range l.index {
		if len(name) > 13 && name[:13] == "model.layers." {
			var layer int
			if _, err := fmt.Sscanf(name[13:], "%d", &layer); err == nil {
				if layer > maxLayer {
					maxLayer = layer
				}
			}
		}
	}
	return maxLayer + 1
}

// Close releases all mmap'd resources.
func (l *MmapLoader) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	var lastErr error
	for _, mf := range l.mmapFiles {
		if err := unix.Munmap(mf.Data); err != nil {
			lastErr = err
		}
		if err := mf.File.Close(); err != nil {
			lastErr = err
		}
	}
	l.mmapFiles = make(map[string]*MmapFile)
	return lastErr
}

// PrefetchTensor advises the kernel to prefetch tensor data into memory.
// This can improve performance when you know you'll need the data soon.
func (l *MmapLoader) PrefetchTensor(name string) error {
	l.mu.RLock()
	info, ok := l.index[name]
	if !ok {
		l.mu.RUnlock()
		return fmt.Errorf("tensor not found: %s", name)
	}

	mmapFile, ok := l.mmapFiles[info.File]
	if !ok {
		l.mu.RUnlock()
		return fmt.Errorf("mmap file not found: %s", info.File)
	}
	l.mu.RUnlock()

	start := info.Offsets[0]
	end := info.Offsets[1]

	// Use madvise to tell kernel we'll need this data soon
	return unix.Madvise(mmapFile.Data[start:end], unix.MADV_WILLNEED)
}

// MemoryStats returns memory statistics for the mmap'd files.
type MemoryStats struct {
	TotalMapped   int64
	TensorCount   int
	FileCount     int
	LargestTensor int64
}

// GetMemoryStats returns statistics about memory-mapped data.
func (l *MmapLoader) GetMemoryStats() MemoryStats {
	l.mu.RLock()
	defer l.mu.RUnlock()

	stats := MemoryStats{
		TensorCount: len(l.index),
		FileCount:   len(l.mmapFiles),
	}

	for _, mf := range l.mmapFiles {
		stats.TotalMapped += mf.Size
	}

	for _, info := range l.index {
		size := info.Offsets[1] - info.Offsets[0]
		if size > stats.LargestTensor {
			stats.LargestTensor = size
		}
	}

	return stats
}

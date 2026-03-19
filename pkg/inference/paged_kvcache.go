// Package inference provides paged KV cache management inspired by vLLM's PagedAttention.
//
// Instead of allocating one contiguous buffer per sequence (wasting VRAM),
// KV cache is divided into fixed-size blocks (pages) allocated on demand.
// This allows 2-4x more concurrent sequences and enables continuous batching.
package inference

import (
	"fmt"
	"sync"
)

// BlockSize is the number of tokens per KV cache block.
// Each block stores BlockSize tokens worth of K and V data for one layer.
// 16 is a good balance: small enough to avoid waste, large enough for efficiency.
const BlockSize = 16

// KVBlock represents a single block (page) of KV cache on GPU.
type KVBlock struct {
	BlockID  int  // Physical block ID (index into GPU memory pool)
	RefCount int  // Number of sequences using this block
	InUse    bool // Whether this block is allocated
}

// BlockPool manages a pool of KV cache blocks on GPU.
// Uses a free list for O(1) allocation and deallocation.
type BlockPool struct {
	blocks     []*KVBlock // All blocks (indexed by BlockID)
	freeList   []int      // Stack of free block IDs (LIFO for cache locality)
	numBlocks  int        // Total number of blocks
	numFree    int        // Number of currently free blocks
	mu         sync.Mutex
}

// NewBlockPool creates a pool of numBlocks KV cache blocks.
// totalGPUMemory is the VRAM to dedicate to KV cache.
// bytesPerBlock = BlockSize * numKVHeads * headDim * 2(FP16) * 2(K+V) * numLayers
func NewBlockPool(numBlocks int) *BlockPool {
	blocks := make([]*KVBlock, numBlocks)
	freeList := make([]int, numBlocks)

	for i := 0; i < numBlocks; i++ {
		blocks[i] = &KVBlock{BlockID: i, RefCount: 0, InUse: false}
		freeList[i] = numBlocks - 1 - i // Reverse so pop gives 0,1,2...
	}

	return &BlockPool{
		blocks:    blocks,
		freeList:  freeList,
		numBlocks: numBlocks,
		numFree:   numBlocks,
	}
}

// Allocate allocates n blocks from the pool. Returns block IDs.
// Returns error if not enough free blocks.
func (p *BlockPool) Allocate(n int) ([]int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if n > p.numFree {
		return nil, fmt.Errorf("not enough free blocks: need %d, have %d", n, p.numFree)
	}

	ids := make([]int, n)
	for i := 0; i < n; i++ {
		// Pop from free list (LIFO)
		blockID := p.freeList[p.numFree-1]
		p.freeList = p.freeList[:p.numFree-1]
		p.numFree--

		p.blocks[blockID].InUse = true
		p.blocks[blockID].RefCount = 1
		ids[i] = blockID
	}

	return ids, nil
}

// Free returns blocks to the pool.
func (p *BlockPool) Free(blockIDs []int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, id := range blockIDs {
		if id < 0 || id >= p.numBlocks {
			continue
		}
		block := p.blocks[id]
		block.RefCount--
		if block.RefCount <= 0 {
			block.InUse = false
			block.RefCount = 0
			p.freeList = append(p.freeList, id)
			p.numFree++
		}
	}
}

// NumFree returns the number of free blocks.
func (p *BlockPool) NumFree() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.numFree
}

// NumTotal returns the total number of blocks.
func (p *BlockPool) NumTotal() int {
	return p.numBlocks
}

// SequenceBlocks tracks the block table for a single sequence.
type SequenceBlocks struct {
	BlockIDs []int // Ordered list of physical block IDs for this sequence
	SeqLen   int   // Current sequence length (number of tokens cached)
}

// NumBlocks returns the number of blocks allocated for this sequence.
func (s *SequenceBlocks) NumBlocks() int {
	return len(s.BlockIDs)
}

// BlockTableForGPU returns the block table as int32 slice (for GPU kernel).
func (s *SequenceBlocks) BlockTableForGPU(maxBlocks int) []int32 {
	table := make([]int32, maxBlocks)
	for i, id := range s.BlockIDs {
		if i >= maxBlocks {
			break
		}
		table[i] = int32(id)
	}
	return table
}

// PagedKVCacheManager manages paged KV caches across all layers.
type PagedKVCacheManager struct {
	pool       *BlockPool
	sequences  map[uint64]*SequenceBlocks // seqID -> blocks
	numLayers  int
	numKVHeads int
	headDim    int
	mu         sync.RWMutex
}

// CalculateNumBlocks computes how many blocks fit in the given VRAM budget.
func CalculateNumBlocks(vramBytes uint64, numLayers, numKVHeads, headDim int) int {
	// Each block stores: BlockSize tokens * numKVHeads * headDim * 2(FP16) * 2(K+V)
	// This is per-layer, and we need blocks for all layers
	bytesPerBlockPerLayer := uint64(BlockSize * numKVHeads * headDim * 2 * 2)
	bytesPerBlock := bytesPerBlockPerLayer * uint64(numLayers)

	if bytesPerBlock == 0 {
		return 0
	}
	return int(vramBytes / bytesPerBlock)
}

// NewPagedKVCacheManager creates a paged KV cache manager.
func NewPagedKVCacheManager(numBlocks, numLayers, numKVHeads, headDim int) *PagedKVCacheManager {
	return &PagedKVCacheManager{
		pool:       NewBlockPool(numBlocks),
		sequences:  make(map[uint64]*SequenceBlocks),
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
	}
}

// AllocateForSequence allocates blocks for a new sequence up to maxTokens.
func (m *PagedKVCacheManager) AllocateForSequence(seqID uint64, numTokens int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	numBlocksNeeded := (numTokens + BlockSize - 1) / BlockSize

	blockIDs, err := m.pool.Allocate(numBlocksNeeded)
	if err != nil {
		return fmt.Errorf("sequence %d: %w", seqID, err)
	}

	m.sequences[seqID] = &SequenceBlocks{
		BlockIDs: blockIDs,
		SeqLen:   0,
	}

	return nil
}

// AppendToken records that a new token has been added to the sequence's KV cache.
// Allocates a new block if the current block is full.
func (m *PagedKVCacheManager) AppendToken(seqID uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	seq, ok := m.sequences[seqID]
	if !ok {
		return fmt.Errorf("sequence %d not found", seqID)
	}

	seq.SeqLen++

	// Check if we need a new block
	numBlocksNeeded := (seq.SeqLen + BlockSize - 1) / BlockSize
	if numBlocksNeeded > len(seq.BlockIDs) {
		// Allocate one more block
		newBlocks, err := m.pool.Allocate(1)
		if err != nil {
			return fmt.Errorf("sequence %d append: %w", seqID, err)
		}
		seq.BlockIDs = append(seq.BlockIDs, newBlocks[0])
	}

	return nil
}

// FreeSequence releases all blocks for a sequence.
func (m *PagedKVCacheManager) FreeSequence(seqID uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	seq, ok := m.sequences[seqID]
	if !ok {
		return
	}

	m.pool.Free(seq.BlockIDs)
	delete(m.sequences, seqID)
}

// GetBlockTable returns the block table for a sequence (for GPU kernel).
func (m *PagedKVCacheManager) GetBlockTable(seqID uint64, maxBlocks int) ([]int32, int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	seq, ok := m.sequences[seqID]
	if !ok {
		return nil, 0, fmt.Errorf("sequence %d not found", seqID)
	}

	return seq.BlockTableForGPU(maxBlocks), seq.SeqLen, nil
}

// GetSequenceLength returns the current cached length for a sequence.
func (m *PagedKVCacheManager) GetSequenceLength(seqID uint64) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	seq, ok := m.sequences[seqID]
	if !ok {
		return 0
	}
	return seq.SeqLen
}

// FreeAll releases all sequences.
func (m *PagedKVCacheManager) FreeAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for seqID, seq := range m.sequences {
		m.pool.Free(seq.BlockIDs)
		delete(m.sequences, seqID)
	}
}

// FirstActiveSequenceID returns the ID of the first active sequence, or 0 if none.
func (m *PagedKVCacheManager) FirstActiveSequenceID() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for seqID := range m.sequences {
		return seqID
	}
	return 0
}

// Stats returns current usage statistics.
func (m *PagedKVCacheManager) Stats() (totalBlocks, freeBlocks, activeSequences int) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.pool.NumTotal(), m.pool.NumFree(), len(m.sequences)
}

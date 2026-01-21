// Package scheduler provides VRAM-aware layer scheduling for distributed inference.
package scheduler

import (
	"context"
	"sync"
)

// PrefetchRequest represents a request to prefetch data for a layer.
type PrefetchRequest struct {
	LayerID  int
	PeerID   string
	SeqID    uint64
	Priority int // Higher = more urgent
}

// PrefetchStatus tracks the status of a prefetch operation.
type PrefetchStatus struct {
	Request   PrefetchRequest
	Completed bool
	Error     error
}

// PrefetchCallback is called when a prefetch completes.
type PrefetchCallback func(status PrefetchStatus)

// PrefetchCoordinator manages prefetching of activations between layers.
type PrefetchCoordinator struct {
	queue       chan PrefetchRequest
	callbacks   map[uint64]PrefetchCallback // keyed by SeqID
	inFlight    map[uint64]bool             // tracks active prefetches
	assignments []LayerAssignment
	mu          sync.RWMutex
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
}

// NewPrefetchCoordinator creates a new prefetch coordinator.
func NewPrefetchCoordinator(assignments []LayerAssignment, queueSize int) *PrefetchCoordinator {
	ctx, cancel := context.WithCancel(context.Background())

	pc := &PrefetchCoordinator{
		queue:       make(chan PrefetchRequest, queueSize),
		callbacks:   make(map[uint64]PrefetchCallback),
		inFlight:    make(map[uint64]bool),
		assignments: assignments,
		ctx:         ctx,
		cancel:      cancel,
	}

	return pc
}

// Start begins processing prefetch requests with the given number of workers.
func (pc *PrefetchCoordinator) Start(numWorkers int) {
	for i := 0; i < numWorkers; i++ {
		pc.wg.Add(1)
		go pc.worker()
	}
}

// Stop gracefully shuts down the coordinator.
func (pc *PrefetchCoordinator) Stop() {
	pc.cancel()
	pc.wg.Wait()
	close(pc.queue)
}

// worker processes prefetch requests from the queue.
func (pc *PrefetchCoordinator) worker() {
	defer pc.wg.Done()

	for {
		select {
		case <-pc.ctx.Done():
			return
		case req, ok := <-pc.queue:
			if !ok {
				return
			}
			pc.processPrefetch(req)
		}
	}
}

// processPrefetch handles a single prefetch request.
func (pc *PrefetchCoordinator) processPrefetch(req PrefetchRequest) {
	status := PrefetchStatus{
		Request:   req,
		Completed: true,
		Error:     nil,
	}

	// Mark as no longer in flight
	pc.mu.Lock()
	delete(pc.inFlight, req.SeqID)
	callback := pc.callbacks[req.SeqID]
	delete(pc.callbacks, req.SeqID)
	pc.mu.Unlock()

	// Invoke callback if registered
	if callback != nil {
		callback(status)
	}
}

// RequestPrefetch queues a prefetch request.
func (pc *PrefetchCoordinator) RequestPrefetch(req PrefetchRequest, callback PrefetchCallback) error {
	pc.mu.Lock()
	pc.inFlight[req.SeqID] = true
	if callback != nil {
		pc.callbacks[req.SeqID] = callback
	}
	pc.mu.Unlock()

	select {
	case pc.queue <- req:
		return nil
	case <-pc.ctx.Done():
		return pc.ctx.Err()
	}
}

// CancelPrefetch cancels a pending prefetch.
func (pc *PrefetchCoordinator) CancelPrefetch(seqID uint64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	delete(pc.inFlight, seqID)
	delete(pc.callbacks, seqID)
}

// IsPrefetching returns whether a sequence is currently being prefetched.
func (pc *PrefetchCoordinator) IsPrefetching(seqID uint64) bool {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	return pc.inFlight[seqID]
}

// QueueLength returns the current number of pending prefetch requests.
func (pc *PrefetchCoordinator) QueueLength() int {
	return len(pc.queue)
}

// GetNextPeer returns the peer that will handle the next layer in the pipeline.
func (pc *PrefetchCoordinator) GetNextPeer(currentLayerID int) (string, bool) {
	nextLayerID := currentLayerID + 1

	for _, a := range pc.assignments {
		if a.LayerID == nextLayerID {
			return a.PeerID, true
		}
	}

	return "", false
}

// GetLayerPeer returns the peer handling a specific layer.
func (pc *PrefetchCoordinator) GetLayerPeer(layerID int) (string, bool) {
	for _, a := range pc.assignments {
		if a.LayerID == layerID {
			return a.PeerID, true
		}
	}
	return "", false
}

// NeedsPrefetch determines if a layer transition requires prefetch.
// Returns true if the next layer is on a different peer.
func (pc *PrefetchCoordinator) NeedsPrefetch(currentLayerID int) bool {
	currentPeer, found := pc.GetLayerPeer(currentLayerID)
	if !found {
		return false
	}

	nextPeer, found := pc.GetNextPeer(currentLayerID)
	if !found {
		return false
	}

	return currentPeer != nextPeer
}

// GetPrefetchPlan returns a plan for which layers need prefetch during a forward pass.
func (pc *PrefetchCoordinator) GetPrefetchPlan(numLayers int) []int {
	var plan []int

	for i := -1; i < numLayers; i++ {
		if pc.NeedsPrefetch(i) {
			plan = append(plan, i)
		}
	}

	return plan
}

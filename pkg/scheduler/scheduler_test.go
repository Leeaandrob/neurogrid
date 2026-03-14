package scheduler

import (
	"testing"
)

func TestScheduler_HeterogeneousVRAM(t *testing.T) {
	// Simulate RTX 4090 (24GB) + RTX 2080 Ti (11GB) with TinyLlama (22 layers)
	cfg := DefaultTinyLlamaConfig()
	sched := NewScheduler(cfg)

	const GB = uint64(1024 * 1024 * 1024)

	// Register two GPUs with different VRAM (no pre-existing usage)
	if err := sched.RegisterPeer("rtx4090", 24*GB, 0); err != nil {
		t.Fatalf("RegisterPeer rtx4090: %v", err)
	}
	if err := sched.RegisterPeer("rtx2080ti", 11*GB, 0); err != nil {
		t.Fatalf("RegisterPeer rtx2080ti: %v", err)
	}

	assignments, err := sched.ComputeAssignments()
	if err != nil {
		t.Fatalf("ComputeAssignments: %v", err)
	}

	// Count transformer layers per peer (exclude embedding layer -1 and output layer)
	layersPerPeer := make(map[string]int)
	for _, a := range assignments {
		if a.LayerID >= 0 && a.LayerID < cfg.NumLayers {
			layersPerPeer[a.PeerID]++
		}
	}

	rtx4090Layers := layersPerPeer["rtx4090"]
	rtx2080tiLayers := layersPerPeer["rtx2080ti"]
	totalLayers := rtx4090Layers + rtx2080tiLayers

	t.Logf("RTX 4090 (24GB): %d layers", rtx4090Layers)
	t.Logf("RTX 2080 Ti (11GB): %d layers", rtx2080tiLayers)
	t.Logf("Total: %d layers (expected %d)", totalLayers, cfg.NumLayers)

	// All layers must be assigned
	if totalLayers != cfg.NumLayers {
		t.Fatalf("expected %d total layers, got %d", cfg.NumLayers, totalLayers)
	}

	// RTX 4090 should get more layers than RTX 2080 Ti
	if rtx4090Layers <= rtx2080tiLayers {
		t.Errorf("RTX 4090 should get more layers than RTX 2080 Ti: got %d vs %d",
			rtx4090Layers, rtx2080tiLayers)
	}

	// Check proportionality: RTX 4090 has ~68.6% of total VRAM (24/35)
	// Allow ±2 layers tolerance for rounding and embedding overhead
	expectedRatio4090 := 24.0 / 35.0
	actualRatio4090 := float64(rtx4090Layers) / float64(totalLayers)
	t.Logf("Expected ratio RTX 4090: %.1f%%, actual: %.1f%%",
		expectedRatio4090*100, actualRatio4090*100)

	if actualRatio4090 < 0.55 || actualRatio4090 > 0.80 {
		t.Errorf("RTX 4090 ratio out of expected range [55%%, 80%%]: got %.1f%%",
			actualRatio4090*100)
	}

	// Verify contiguous assignment: layers for each peer should be sequential
	for _, peerID := range []string{"rtx4090", "rtx2080ti"} {
		layers := sched.GetPeerLayers(assignments, peerID)
		// Filter to transformer layers only
		var xformerLayers []int
		for _, l := range layers {
			if l >= 0 && l < cfg.NumLayers {
				xformerLayers = append(xformerLayers, l)
			}
		}
		for i := 1; i < len(xformerLayers); i++ {
			if xformerLayers[i] != xformerLayers[i-1]+1 {
				t.Errorf("peer %s has non-contiguous layers: %v", peerID, xformerLayers)
				break
			}
		}
	}
}

func TestScheduler_EqualVRAM(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	sched := NewScheduler(cfg)

	const GB = uint64(1024 * 1024 * 1024)

	if err := sched.RegisterPeer("gpu0", 24*GB, 0); err != nil {
		t.Fatal(err)
	}
	if err := sched.RegisterPeer("gpu1", 24*GB, 0); err != nil {
		t.Fatal(err)
	}

	assignments, err := sched.ComputeAssignments()
	if err != nil {
		t.Fatalf("ComputeAssignments: %v", err)
	}

	layersPerPeer := make(map[string]int)
	for _, a := range assignments {
		if a.LayerID >= 0 && a.LayerID < cfg.NumLayers {
			layersPerPeer[a.PeerID]++
		}
	}

	// With equal VRAM, layers should be split evenly (11 each for 22 layers)
	gpu0 := layersPerPeer["gpu0"]
	gpu1 := layersPerPeer["gpu1"]
	t.Logf("gpu0: %d layers, gpu1: %d layers", gpu0, gpu1)

	if gpu0+gpu1 != cfg.NumLayers {
		t.Fatalf("total layers %d != %d", gpu0+gpu1, cfg.NumLayers)
	}

	diff := gpu0 - gpu1
	if diff < 0 {
		diff = -diff
	}
	// With equal VRAM, difference should be at most 1 (due to embedding overhead on one peer)
	if diff > 1 {
		t.Errorf("unbalanced split with equal VRAM: %d vs %d", gpu0, gpu1)
	}
}

func TestScheduler_SinglePeer(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	sched := NewScheduler(cfg)

	const GB = uint64(1024 * 1024 * 1024)

	if err := sched.RegisterPeer("solo", 24*GB, 0); err != nil {
		t.Fatal(err)
	}

	assignments, err := sched.ComputeAssignments()
	if err != nil {
		t.Fatalf("ComputeAssignments: %v", err)
	}

	for _, a := range assignments {
		if a.PeerID != "solo" {
			t.Errorf("layer %d assigned to %s, expected solo", a.LayerID, a.PeerID)
		}
	}

	// Should have embedding (-1) + 22 transformer layers + output (22)
	if len(assignments) != cfg.NumLayers+2 {
		t.Errorf("expected %d assignments, got %d", cfg.NumLayers+2, len(assignments))
	}
}

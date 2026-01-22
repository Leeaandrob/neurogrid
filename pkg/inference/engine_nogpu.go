//go:build !cuda

// Package inference provides CPU-only inference stubs when CUDA is not available.
package inference

import (
	"errors"

	"github.com/neurogrid/engine/pkg/model"
)

// GPUComponents is a placeholder for non-CUDA builds.
type GPUComponents struct {
	Initialized bool
}

// InitializeGPU returns an error when CUDA is not available.
func (e *Engine) InitializeGPU(loader *model.WeightLoader, deviceID int) (*GPUComponents, error) {
	return nil, errors.New("GPU inference not available: compiled without CUDA support")
}

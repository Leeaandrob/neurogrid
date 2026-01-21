package huggingface

import "errors"

// Errors for the huggingface package.
var (
	// ErrNotImplemented is returned when a feature is not yet implemented.
	ErrNotImplemented = errors.New("not implemented")

	// ErrAuthRequired is returned when authentication is required.
	ErrAuthRequired = errors.New("authentication required for gated model")

	// ErrChecksumMismatch is returned when file checksum doesn't match.
	ErrChecksumMismatch = errors.New("checksum mismatch")

	// ErrInsufficientSpace is returned when there's not enough disk space.
	ErrInsufficientSpace = errors.New("insufficient disk space")

	// ErrModelNotFound is returned when the model is not found.
	ErrModelNotFound = errors.New("model not found")
)

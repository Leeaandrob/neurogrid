// Package types provides core data types for the NeuroGrid engine.
package types

import (
	"encoding/binary"
	"math"
)

// BFloat16 represents a Brain Floating Point 16-bit number.
// It has the same exponent range as FP32 (8-bit exponent) but reduced precision (7-bit mantissa).
type BFloat16 uint16

// Float32ToBFloat16 converts a float32 to BFloat16 with round-to-nearest-even.
func Float32ToBFloat16(f float32) BFloat16 {
	bits := math.Float32bits(f)
	// Round to nearest even: add rounding bias
	// Check if the lower 16 bits are exactly at the midpoint (0x8000)
	// If so, round to even (round up if bit 16 is set)
	lsb := (bits >> 16) & 1
	roundingBias := uint32(0x7FFF) + lsb
	bits += roundingBias
	return BFloat16(bits >> 16)
}

// Float32 converts a BFloat16 back to float32.
func (b BFloat16) Float32() float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// ReadBFloat16Slice reads a byte slice as BFloat16 values and returns them.
// The input must have an even number of bytes (2 bytes per BFloat16).
func ReadBFloat16Slice(data []byte) []BFloat16 {
	if len(data)%2 != 0 {
		return nil
	}
	result := make([]BFloat16, len(data)/2)
	for i := range result {
		result[i] = BFloat16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
	}
	return result
}

// BFloat16SliceToFloat32 converts a slice of BFloat16 to float32.
func BFloat16SliceToFloat32(bf16 []BFloat16) []float32 {
	result := make([]float32, len(bf16))
	for i, b := range bf16 {
		result[i] = b.Float32()
	}
	return result
}

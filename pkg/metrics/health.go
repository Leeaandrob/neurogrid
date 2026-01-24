// Package metrics provides Prometheus metrics for NeuroGrid inference engine.
// This file contains tensor health metrics for numerical validation.
package metrics

import (
	"encoding/binary"
	"math"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const (
	tensorSubsystem = "tensor"

	// SampleInterval defines how often to sample tensor elements for health checks.
	// Sampling every 64th element keeps overhead <5% while maintaining detection accuracy.
	SampleInterval = 64

	// FP16ByteSize is the size of a FP16 value in bytes.
	FP16ByteSize = 2
)

// Tensor health metrics
var (
	// TensorNaNCount tracks the total number of NaN values detected in tensors.
	TensorNaNCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: tensorSubsystem,
			Name:      "nan_count_total",
			Help:      "Total NaN values detected in tensors (extrapolated from sampling)",
		},
		[]string{"layer", "tensor_name"},
	)

	// TensorInfCount tracks the total number of Inf values detected in tensors.
	TensorInfCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: tensorSubsystem,
			Name:      "inf_count_total",
			Help:      "Total Inf values detected in tensors (extrapolated from sampling)",
		},
		[]string{"layer", "tensor_name"},
	)

	// HiddenStateMean tracks the mean of hidden state activations per layer.
	HiddenStateMean = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: tensorSubsystem,
			Name:      "hidden_state_mean",
			Help:      "Mean of hidden state activations per layer",
		},
		[]string{"layer"},
	)

	// HiddenStateStd tracks the standard deviation of hidden state activations per layer.
	HiddenStateStd = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: tensorSubsystem,
			Name:      "hidden_state_std",
			Help:      "Standard deviation of hidden state activations per layer",
		},
		[]string{"layer"},
	)

	// LayerExecutionDuration tracks the duration of layer execution.
	LayerExecutionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: tensorSubsystem,
			Name:      "layer_execution_duration_seconds",
			Help:      "Duration of layer execution in seconds",
			Buckets:   []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
		},
		[]string{"layer", "location"},
	)

	// TensorHealthCheckDuration tracks the overhead of health checks.
	TensorHealthCheckDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: tensorSubsystem,
			Name:      "health_check_duration_seconds",
			Help:      "Duration of tensor health check in seconds",
			Buckets:   []float64{.00001, .00005, .0001, .0005, .001, .005, .01},
		},
	)
)

// TensorHealthResult contains the results of a tensor health check.
type TensorHealthResult struct {
	NaNCount    int
	InfCount    int
	Mean        float64
	Std         float64
	SampleCount int
}

// CheckTensorHealthFP16 performs a sampled health check on FP16 tensor data.
// It samples every SampleInterval-th element to maintain <5% overhead.
// The results are extrapolated to estimate full tensor statistics.
func CheckTensorHealthFP16(data []byte) TensorHealthResult {
	if len(data) < FP16ByteSize {
		return TensorHealthResult{}
	}

	var (
		nanCount    int
		infCount    int
		sum         float64
		sumSquared  float64
		sampleCount int
	)

	// Sample every SampleInterval-th element
	stepBytes := SampleInterval * FP16ByteSize
	for i := 0; i+FP16ByteSize <= len(data); i += stepBytes {
		fp16Bits := binary.LittleEndian.Uint16(data[i : i+FP16ByteSize])
		fp32Val := fp16ToFloat32(fp16Bits)

		if math.IsNaN(float64(fp32Val)) {
			nanCount++
		} else if math.IsInf(float64(fp32Val), 0) {
			infCount++
		} else {
			sum += float64(fp32Val)
			sumSquared += float64(fp32Val) * float64(fp32Val)
		}
		sampleCount++
	}

	// Calculate statistics from valid samples
	validSamples := sampleCount - nanCount - infCount
	var mean, std float64
	if validSamples > 0 {
		mean = sum / float64(validSamples)
		variance := (sumSquared / float64(validSamples)) - (mean * mean)
		if variance > 0 {
			std = math.Sqrt(variance)
		}
	}

	return TensorHealthResult{
		NaNCount:    nanCount,
		InfCount:    infCount,
		Mean:        mean,
		Std:         std,
		SampleCount: sampleCount,
	}
}

// RecordTensorHealth records tensor health metrics to Prometheus.
// layerID is the layer index, tensorName identifies the tensor (e.g., "hidden_state", "attention").
func RecordTensorHealth(layerID int, tensorName string, result TensorHealthResult) {
	layerStr := strconv.Itoa(layerID)

	// Extrapolate counts from sampling
	if result.NaNCount > 0 {
		TensorNaNCount.WithLabelValues(layerStr, tensorName).Add(float64(result.NaNCount * SampleInterval))
	}
	if result.InfCount > 0 {
		TensorInfCount.WithLabelValues(layerStr, tensorName).Add(float64(result.InfCount * SampleInterval))
	}

	// Record statistics
	HiddenStateMean.WithLabelValues(layerStr).Set(result.Mean)
	HiddenStateStd.WithLabelValues(layerStr).Set(result.Std)
}

// RecordLayerDuration records the execution duration for a layer.
// location should be "local" or "remote".
func RecordLayerDuration(layerID int, location string, durationSeconds float64) {
	LayerExecutionDuration.WithLabelValues(strconv.Itoa(layerID), location).Observe(durationSeconds)
}

// fp16ToFloat32 converts a FP16 bit pattern to float32.
// FP16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits.
func fp16ToFloat32(fp16Bits uint16) float32 {
	sign := uint32((fp16Bits >> 15) & 0x1)
	exp := uint32((fp16Bits >> 10) & 0x1F)
	mant := uint32(fp16Bits & 0x3FF)

	var fp32Bits uint32

	switch exp {
	case 0:
		if mant == 0 {
			// Zero
			fp32Bits = sign << 31
		} else {
			// Subnormal FP16 -> Normal FP32
			exp = 127 - 14
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			fp32Bits = (sign << 31) | (exp << 23) | (mant << 13)
		}
	case 31:
		// Inf or NaN
		fp32Bits = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		// Normal number
		fp32Bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}

	return math.Float32frombits(fp32Bits)
}

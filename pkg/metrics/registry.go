// Package metrics provides Prometheus metrics for NeuroGrid inference engine.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const (
	namespace = "neurogrid"
	subsystem = "inference"
)

var (
	// RequestsTotal is the total number of inference requests.
	RequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "requests_total",
			Help:      "Total number of inference requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	// RequestDuration is the duration of inference requests.
	RequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_duration_seconds",
			Help:      "Duration of inference requests in seconds",
			Buckets:   []float64{.1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120},
		},
		[]string{"method", "endpoint"},
	)

	// TTFT is the Time To First Token histogram.
	TTFT = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "ttft_seconds",
			Help:      "Time to first token in seconds",
			Buckets:   []float64{.1, .25, .5, 1, 1.5, 2, 2.5, 3, 5, 10},
		},
	)

	// TokensGeneratedTotal is the total number of tokens generated.
	TokensGeneratedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "tokens_generated_total",
			Help:      "Total number of tokens generated",
		},
	)

	// TokensPerSecond is a gauge for current generation speed.
	TokensPerSecond = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "tokens_per_second",
			Help:      "Current token generation rate per second",
		},
	)

	// ActiveRequests is the number of requests currently being processed.
	ActiveRequests = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "active_requests",
			Help:      "Number of requests currently being processed",
		},
	)
)

// Cluster metrics
var (
	// PeersConnected is the number of connected peers in the cluster.
	PeersConnected = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "cluster",
			Name:      "peers_connected",
			Help:      "Number of connected peers in the cluster",
		},
	)

	// PeerReconnectionsTotal is the total number of peer reconnections.
	PeerReconnectionsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "cluster",
			Name:      "peer_reconnections_total",
			Help:      "Total number of peer reconnection attempts",
		},
	)

	// PeerHealthCheckFailures is the total number of peer health check failures.
	PeerHealthCheckFailures = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "cluster",
			Name:      "peer_health_check_failures_total",
			Help:      "Total number of peer health check failures",
		},
		[]string{"peer_id"},
	)
)

// GPU metrics
var (
	// GPUMemoryUsedBytes is the GPU memory currently in use.
	GPUMemoryUsedBytes = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "gpu",
			Name:      "memory_used_bytes",
			Help:      "GPU memory currently in use (bytes)",
		},
	)

	// GPUMemoryTotalBytes is the total GPU memory available.
	GPUMemoryTotalBytes = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "gpu",
			Name:      "memory_total_bytes",
			Help:      "Total GPU memory available (bytes)",
		},
	)

	// GPUUtilization is the current GPU utilization percentage.
	GPUUtilization = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "gpu",
			Name:      "utilization_percent",
			Help:      "Current GPU utilization percentage",
		},
	)
)

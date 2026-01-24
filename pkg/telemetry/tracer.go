// Package telemetry provides OpenTelemetry tracing support for NeuroGrid.
package telemetry

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	// DefaultOTELEndpoint is the default OTLP gRPC endpoint for Jaeger.
	DefaultOTELEndpoint = "localhost:4317"

	// TracerName is the name of the NeuroGrid tracer.
	TracerName = "github.com/neurogrid/engine"
)

var (
	// globalTracer is the NeuroGrid tracer instance.
	globalTracer trace.Tracer

	// globalTracerProvider is the tracer provider for cleanup.
	globalTracerProvider *sdktrace.TracerProvider
)

// TracerConfig holds configuration for the tracer.
type TracerConfig struct {
	ServiceName string
	Endpoint    string
	Insecure    bool // Use insecure connection (for local development)
}

// InitTracer initializes the OpenTelemetry tracer with OTLP gRPC exporter.
// Returns a cleanup function that should be called on shutdown.
func InitTracer(cfg TracerConfig) (func(context.Context) error, error) {
	if cfg.ServiceName == "" {
		cfg.ServiceName = "neurogrid"
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = DefaultOTELEndpoint
	}

	ctx := context.Background()

	// Create OTLP gRPC connection
	dialOpts := []grpc.DialOption{}
	if cfg.Insecure {
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	conn, err := grpc.NewClient(cfg.Endpoint, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC connection: %w", err)
	}

	// Create OTLP trace exporter
	exporter, err := otlptracegrpc.New(ctx, otlptracegrpc.WithGRPCConn(conn))
	if err != nil {
		return nil, fmt.Errorf("failed to create OTLP exporter: %w", err)
	}

	// Create resource with service information
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(cfg.ServiceName),
			semconv.ServiceVersion("0.1.0"),
			attribute.String("environment", "development"),
		),
		resource.WithHost(),
		resource.WithProcess(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create tracer provider with batched exporter
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter,
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithMaxExportBatchSize(512),
		),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tp)

	// Set global propagator for W3C Trace Context
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	// Store globals for package-level access
	globalTracerProvider = tp
	globalTracer = tp.Tracer(TracerName)

	// Return cleanup function
	cleanup := func(ctx context.Context) error {
		ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		return tp.Shutdown(ctx)
	}

	return cleanup, nil
}

// Tracer returns the global NeuroGrid tracer.
// Returns a no-op tracer if InitTracer has not been called.
func Tracer() trace.Tracer {
	if globalTracer != nil {
		return globalTracer
	}
	return otel.Tracer(TracerName)
}

// TracerProvider returns the global tracer provider.
func TracerProvider() trace.TracerProvider {
	if globalTracerProvider != nil {
		return globalTracerProvider
	}
	return otel.GetTracerProvider()
}

// IsEnabled returns true if tracing has been initialized.
func IsEnabled() bool {
	return globalTracerProvider != nil
}

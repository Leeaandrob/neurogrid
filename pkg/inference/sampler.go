// Package inference provides distributed inference capabilities for LLM generation.
package inference

import (
	"math"
	"math/rand"
	"sort"
)

// Sampler implements token sampling strategies including temperature and top-p.
type Sampler struct {
	rng *rand.Rand
}

// NewSampler creates a new sampler with the given random seed.
func NewSampler(seed int64) *Sampler {
	return &Sampler{
		rng: rand.New(rand.NewSource(seed)),
	}
}

// Sample selects a token from logits using temperature and top-p sampling.
// If temperature <= 0, uses greedy decoding (argmax).
// If topP < 1.0, uses nucleus sampling.
func (s *Sampler) Sample(logits []float32, temperature, topP float32) int {
	if len(logits) == 0 {
		return 0
	}

	// Greedy decoding for temperature <= 0
	if temperature <= 0 {
		return argmax(logits)
	}

	// Apply temperature scaling
	scaled := make([]float32, len(logits))
	for i, l := range logits {
		scaled[i] = l / temperature
	}

	// Convert to probabilities via softmax
	probs := softmax(scaled)

	// Apply top-p (nucleus) sampling if specified
	if topP < 1.0 && topP > 0 {
		probs = nucleusSample(probs, topP)
	}

	// Sample from the distribution
	return sampleFromDistribution(s.rng, probs)
}

// SampleGreedy returns the token with highest logit (argmax).
func (s *Sampler) SampleGreedy(logits []float32) int {
	return argmax(logits)
}

// argmax returns the index of the maximum value.
func argmax(values []float32) int {
	if len(values) == 0 {
		return 0
	}

	maxIdx := 0
	maxVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// softmax converts logits to probabilities.
func softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}

	// Find max for numerical stability
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// Compute exp(x - max) and sum
	probs := make([]float32, len(logits))
	var sum float64
	for i, l := range logits {
		exp := math.Exp(float64(l - maxLogit))
		probs[i] = float32(exp)
		sum += exp
	}

	// Normalize
	for i := range probs {
		probs[i] /= float32(sum)
	}

	return probs
}

// indexedProb pairs a probability with its original index.
type indexedProb struct {
	index int
	prob  float32
}

// nucleusSample implements top-p (nucleus) sampling.
// It keeps only the smallest set of tokens whose cumulative probability >= p.
func nucleusSample(probs []float32, p float32) []float32 {
	if len(probs) == 0 {
		return nil
	}

	// Create indexed probabilities for sorting
	indexed := make([]indexedProb, len(probs))
	for i, prob := range probs {
		indexed[i] = indexedProb{index: i, prob: prob}
	}

	// Sort by probability descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Find cutoff where cumulative probability >= p
	var cumSum float32
	cutoff := len(indexed) - 1
	for i, ip := range indexed {
		cumSum += ip.prob
		if cumSum >= p {
			cutoff = i
			break
		}
	}

	// Create result with only tokens in nucleus
	result := make([]float32, len(probs))
	for i := 0; i <= cutoff; i++ {
		result[indexed[i].index] = indexed[i].prob
	}

	// Renormalize
	return normalize(result)
}

// normalize normalizes probabilities to sum to 1.
func normalize(probs []float32) []float32 {
	var sum float32
	for _, p := range probs {
		sum += p
	}

	if sum == 0 {
		return probs
	}

	result := make([]float32, len(probs))
	for i, p := range probs {
		result[i] = p / sum
	}
	return result
}

// sampleFromDistribution samples an index from a probability distribution.
func sampleFromDistribution(rng *rand.Rand, probs []float32) int {
	if len(probs) == 0 {
		return 0
	}

	// Generate random number in [0, 1)
	r := rng.Float32()

	// Find the token by cumulative probability
	var cumSum float32
	for i, p := range probs {
		cumSum += p
		if r < cumSum {
			return i
		}
	}

	// Return last token if rounding error
	return len(probs) - 1
}

// TopKFilter zeroes out all but the top k probabilities.
func TopKFilter(probs []float32, k int) []float32 {
	if k >= len(probs) || k <= 0 {
		return probs
	}

	// Create indexed probabilities
	indexed := make([]indexedProb, len(probs))
	for i, prob := range probs {
		indexed[i] = indexedProb{index: i, prob: prob}
	}

	// Sort by probability descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Keep only top k
	result := make([]float32, len(probs))
	for i := 0; i < k; i++ {
		result[indexed[i].index] = indexed[i].prob
	}

	return normalize(result)
}

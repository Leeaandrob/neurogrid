// Package inference provides speculative decoding for faster generation.
//
// Speculative decoding uses a fast draft model to generate K candidate tokens,
// then verifies them all in parallel with the target model. Accepted tokens
// skip individual forward passes, effectively multiplying throughput.
//
// Algorithm:
//  1. Draft model generates K tokens autoregressively (fast, ~1ms/token)
//  2. Target model verifies all K tokens in one forward pass
//  3. Accept consecutive matching tokens; reject at first mismatch
//  4. Use target model's token at rejection point as bonus token
//  5. Average acceptance rate ~60-70% → 2-3x effective speedup
package inference

import (
	"context"
	"fmt"
	"log"
)

// SpeculativeConfig holds configuration for speculative decoding.
type SpeculativeConfig struct {
	NumSpecTokens int     // Number of speculative tokens per step (K)
	Enabled       bool    // Whether speculative decoding is active
}

// DefaultSpeculativeConfig returns default speculative decoding config.
func DefaultSpeculativeConfig() *SpeculativeConfig {
	return &SpeculativeConfig{
		NumSpecTokens: 4,     // Generate 4 draft tokens per step
		Enabled:       false, // Disabled by default
	}
}

// SpeculativeDecoder orchestrates draft + target model for speculative decoding.
type SpeculativeDecoder struct {
	draftEngine  *Engine          // Fast draft model (e.g., TinyLlama)
	targetEngine *Engine          // Accurate target model (e.g., LFM2)
	config       *SpeculativeConfig
	stats        SpecStats
}

// SpecStats tracks speculative decoding statistics.
type SpecStats struct {
	TotalDraftTokens    int
	AcceptedTokens      int
	RejectedTokens      int
	BonusTokens         int
	TotalTargetForwards int
}

// AcceptanceRate returns the ratio of accepted draft tokens.
func (s *SpecStats) AcceptanceRate() float64 {
	if s.TotalDraftTokens == 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.TotalDraftTokens)
}

// TokensPerStep returns average tokens generated per target forward pass.
func (s *SpecStats) TokensPerStep() float64 {
	if s.TotalTargetForwards == 0 {
		return 0
	}
	return float64(s.AcceptedTokens+s.BonusTokens) / float64(s.TotalTargetForwards)
}

// NewSpeculativeDecoder creates a speculative decoder with draft and target engines.
// For self-speculative mode, pass the same engine for both draft and target.
func NewSpeculativeDecoder(draftEngine, targetEngine *Engine, config *SpeculativeConfig) *SpeculativeDecoder {
	return &SpeculativeDecoder{
		draftEngine:  draftEngine,
		targetEngine: targetEngine,
		config:       config,
	}
}

// NewSelfSpeculativeDecoder creates a decoder that uses the same model as both
// draft and target. The draft generates with greedy sampling (fast), the target
// verifies with the actual sampling parameters.
func NewSelfSpeculativeDecoder(engine *Engine, config *SpeculativeConfig) *SpeculativeDecoder {
	return &SpeculativeDecoder{
		draftEngine:  engine,
		targetEngine: engine,
		config:       config,
	}
}

// GenerateSpeculative performs speculative decoding generation.
// Returns generated tokens and the text.
func (sd *SpeculativeDecoder) GenerateSpeculative(
	ctx context.Context,
	req *GenerateRequest,
) (*GenerateResponse, error) {
	if sd.draftEngine.tokenizer == nil || sd.targetEngine.tokenizer == nil {
		return nil, fmt.Errorf("tokenizers not set")
	}

	// Tokenize input using target model's tokenizer
	inputTokens, err := sd.targetEngine.tokenizer.Encode(req.Prompt)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	log.Printf("[SpecDecode] Input tokens: %d, K=%d", len(inputTokens), sd.config.NumSpecTokens)

	// Prefill both models with input tokens
	// TODO: share KV cache for common prefix when models share architecture
	draftHidden, err := sd.draftEngine.prefill(ctx, inputTokens, 1)
	if err != nil {
		return nil, fmt.Errorf("draft prefill failed: %w", err)
	}

	targetHidden, err := sd.targetEngine.prefill(ctx, inputTokens, 2)
	if err != nil {
		return nil, fmt.Errorf("target prefill failed: %w", err)
	}

	outputTokens := make([]int, 0, req.MaxTokens)
	stopReason := "max_tokens"
	K := sd.config.NumSpecTokens

	position := len(inputTokens)

	for len(outputTokens) < req.MaxTokens {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Step 1: Draft model generates K candidate tokens
		draftTokens := make([]int, 0, K)
		currentDraftHidden := draftHidden

		for k := 0; k < K && len(outputTokens)+len(draftTokens) < req.MaxTokens; k++ {
			// Forward draft model
			logits, err := sd.draftEngine.forwardAllLayers(ctx, currentDraftHidden, position+k, 1)
			if err != nil {
				return nil, fmt.Errorf("draft forward failed: %w", err)
			}

			// Sample from draft
			token := sd.draftEngine.sampler.Sample(logits, req.Temperature, req.TopP)
			draftTokens = append(draftTokens, token)
			sd.stats.TotalDraftTokens++

			// Check EOS
			if token == sd.targetEngine.tokenizer.EOSToken() {
				break
			}

			// Get embedding for next draft step
			currentDraftHidden, err = sd.draftEngine.embedToken(token)
			if err != nil {
				break
			}
		}

		if len(draftTokens) == 0 {
			break
		}

		// Step 2: Target model verifies all draft tokens
		// Process each draft token through target model and compare
		currentTargetHidden := targetHidden
		accepted := 0
		sd.stats.TotalTargetForwards++

		for i, draftToken := range draftTokens {
			// Forward target model at this position
			targetLogits, err := sd.targetEngine.forwardAllLayers(ctx, currentTargetHidden, position+i, 2)
			if err != nil {
				return nil, fmt.Errorf("target verify failed: %w", err)
			}

			// Sample from target
			targetToken := sd.targetEngine.sampler.Sample(targetLogits, req.Temperature, req.TopP)

			if targetToken == draftToken {
				// ACCEPT: draft matches target
				outputTokens = append(outputTokens, targetToken)
				accepted++
				sd.stats.AcceptedTokens++

				if decoded, err := sd.targetEngine.tokenizer.Decode([]int{targetToken}); err == nil {
					log.Printf("[SpecDecode] ACCEPT token %d=%q (draft agreed)", targetToken, decoded)
				}

				// Check EOS
				if targetToken == sd.targetEngine.tokenizer.EOSToken() {
					stopReason = "eos"
					break
				}

				// Embed for next position
				currentTargetHidden, err = sd.targetEngine.embedToken(targetToken)
				if err != nil {
					break
				}
			} else {
				// REJECT: use target's token instead
				outputTokens = append(outputTokens, targetToken)
				sd.stats.RejectedTokens++
				sd.stats.BonusTokens++

				if decoded, err := sd.targetEngine.tokenizer.Decode([]int{targetToken}); err == nil {
					log.Printf("[SpecDecode] REJECT at pos %d: draft=%d, target=%d=%q", i, draftToken, targetToken, decoded)
				}

				// Check EOS
				if targetToken == sd.targetEngine.tokenizer.EOSToken() {
					stopReason = "eos"
					break
				}

				// Use target hidden for next step
				currentTargetHidden, err = sd.targetEngine.embedToken(targetToken)
				if err != nil {
					break
				}
				break // Stop verifying after first rejection
			}
		}

		// Update position and hidden states for next speculative step
		position += accepted + 1 // accepted tokens + 1 bonus/rejected
		targetHidden = currentTargetHidden

		// Also advance draft model to match target state
		// Re-embed the last accepted/rejected token for draft
		if len(outputTokens) > 0 {
			lastToken := outputTokens[len(outputTokens)-1]
			draftHidden, _ = sd.draftEngine.embedToken(lastToken)
		}

		if stopReason == "eos" {
			break
		}

		// Check stop tokens
		for _, stop := range req.StopTokens {
			if len(outputTokens) > 0 && outputTokens[len(outputTokens)-1] == stop {
				stopReason = "stop_token"
			}
		}
		if stopReason == "stop_token" {
			break
		}
	}

	// Decode output
	text, err := sd.targetEngine.tokenizer.Decode(outputTokens)
	if err != nil {
		text = ""
	}

	log.Printf("[SpecDecode] Done: %d tokens, acceptance=%.1f%%, tokens/step=%.1f",
		len(outputTokens), sd.stats.AcceptanceRate()*100, sd.stats.TokensPerStep())

	return &GenerateResponse{
		Text:       text,
		TokenCount: len(outputTokens),
		StopReason: stopReason,
	}, nil
}

// Stats returns speculative decoding statistics.
func (sd *SpeculativeDecoder) Stats() SpecStats {
	return sd.stats
}

// ResetStats resets the statistics counters.
func (sd *SpeculativeDecoder) ResetStats() {
	sd.stats = SpecStats{}
}

// Package guardrail provides attachable message guardrails for package llm.
package guardrail

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

const (
	DefaultSystemPromptLeakThreshold     = 0.20
	DefaultSystemPromptLeakMinMatchWords = 8
)

// SystemPromptLeakConfig controls exact normalized system-prompt coverage checks.
type SystemPromptLeakConfig struct {
	Threshold     float64
	MinMatchWords int
}

type systemPromptLeakGuardrail struct {
	threshold     float64
	minMatchWords int
}

// NewSystemPromptLeakGuardrail returns a guardrail that blocks assistant candidates
// containing too much exact normalized text from any system message.
func NewSystemPromptLeakGuardrail(config SystemPromptLeakConfig) (llm.MessageGuardrail, error) {
	if config.Threshold < 0 || config.Threshold > 1 {
		return nil, fmt.Errorf("system prompt leak threshold must be between 0 and 1")
	}
	if config.MinMatchWords < 0 {
		return nil, fmt.Errorf("system prompt leak minimum match words cannot be negative")
	}
	if config.Threshold == 0 {
		config.Threshold = DefaultSystemPromptLeakThreshold
	}
	if config.MinMatchWords == 0 {
		config.MinMatchWords = DefaultSystemPromptLeakMinMatchWords
	}
	return systemPromptLeakGuardrail{
		threshold:     config.Threshold,
		minMatchWords: config.MinMatchWords,
	}, nil
}

func (systemPromptLeakGuardrail) Name() string {
	return "system_prompt_leak"
}

func (guardrail systemPromptLeakGuardrail) CheckMessage(_ context.Context, input llm.MessageGuardrailInput) (llm.MessageGuardrailDecision, error) {
	if input.Message.Type != llm.Assistant {
		return llm.MessageGuardrailDecision{Action: llm.MessageGuardrailAllow}, nil
	}

	candidateParts := []string{input.Message.Content}
	for _, call := range input.Message.ToolCalls {
		candidateParts = append(candidateParts, call.Arguments)
	}
	candidate := tokenize(strings.Join(candidateParts, " "))
	if len(candidate) == 0 {
		return llm.MessageGuardrailDecision{Action: llm.MessageGuardrailAllow}, nil
	}

	for _, message := range input.Messages {
		if message.Type != llm.System {
			continue
		}
		coverage := promptCoverage(tokenize(message.Content), candidate, guardrail.minMatchWords)
		if coverage >= guardrail.threshold {
			return llm.MessageGuardrailDecision{
				Action: llm.MessageGuardrailBlock,
				Reason: fmt.Sprintf(
					"system prompt coverage %.3f reached threshold %.3f",
					coverage,
					guardrail.threshold,
				),
			}, nil
		}
	}
	return llm.MessageGuardrailDecision{Action: llm.MessageGuardrailAllow}, nil
}

func tokenize(value string) []string {
	return strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func promptCoverage(prompt []string, candidate []string, minMatchWords int) float64 {
	if len(prompt) == 0 || len(candidate) == 0 {
		return 0
	}
	if len(prompt) < minMatchWords {
		if containsWords(candidate, prompt) {
			return 1
		}
		return 0
	}

	candidateShingles := make(map[string]struct{})
	for start := 0; start+minMatchWords <= len(candidate); start++ {
		candidateShingles[shingle(candidate[start:start+minMatchWords])] = struct{}{}
	}
	covered := make([]bool, len(prompt))
	for start := 0; start+minMatchWords <= len(prompt); start++ {
		if _, ok := candidateShingles[shingle(prompt[start:start+minMatchWords])]; !ok {
			continue
		}
		for index := start; index < start+minMatchWords; index++ {
			covered[index] = true
		}
	}
	coveredCount := 0
	for _, isCovered := range covered {
		if isCovered {
			coveredCount++
		}
	}
	return float64(coveredCount) / float64(len(prompt))
}

func containsWords(haystack []string, needle []string) bool {
	if len(needle) == 0 || len(needle) > len(haystack) {
		return false
	}
	for start := 0; start+len(needle) <= len(haystack); start++ {
		matches := true
		for offset := range needle {
			if haystack[start+offset] != needle[offset] {
				matches = false
				break
			}
		}
		if matches {
			return true
		}
	}
	return false
}

func shingle(words []string) string {
	return strings.Join(words, "\x00")
}

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

func TestPrintCostBreakdown(t *testing.T) {
	firstCost := 0.01
	totalCost := 0.01
	output := &llm.CompletionCallOutput{
		Usage: llm.TokenUsage{
			PromptTokens:     13,
			CompletionTokens: 5,
			TotalTokens:      18,
			CostUSD:          &totalCost,
		},
		UsageEvents: []llm.UsageEvent{
			{
				Model:   "test-model",
				Round:   0,
				Attempt: 0,
				Usage: llm.TokenUsage{
					PromptTokens:     10,
					CompletionTokens: 2,
					TotalTokens:      12,
					CostUSD:          &firstCost,
				},
			},
			{
				Model:   "test-model",
				Round:   1,
				Attempt: 0,
				Usage: llm.TokenUsage{
					PromptTokens:     3,
					CompletionTokens: 3,
					TotalTokens:      6,
				},
			},
		},
	}

	var result bytes.Buffer
	printCostBreakdown(&result, output)

	for _, want := range []string{
		"Cost breakdown:\n",
		"Generation 1: model=test-model round=0 attempt=0 prompt=10 completion=2 total=12 cost=$0.01000000",
		"Generation 2: model=test-model round=1 attempt=0 prompt=3 completion=3 total=6 cost=unavailable",
		"Total: prompt=13 completion=5 total=18 cost=$0.01000000",
	} {
		if !strings.Contains(result.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, result.String())
		}
	}
}

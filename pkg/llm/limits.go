package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const (
	DefaultMaxProviderAttempts  = 32
	DefaultMaxToolCalls         = 64
	DefaultMaxRepeatedToolCalls = 4
	DefaultMaxApprovalDenials   = 3
)

// ErrCompletionLimitExceeded identifies an aggregate completion budget breach.
var ErrCompletionLimitExceeded = errors.New("completion limit exceeded")

// ErrCompletionUsageUnavailable identifies an unmetered response under a token ceiling.
var ErrCompletionUsageUnavailable = errors.New("completion usage unavailable")

// CompletionLimitError reports one exhausted budget without exposing message content.
type CompletionLimitError struct {
	Limit string
	Used  int64
	Max   int64
	Err   error
}

func (err *CompletionLimitError) Error() string {
	if err == nil {
		return "<nil>"
	}
	if err.Err != nil {
		return fmt.Sprintf("%v: %s: %v", ErrCompletionLimitExceeded, err.Limit, err.Err)
	}
	return fmt.Sprintf("%v: %s used %d, max %d", ErrCompletionLimitExceeded, err.Limit, err.Used, err.Max)
}

func (err *CompletionLimitError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Err
}

func (err *CompletionLimitError) Is(target error) bool {
	return target == ErrCompletionLimitExceeded || target == err.Err
}

type completionBudgetState struct {
	maxProviderAttempts  int
	maxToolCalls         int
	maxRepeatedToolCalls int
	maxApprovalDenials   int
	maxTotalTokens       int
	providerAttempts     int
	toolCalls            int
	approvalDenials      int
	repeatedToolCalls    map[string]int
}

func newCompletionBudgetState(input CompletionCallInput) *completionBudgetState {
	return &completionBudgetState{
		maxProviderAttempts:  effectiveLimit(input.MaxProviderAttempts, DefaultMaxProviderAttempts),
		maxToolCalls:         effectiveLimit(input.MaxToolCalls, DefaultMaxToolCalls),
		maxRepeatedToolCalls: effectiveLimit(input.MaxRepeatedToolCalls, DefaultMaxRepeatedToolCalls),
		maxApprovalDenials:   effectiveLimit(input.MaxApprovalDenials, DefaultMaxApprovalDenials),
		maxTotalTokens:       input.MaxTotalTokens,
		repeatedToolCalls:    make(map[string]int),
	}
}

func effectiveLimit(configured int, fallback int) int {
	if configured > 0 {
		return configured
	}
	return fallback
}

func (state *completionBudgetState) consumeProviderAttempt() error {
	state.providerAttempts++
	if state.providerAttempts > state.maxProviderAttempts {
		return &CompletionLimitError{Limit: "provider_attempts", Used: int64(state.providerAttempts), Max: int64(state.maxProviderAttempts)}
	}
	return nil
}

func (state *completionBudgetState) consumeToolCall(call ToolCall) error {
	state.toolCalls++
	if state.toolCalls > state.maxToolCalls {
		return &CompletionLimitError{Limit: "tool_calls", Used: int64(state.toolCalls), Max: int64(state.maxToolCalls)}
	}
	key := call.Name + "\x00" + canonicalJSON(call.Arguments)
	state.repeatedToolCalls[key]++
	if state.repeatedToolCalls[key] > state.maxRepeatedToolCalls {
		return &CompletionLimitError{
			Limit: "repeated_tool_calls",
			Used:  int64(state.repeatedToolCalls[key]),
			Max:   int64(state.maxRepeatedToolCalls),
		}
	}
	return nil
}

func (state *completionBudgetState) consumeApprovalDenial() error {
	state.approvalDenials++
	if state.approvalDenials >= state.maxApprovalDenials {
		return &CompletionLimitError{Limit: "approval_denials", Used: int64(state.approvalDenials), Max: int64(state.maxApprovalDenials)}
	}
	return nil
}

func (state *completionBudgetState) checkUsage(output *CompletionCallOutput, response *ChatResponse) error {
	if state.maxTotalTokens <= 0 {
		return nil
	}
	if response == nil || response.Usage == nil {
		return &CompletionLimitError{Limit: "total_tokens", Max: int64(state.maxTotalTokens), Err: ErrCompletionUsageUnavailable}
	}
	if output.Usage.TotalTokens > int64(state.maxTotalTokens) {
		return &CompletionLimitError{Limit: "total_tokens", Used: output.Usage.TotalTokens, Max: int64(state.maxTotalTokens)}
	}
	return nil
}

func canonicalJSON(value string) string {
	var decoded any
	decoder := json.NewDecoder(strings.NewReader(value))
	decoder.UseNumber()
	if err := decoder.Decode(&decoded); err != nil {
		return value
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return value
	}
	data, err := json.Marshal(decoded)
	if err != nil {
		return value
	}
	return string(data)
}

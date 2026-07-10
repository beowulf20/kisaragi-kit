package llm

import (
	"context"
	"errors"
	"testing"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestCompletionCapsProviderAttemptsAcrossRetries(t *testing.T) {
	client := &fakeChatClient{
		errors:    []error{errors.New("temporary")},
		responses: []*ChatResponse{{Content: "unreachable"}},
	}
	output, err := Completion(CompletionCallInput{
		Client:              client,
		Model:               "test-model",
		Messages:            []Message{NewUserMessage("hello")},
		MaxProviderAttempts: 1,
	})
	if err == nil || !errors.Is(err, ErrCompletionLimitExceeded) {
		t.Fatalf("expected provider-attempt limit, got output=%#v err=%v", output, err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
}

func TestCompletionCapsTotalToolCallsBeforeNextHandler(t *testing.T) {
	called := 0
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("count", "Counts calls.", func(context.Context, struct{}) (struct{}, error) {
		called++
		return struct{}{}, nil
	})); err != nil {
		t.Fatal(err)
	}
	client := &fakeChatClient{responses: []*ChatResponse{{ToolCalls: []ToolCall{
		{ID: "call_1", Name: "count", Arguments: `{}`},
		{ID: "call_2", Name: "count", Arguments: `{"ignored":true}`},
	}}}}

	_, err := Completion(CompletionCallInput{
		Client:       client,
		Model:        "test-model",
		Messages:     []Message{NewUserMessage("count")},
		Tools:        *tools,
		MaxToolCalls: 1,
	})
	if err == nil || !errors.Is(err, ErrCompletionLimitExceeded) {
		t.Fatalf("expected tool-call limit, got %v", err)
	}
	if called != 1 {
		t.Fatalf("handler calls = %d, want 1", called)
	}
}

func TestCompletionCanonicalizesRepeatedToolCallArguments(t *testing.T) {
	called := 0
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("sum", "Sums.", func(_ context.Context, input struct {
		A int `json:"a"`
		B int `json:"b"`
	}) (int, error) {
		called++
		return input.A + input.B, nil
	})); err != nil {
		t.Fatal(err)
	}
	client := &fakeChatClient{responses: []*ChatResponse{
		{ToolCalls: []ToolCall{{ID: "call_1", Name: "sum", Arguments: `{"a":1,"b":2}`}}},
		{ToolCalls: []ToolCall{{ID: "call_2", Name: "sum", Arguments: `{"b":2,"a":1}`}}},
	}}

	_, err := Completion(CompletionCallInput{
		Client:               client,
		Model:                "test-model",
		Messages:             []Message{NewUserMessage("sum")},
		Tools:                *tools,
		MaxRepeatedToolCalls: 1,
	})
	if err == nil || !errors.Is(err, ErrCompletionLimitExceeded) {
		t.Fatalf("expected repeated-call limit, got %v", err)
	}
	if called != 1 {
		t.Fatalf("handler calls = %d, want 1", called)
	}
}

func TestCanonicalJSONPreservesLargeIntegers(t *testing.T) {
	value := canonicalJSON(`{"id":9007199254740993}`)
	if value != `{"id":9007199254740993}` {
		t.Fatalf("canonical JSON = %s", value)
	}
}

func TestCompletionCapsApprovalDenials(t *testing.T) {
	tools := llmtool.NewToolbox(llmtool.WithApprovalHook(func(context.Context, llmtool.ApprovalRequest) (llmtool.ApprovalDecision, error) {
		return llmtool.ApprovalDecision{Approved: false}, nil
	}))
	if err := tools.RegisterTool(llmtool.NewTool("risky", "Risky.", func(context.Context, struct{}) (struct{}, error) {
		t.Fatal("handler should not run")
		return struct{}{}, nil
	}, llmtool.WithApproval(llmtool.ApprovalPolicy{Mode: llmtool.ApprovalAlways}))); err != nil {
		t.Fatal(err)
	}
	client := &fakeChatClient{responses: []*ChatResponse{{ToolCalls: []ToolCall{{ID: "call_1", Name: "risky", Arguments: `{}`}}}}}

	_, err := Completion(CompletionCallInput{
		Client:             client,
		Model:              "test-model",
		Messages:           []Message{NewUserMessage("risky")},
		Tools:              *tools,
		MaxApprovalDenials: 1,
	})
	if err == nil || !errors.Is(err, ErrCompletionLimitExceeded) {
		t.Fatalf("expected approval-denial limit, got %v", err)
	}
}

func TestCompletionCapsReportedTotalTokens(t *testing.T) {
	client := &fakeChatClient{responses: []*ChatResponse{{
		Content: "done",
		Usage:   &TokenUsage{TotalTokens: 11},
	}}}
	output, err := Completion(CompletionCallInput{
		Client:         client,
		Model:          "test-model",
		Messages:       []Message{NewUserMessage("hello")},
		MaxTotalTokens: 10,
	})
	if err == nil || !errors.Is(err, ErrCompletionLimitExceeded) {
		t.Fatalf("expected token limit, got output=%#v err=%v", output, err)
	}
	if output == nil || output.Usage.TotalTokens != 11 {
		t.Fatalf("output = %#v, want recorded usage", output)
	}
}

func TestCompletionRequiresUsageBeforeUnmeteredToolRound(t *testing.T) {
	client := &fakeChatClient{responses: []*ChatResponse{{
		ToolCalls: []ToolCall{{ID: "call_1", Name: "missing", Arguments: `{}`}},
	}}}
	_, err := Completion(CompletionCallInput{
		Client:         client,
		Model:          "test-model",
		Messages:       []Message{NewUserMessage("hello")},
		MaxTotalTokens: 10,
	})
	if err == nil || !errors.Is(err, ErrCompletionUsageUnavailable) {
		t.Fatalf("expected usage unavailable error, got %v", err)
	}
}

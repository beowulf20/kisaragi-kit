package llm

import (
	"context"
	"errors"
	"testing"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestCompletionToolPolicyDenialStopsRemainingToolCalls(t *testing.T) {
	firstCalled := 0
	lastCalled := 0
	tools := llmtool.NewToolbox(llmtool.WithToolPolicyHook(func(_ context.Context, request llmtool.ToolPolicyRequest) (llmtool.ToolPolicyDecision, error) {
		if request.ToolName == "deny" {
			return llmtool.ToolPolicyDecision{Action: llmtool.ToolPolicyDeny, Reason: "blocked by application"}, nil
		}
		return llmtool.ToolPolicyDecision{Action: llmtool.ToolPolicyAllow}, nil
	}))
	for _, registration := range []struct {
		name string
		call func(context.Context, struct{}) (struct{}, error)
	}{
		{name: "first", call: func(context.Context, struct{}) (struct{}, error) {
			firstCalled++
			return struct{}{}, nil
		}},
		{name: "deny", call: func(context.Context, struct{}) (struct{}, error) {
			t.Fatal("denied handler should not run")
			return struct{}{}, nil
		}},
		{name: "last", call: func(context.Context, struct{}) (struct{}, error) {
			lastCalled++
			return struct{}{}, nil
		}},
	} {
		if err := tools.RegisterTool(llmtool.NewTool(registration.name, registration.name, registration.call)); err != nil {
			t.Fatal(err)
		}
	}
	client := &fakeChatClient{responses: []*ChatResponse{{ToolCalls: []ToolCall{
		{ID: "call_1", Name: "first", Arguments: `{}`},
		{ID: "call_2", Name: "deny", Arguments: `{}`},
		{ID: "call_3", Name: "last", Arguments: `{}`},
	}}}}

	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("run tools")},
		Tools:    *tools,
	})
	if err == nil || !errors.Is(err, llmtool.ErrToolPolicyDenied) {
		t.Fatalf("expected tool policy denial, got output=%#v err=%v", output, err)
	}
	if firstCalled != 1 || lastCalled != 0 {
		t.Fatalf("first=%d last=%d, want first only", firstCalled, lastCalled)
	}
	if output == nil || len(output.ToolCalls) != 1 || output.ToolCalls[0].Name != "first" {
		t.Fatalf("output = %#v, want first result only", output)
	}
}

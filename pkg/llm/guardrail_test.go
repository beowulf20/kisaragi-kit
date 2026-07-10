package llm

import (
	"context"
	"errors"
	"strings"
	"testing"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestCompletionChecksEveryInitialMessageRole(t *testing.T) {
	var types []MessageType
	guardrail := NewMessageGuardrail("record", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseInput {
			types = append(types, input.Message.Type)
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := &fakeChatClient{responses: []*ChatResponse{{Content: "done"}}}

	_, err := Completion(CompletionCallInput{
		Client: client,
		Model:  "test-model",
		Messages: []Message{
			NewSystemMessage("system"),
			NewUserMessage("user"),
			NewAssistantToolCallMessage("assistant", []ToolCall{{ID: "call_1", Name: "lookup", Arguments: `{}`}}),
			NewToolMessage("call_1", "tool"),
		},
		MessageGuardrails: []MessageGuardrail{guardrail},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []MessageType{System, User, Assistant, Tool}
	if len(types) != len(want) {
		t.Fatalf("types = %v, want %v", types, want)
	}
	for index := range want {
		if types[index] != want[index] {
			t.Fatalf("types = %v, want %v", types, want)
		}
	}
}

func TestCompletionMessageGuardrailStopsBeforeBlockedDelta(t *testing.T) {
	guardrail := NewMessageGuardrail("secret", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseAssistantContentDelta && strings.Contains(input.Message.Content, "secret") {
			return MessageGuardrailDecision{Action: MessageGuardrailBlock, Reason: "secret detected"}, nil
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := streamingGuardrailClient{contentDeltas: []string{"safe ", "secret", " unreachable"}}
	var observed []string

	output, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{guardrail},
		Hooks: CompletionHooks{OnContentDelta: func(delta string) {
			observed = append(observed, delta)
		}},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected guardrail error, got output=%#v err=%v", output, err)
	}
	if output == nil || output.Content != "safe " {
		t.Fatalf("output = %#v, want safe prefix", output)
	}
	if strings.Join(observed, "") != "safe " {
		t.Fatalf("observed = %q, want safe prefix", strings.Join(observed, ""))
	}
	if len(output.Messages) != 0 {
		t.Fatalf("messages = %#v, want blocked candidate omitted", output.Messages)
	}
	var guardrailErr *MessageGuardrailError
	if !errors.As(err, &guardrailErr) || guardrailErr.Phase != MessageGuardrailPhaseAssistantContentDelta {
		t.Fatalf("guardrail error = %#v", guardrailErr)
	}
	if strings.Contains(err.Error(), "secret detected payload") {
		t.Fatalf("error exposed candidate: %v", err)
	}
}

func TestCompletionMessageGuardrailChecksReasoningDeltas(t *testing.T) {
	guardrail := NewMessageGuardrail("reasoning", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseAssistantReasoningDelta && strings.Contains(input.Message.Content, "blocked") {
			return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := streamingGuardrailClient{reasoningDeltas: []string{"safe ", "blocked"}}
	var observed []string

	_, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{guardrail},
		Hooks: CompletionHooks{OnReasoningDelta: func(delta string) {
			observed = append(observed, delta)
		}},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected guardrail error, got %v", err)
	}
	if strings.Join(observed, "") != "safe " {
		t.Fatalf("observed = %q, want safe reasoning prefix", strings.Join(observed, ""))
	}
}

func TestCompletionMessageGuardrailChecksFinalResponse(t *testing.T) {
	guardrail := NewMessageGuardrail("final", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseAssistantFinal {
			return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := &fakeChatClient{responses: []*ChatResponse{{Content: "blocked final"}}}

	output, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{guardrail},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected final guardrail error, got %v", err)
	}
	if output == nil || output.Content != "" || len(output.Messages) != 0 {
		t.Fatalf("output = %#v, want empty non-streamed candidate", output)
	}
}

func TestCompletionRetainsGuardrailBlockWhenClientIgnoresDeltaError(t *testing.T) {
	guardrail := NewMessageGuardrail("secret", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseAssistantContentDelta && strings.Contains(input.Message.Content, "secret") {
			return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := &fakeChatClient{
		responses: []*ChatResponse{{Content: "safe secret unreachable"}},
		onComplete: func(_ ChatRequest, hooks CompletionHooks) {
			hooks.EmitContentDelta("safe ")
			hooks.EmitContentDelta("secret")
			hooks.EmitContentDelta(" unreachable")
		},
	}
	var observed []string
	output, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{guardrail},
		Hooks: CompletionHooks{OnContentDelta: func(delta string) {
			observed = append(observed, delta)
		}},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected retained guardrail block, got %v", err)
	}
	if output == nil || output.Content != "safe " || strings.Join(observed, "") != "safe " {
		t.Fatalf("output=%#v observed=%q, want safe prefix", output, strings.Join(observed, ""))
	}
}

func TestCompletionGuardsClientsThatInvokeDeltaCallbacksDirectly(t *testing.T) {
	guardrail := NewMessageGuardrail("secret", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseAssistantContentDelta && strings.Contains(input.Message.Content, "secret") {
			return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := directCallbackClient{}
	var observed []string
	output, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{guardrail},
		Hooks: CompletionHooks{OnContentDelta: func(delta string) {
			observed = append(observed, delta)
		}},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected direct-callback guardrail block, got %v", err)
	}
	if output == nil || output.Content != "safe " || strings.Join(observed, "") != "safe " {
		t.Fatalf("output=%#v observed=%q, want safe prefix", output, strings.Join(observed, ""))
	}
}

func TestCompletionMessageGuardrailBlocksToolResult(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("secret", "Returns secret data.", func(context.Context, struct{}) (string, error) {
		return "secret result", nil
	})); err != nil {
		t.Fatal(err)
	}
	guardrail := NewMessageGuardrail("tool-result", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Phase == MessageGuardrailPhaseToolResult {
			return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	client := &fakeChatClient{responses: []*ChatResponse{
		{ToolCalls: []ToolCall{{ID: "call_1", Name: "secret", Arguments: `{}`}}},
		{Content: "unreachable"},
	}}

	output, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("call secret")},
		Tools:             *tools,
		MessageGuardrails: []MessageGuardrail{guardrail},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected tool result block, got %v", err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if output == nil || len(output.Messages) != 1 || output.Messages[0].Type != Assistant {
		t.Fatalf("output = %#v, want assistant tool-call transcript only", output)
	}
}

func TestMessageGuardrailsReceiveIsolatedCopies(t *testing.T) {
	first := NewMessageGuardrail("mutator", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		input.Message.Content = "changed"
		input.Messages[0].Content = "changed context"
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	second := NewMessageGuardrail("observer", func(_ context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
		if input.Message.Content != "candidate" || input.Messages[0].Content != "context" {
			t.Fatalf("input was mutated across guardrails: %#v", input)
		}
		return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
	})
	err := evaluateMessageGuardrails(context.Background(), []MessageGuardrail{first, second}, MessageGuardrailInput{
		Message:  NewAssistantMessage("candidate"),
		Messages: []Message{NewSystemMessage("context")},
		Phase:    MessageGuardrailPhaseAssistantFinal,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCompletionRejectsInvalidMessageGuardrails(t *testing.T) {
	input := CompletionCallInput{
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{
			NewMessageGuardrail("duplicate", func(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error) {
				return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
			}),
			NewMessageGuardrail("duplicate", func(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error) {
				return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
			}),
		},
	}
	if err := input.Validate(); err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("expected duplicate error, got %v", err)
	}
}

func TestCompletionInvalidMessageGuardrailDecisionFailsClosed(t *testing.T) {
	guardrail := NewMessageGuardrail("invalid", func(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error) {
		return MessageGuardrailDecision{}, nil
	})
	client := &fakeChatClient{responses: []*ChatResponse{{Content: "unreachable"}}}
	_, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("hello")},
		MessageGuardrails: []MessageGuardrail{guardrail},
	})
	if err == nil || !errors.Is(err, ErrMessageGuardrailBlocked) {
		t.Fatalf("expected fail-closed guardrail error, got %v", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(client.requests))
	}
}

type streamingGuardrailClient struct {
	contentDeltas   []string
	reasoningDeltas []string
}

type directCallbackClient struct{}

func (directCallbackClient) Complete(_ context.Context, _ ChatRequest, hooks CompletionHooks) (*ChatResponse, error) {
	for _, delta := range []string{"safe ", "secret", " unreachable"} {
		if hooks.OnContentDelta != nil {
			hooks.OnContentDelta(delta)
		}
	}
	return &ChatResponse{Content: "safe secret unreachable"}, nil
}

func (directCallbackClient) ListModels(context.Context) ([]string, error) {
	return nil, nil
}

func (client streamingGuardrailClient) Complete(_ context.Context, _ ChatRequest, hooks CompletionHooks) (*ChatResponse, error) {
	for _, delta := range client.reasoningDeltas {
		if err := hooks.EmitReasoningDelta(delta); err != nil {
			return nil, err
		}
	}
	for _, delta := range client.contentDeltas {
		if err := hooks.EmitContentDeltaEvent(delta); err != nil {
			return nil, err
		}
	}
	return &ChatResponse{
		Content:   strings.Join(client.contentDeltas, ""),
		Reasoning: strings.Join(client.reasoningDeltas, ""),
	}, nil
}

func (streamingGuardrailClient) ListModels(context.Context) ([]string, error) {
	return nil, nil
}

package llm

import (
	"context"
	"errors"
	"strings"
	"testing"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestShortToolError(t *testing.T) {
	result := shortToolError(errors.New("device 99 not found"))

	if !strings.Contains(result, `"error":"device 99 not found"`) {
		t.Fatalf("result = %s", result)
	}
}

func TestCompletionStreamsContentDeltas(t *testing.T) {
	client := &fakeChatClient{
		responses: []*ChatResponse{{Content: "hello world"}},
		onComplete: func(_ ChatRequest, hooks CompletionHooks) {
			hooks.EmitContentDelta("hello")
			hooks.EmitContentDelta(" world")
		},
	}

	var deltas []string
	output, err := Completion(CompletionCallInput{
		Client: client,
		Model:  "test-model",
		Messages: []Message{
			NewUserMessage("say hello"),
		},
		Hooks: CompletionHooks{
			OnContentDelta: func(delta string) {
				deltas = append(deltas, delta)
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "hello world" {
		t.Fatalf("content = %q, want hello world", output.Content)
	}
	if strings.Join(deltas, "") != "hello world" {
		t.Fatalf("deltas = %q, want hello world", strings.Join(deltas, ""))
	}
}

func TestCompletionRunsToolLoop(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("greet", "Greets a person.", func(_ context.Context, input struct {
		Name string `json:"name"`
	}) (struct {
		Greeting string `json:"greeting"`
	}, error) {
		return struct {
			Greeting string `json:"greeting"`
		}{Greeting: "hello " + input.Name}, nil
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{
				ToolCalls: []ToolCall{
					{ID: "call_123", Name: "greet", Arguments: `{"name":"Ada"}`},
				},
			},
			{Content: "done"},
		},
	}

	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("say hello")},
		Tools:    *tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}

	secondRequest := client.requests[1]
	if len(secondRequest.Messages) != 3 {
		t.Fatalf("second request messages = %d, want 3", len(secondRequest.Messages))
	}
	if secondRequest.Messages[1].Type != Assistant || len(secondRequest.Messages[1].ToolCalls) != 1 {
		t.Fatalf("assistant tool call message = %#v", secondRequest.Messages[1])
	}
	if secondRequest.Messages[2].Type != Tool || !strings.Contains(secondRequest.Messages[2].Content, "hello Ada") {
		t.Fatalf("tool result message = %#v", secondRequest.Messages[2])
	}
}

func TestResolveModelUsesFirstListedModel(t *testing.T) {
	client := &fakeChatClient{models: []string{"model-a", "model-b"}}

	model, err := ResolveModel(context.Background(), client, "")
	if err != nil {
		t.Fatal(err)
	}
	if model != "model-a" {
		t.Fatalf("model = %q, want model-a", model)
	}
}

type fakeChatClient struct {
	responses  []*ChatResponse
	requests   []ChatRequest
	models     []string
	onComplete func(ChatRequest, CompletionHooks)
}

func (c *fakeChatClient) Complete(_ context.Context, request ChatRequest, hooks CompletionHooks) (*ChatResponse, error) {
	c.requests = append(c.requests, request)
	if c.onComplete != nil {
		c.onComplete(request, hooks)
	}
	if len(c.responses) == 0 {
		return &ChatResponse{}, nil
	}
	response := c.responses[0]
	c.responses = c.responses[1:]
	return response, nil
}

func (c *fakeChatClient) ListModels(context.Context) ([]string, error) {
	return c.models, nil
}

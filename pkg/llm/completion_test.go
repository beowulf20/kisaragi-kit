package llm

import (
	"context"
	"errors"
	"strings"
	"testing"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestShortToolError(t *testing.T) {
	result := shortToolError(errors.New("device 99 not found"), DefaultMaxToolErrorLength)

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

func TestCompletionRetriesProviderErrorsByDefault(t *testing.T) {
	client := &fakeChatClient{
		errors:    []error{errors.New("temporary one"), errors.New("temporary two")},
		responses: []*ChatResponse{{Content: "done"}},
	}

	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if len(client.requests) != DefaultProviderErrorRetries+1 {
		t.Fatalf("requests = %d, want %d", len(client.requests), DefaultProviderErrorRetries+1)
	}
}

func TestCompletionUsesConfiguredProviderErrorRetries(t *testing.T) {
	retries := 0
	client := &fakeChatClient{
		errors:    []error{errors.New("temporary")},
		responses: []*ChatResponse{{Content: "done"}},
	}

	_, err := Completion(CompletionCallInput{
		Client:               client,
		Model:                "test-model",
		Messages:             []Message{NewUserMessage("hello")},
		ProviderErrorRetries: &retries,
	})
	if err == nil || !strings.Contains(err.Error(), "temporary") {
		t.Fatalf("expected provider error, got %v", err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
}

func TestCompletionUsesConfiguredMaxToolCallRounds(t *testing.T) {
	client := &fakeChatClient{
		responses: []*ChatResponse{
			{ToolCalls: []ToolCall{{ID: "call_1", Name: "missing", Arguments: `{}`}}},
			{ToolCalls: []ToolCall{{ID: "call_2", Name: "missing", Arguments: `{}`}}},
		},
	}

	_, err := Completion(CompletionCallInput{
		Client:            client,
		Model:             "test-model",
		Messages:          []Message{NewUserMessage("call tools")},
		MaxToolCallRounds: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "exceeded 1 tool call rounds") {
		t.Fatalf("expected max rounds error, got %v", err)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
}

func TestCompletionUsesConfiguredMaxToolErrorLength(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("fail", "Fails.", func(context.Context, struct{}) (struct{}, error) {
		return struct{}{}, errors.New("abcdef")
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{ToolCalls: []ToolCall{{ID: "call_1", Name: "fail", Arguments: `{}`}}},
			{Content: "done"},
		},
	}

	output, err := Completion(CompletionCallInput{
		Client:             client,
		Model:              "test-model",
		Messages:           []Message{NewUserMessage("call tool")},
		Tools:              *tools,
		MaxToolErrorLength: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(output.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(output.ToolCalls))
	}
	if !strings.Contains(output.ToolCalls[0].Result, `"error":"abc"`) {
		t.Fatalf("tool result = %s, want truncated error", output.ToolCalls[0].Result)
	}
}

func TestCompletionUsesToolErrorInterceptorFeedback(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("weather", "Gets weather.", func(context.Context, struct {
		City string `json:"city"`
	}) (struct{}, error) {
		return struct{}{}, errors.New("missing city")
	})); err != nil {
		t.Fatal(err)
	}

	feedback := `{"error":"missing city","retryable":true,"hint":"include city"}`
	client := &fakeChatClient{
		responses: []*ChatResponse{
			{ToolCalls: []ToolCall{{ID: "call_1", Name: "weather", Arguments: `{}`}}},
			{Content: "done"},
		},
	}

	var sawContext bool
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("weather please")},
		Tools:    *tools,
		ToolErrorInterceptor: func(ctx ToolErrorContext) ToolErrorDecision {
			sawContext = ctx.ToolCall.Name == "weather" &&
				ctx.Round == 0 &&
				strings.Contains(ctx.DefaultFeedback, "missing city") &&
				strings.Contains(ctx.Err.Error(), "missing city")
			return ToolErrorDecision{Feedback: feedback}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !sawContext {
		t.Fatal("interceptor context was not populated")
	}
	if len(output.ToolCalls) != 1 || output.ToolCalls[0].Result != feedback {
		t.Fatalf("tool calls = %#v, want interceptor feedback", output.ToolCalls)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	if got := client.requests[1].Messages[2].Content; got != feedback {
		t.Fatalf("feedback message = %s, want %s", got, feedback)
	}
}

func TestCompletionToolErrorInterceptorCanAbort(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("fail", "Fails.", func(context.Context, struct{}) (struct{}, error) {
		return struct{}{}, errors.New("stop now")
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{ToolCalls: []ToolCall{{ID: "call_1", Name: "fail", Arguments: `{}`}}},
			{Content: "unreachable"},
		},
	}

	_, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("call tool")},
		Tools:    *tools,
		ToolErrorInterceptor: func(ToolErrorContext) ToolErrorDecision {
			return ToolErrorDecision{Abort: true}
		},
	})
	if err == nil || !strings.Contains(err.Error(), `tool "fail" failed`) {
		t.Fatalf("expected abort error, got %v", err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
}

func TestCompletionRejectsNegativeLimits(t *testing.T) {
	input := CompletionCallInput{
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
	}

	input.MaxToolCallRounds = -1
	if err := input.Validate(); err == nil || !strings.Contains(err.Error(), "rounds") {
		t.Fatalf("expected max tool call rounds error, got %v", err)
	}

	input.MaxToolCallRounds = 0
	input.MaxToolErrorLength = -1
	if err := input.Validate(); err == nil || !strings.Contains(err.Error(), "error length") {
		t.Fatalf("expected max tool error length error, got %v", err)
	}

	retries := -1
	input.MaxToolErrorLength = 0
	input.ProviderErrorRetries = &retries
	if err := input.Validate(); err == nil || !strings.Contains(err.Error(), "provider error retries") {
		t.Fatalf("expected provider error retries error, got %v", err)
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
	errors     []error
	requests   []ChatRequest
	models     []string
	onComplete func(ChatRequest, CompletionHooks)
}

func (c *fakeChatClient) Complete(_ context.Context, request ChatRequest, hooks CompletionHooks) (*ChatResponse, error) {
	c.requests = append(c.requests, request)
	if c.onComplete != nil {
		c.onComplete(request, hooks)
	}
	if len(c.errors) > 0 {
		err := c.errors[0]
		c.errors = c.errors[1:]
		return nil, err
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

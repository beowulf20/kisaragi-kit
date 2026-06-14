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

func TestCompletionForwardsReasoningEffort(t *testing.T) {
	client := &fakeChatClient{
		responses: []*ChatResponse{{Content: "done"}},
	}

	output, err := Completion(CompletionCallInput{
		Client:          client,
		Model:           "test-model",
		Messages:        []Message{NewUserMessage("think lightly")},
		ReasoningEffort: ReasoningEffortLow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if client.requests[0].ReasoningEffort != ReasoningEffortLow {
		t.Fatalf("reasoning effort = %q, want %q", client.requests[0].ReasoningEffort, ReasoningEffortLow)
	}
}

func TestCompletionUsesCallerContext(t *testing.T) {
	type contextKey string
	key := contextKey("request-id")
	ctx := context.WithValue(context.Background(), key, "req-123")

	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("check_context", "Checks context.", func(ctx context.Context, _ struct{}) (struct {
		RequestID string `json:"request_id"`
	}, error) {
		value, _ := ctx.Value(key).(string)
		return struct {
			RequestID string `json:"request_id"`
		}{RequestID: value}, nil
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{ToolCalls: []ToolCall{{ID: "call_1", Name: "check_context", Arguments: `{}`}}},
			{Content: "done"},
		},
	}

	output, err := Completion(CompletionCallInput{
		Context:  ctx,
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("check context")},
		Tools:    *tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.contexts) != 2 || client.contexts[0].Value(key) != "req-123" || client.contexts[1].Value(key) != "req-123" {
		t.Fatalf("provider contexts = %#v, want caller context", client.contexts)
	}
	if len(output.ToolCalls) != 1 || !strings.Contains(output.ToolCalls[0].Result, `"request_id":"req-123"`) {
		t.Fatalf("tool calls = %#v, want caller context in result", output.ToolCalls)
	}
}

func TestCompletionEmitsGenerationLifecycle(t *testing.T) {
	client := &fakeChatClient{
		responses: []*ChatResponse{{Content: "done"}},
	}

	var starts []GenerationStartEvent
	var ends []GenerationEndEvent
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		Hooks: CompletionHooks{
			OnGenerationStart: func(event GenerationStartEvent) {
				starts = append(starts, event)
			},
			OnGenerationEnd: func(event GenerationEndEvent) {
				ends = append(ends, event)
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if len(starts) != 1 {
		t.Fatalf("starts = %d, want 1", len(starts))
	}
	if starts[0].Model != "test-model" || starts[0].Round != 0 || starts[0].Attempt != 0 || starts[0].MessageCount != 1 || starts[0].AvailableToolCount != 0 {
		t.Fatalf("start = %#v, want first test-model attempt", starts[0])
	}
	if len(ends) != 1 {
		t.Fatalf("ends = %d, want 1", len(ends))
	}
	if ends[0].Model != "test-model" || ends[0].Round != 0 || ends[0].Attempt != 0 || ends[0].MessageCount != 1 || ends[0].ToolCallCount != 0 || ends[0].Err != nil {
		t.Fatalf("end = %#v, want successful first attempt", ends[0])
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

func TestCompletionCanAppendAcceptedApprovalDecisionMessages(t *testing.T) {
	tools := llmtool.NewToolbox(llmtool.WithApprovalHook(func(context.Context, llmtool.ApprovalRequest) (llmtool.ApprovalDecision, error) {
		return llmtool.ApprovalDecision{Approved: true, Reason: "looks safe"}, nil
	}))
	if err := tools.RegisterTool(llmtool.NewTool("greet", "Greets a person.", func(_ context.Context, input struct {
		Name string `json:"name"`
	}) (struct {
		Greeting string `json:"greeting"`
	}, error) {
		return struct {
			Greeting string `json:"greeting"`
		}{Greeting: "hello " + input.Name}, nil
	}, llmtool.WithApproval(llmtool.ApprovalPolicy{
		Mode: llmtool.ApprovalAlways,
		Risk: llmtool.RiskMedium,
	}))); err != nil {
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
		ApprovalDecisionMessages: ApprovalDecisionMessages{
			AppendAccepted: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(output.Messages) != 4 {
		t.Fatalf("output messages = %d, want 4", len(output.Messages))
	}
	approvalMessage := output.Messages[2]
	if approvalMessage.Type != User || !strings.Contains(approvalMessage.Content, `"tool_approval":"accepted"`) || !strings.Contains(approvalMessage.Content, `"reason":"looks safe"`) {
		t.Fatalf("approval message = %#v, want accepted user transcript", approvalMessage)
	}
	if len(client.requests[1].Messages) != 3 {
		t.Fatalf("second request messages = %d, want approval omitted from current provider loop", len(client.requests[1].Messages))
	}
}

func TestCompletionCanAppendRejectedApprovalDecisionMessages(t *testing.T) {
	tools := llmtool.NewToolbox(llmtool.WithApprovalHook(func(context.Context, llmtool.ApprovalRequest) (llmtool.ApprovalDecision, error) {
		return llmtool.ApprovalDecision{Approved: false, Reason: "too risky"}, nil
	}))
	if err := tools.RegisterTool(llmtool.NewTool("greet", "Greets a person.", func(context.Context, struct{}) (struct{}, error) {
		return struct{}{}, nil
	}, llmtool.WithApproval(llmtool.ApprovalPolicy{
		Mode: llmtool.ApprovalAlways,
		Risk: llmtool.RiskHigh,
	}))); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{
				ToolCalls: []ToolCall{
					{ID: "call_123", Name: "greet", Arguments: `{}`},
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
		ApprovalDecisionMessages: ApprovalDecisionMessages{
			AppendRejected: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(output.Messages) != 4 {
		t.Fatalf("output messages = %d, want 4", len(output.Messages))
	}
	approvalMessage := output.Messages[2]
	if approvalMessage.Type != User || !strings.Contains(approvalMessage.Content, `"tool_approval":"rejected"`) || !strings.Contains(approvalMessage.Content, `"reason":"too risky"`) {
		t.Fatalf("approval message = %#v, want rejected user transcript", approvalMessage)
	}
	if len(client.requests[1].Messages) != 3 {
		t.Fatalf("second request messages = %d, want approval omitted from current provider loop", len(client.requests[1].Messages))
	}
}

func TestCompletionEmitsGenerationLifecycleForProviderRetries(t *testing.T) {
	client := &fakeChatClient{
		errors:    []error{errors.New("temporary")},
		responses: []*ChatResponse{{Content: "done"}},
	}

	var starts []GenerationStartEvent
	var ends []GenerationEndEvent
	var callErrors []error
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		Hooks: CompletionHooks{
			OnGenerationStart: func(event GenerationStartEvent) {
				starts = append(starts, event)
			},
			OnGenerationEnd: func(event GenerationEndEvent) {
				ends = append(ends, event)
			},
			OnCallError: func(err error) {
				callErrors = append(callErrors, err)
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if len(starts) != 2 || starts[0].Attempt != 0 || starts[1].Attempt != 1 {
		t.Fatalf("starts = %#v, want attempts 0 and 1", starts)
	}
	if len(ends) != 2 {
		t.Fatalf("ends = %d, want 2", len(ends))
	}
	if ends[0].Attempt != 0 || ends[0].Err == nil || ends[0].Err.Error() != "temporary" {
		t.Fatalf("first end = %#v, want temporary error", ends[0])
	}
	if ends[1].Attempt != 1 || ends[1].Err != nil {
		t.Fatalf("second end = %#v, want successful retry", ends[1])
	}
	if len(callErrors) != 1 || callErrors[0].Error() != "temporary" {
		t.Fatalf("call errors = %#v, want temporary", callErrors)
	}
}

func TestCompletionEmitsAndAggregatesUsage(t *testing.T) {
	client := &fakeChatClient{
		responses: []*ChatResponse{{
			Content: "done",
			Usage: &TokenUsage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
				PromptTokenDetails: map[string]int64{
					"cached_tokens": 4,
				},
				CompletionTokenDetails: map[string]int64{
					"reasoning_tokens": 7,
				},
			},
		}},
	}

	var events []UsageEvent
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		Hooks: CompletionHooks{
			OnUsage: func(event UsageEvent) {
				events = append(events, event)
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Fatalf("usage events = %d, want 1", len(events))
	}
	if events[0].Model != "test-model" || events[0].Round != 0 || events[0].Attempt != 0 {
		t.Fatalf("usage event = %#v, want first test-model generation", events[0])
	}
	if output.Usage.PromptTokens != 10 || output.Usage.CompletionTokens != 20 || output.Usage.TotalTokens != 30 {
		t.Fatalf("usage = %#v, want 10/20/30", output.Usage)
	}
	if output.Usage.PromptTokenDetails["cached_tokens"] != 4 || output.Usage.CompletionTokenDetails["reasoning_tokens"] != 7 {
		t.Fatalf("usage details = %#v %#v", output.Usage.PromptTokenDetails, output.Usage.CompletionTokenDetails)
	}
	if len(output.UsageEvents) != 1 || output.UsageEvents[0].Usage.TotalTokens != 30 {
		t.Fatalf("output usage events = %#v, want one event", output.UsageEvents)
	}
}

func TestCompletionAggregatesUsageAcrossToolCallRounds(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("greet", "Greets.", func(_ context.Context, _ struct{}) (struct {
		Greeting string `json:"greeting"`
	}, error) {
		return struct {
			Greeting string `json:"greeting"`
		}{Greeting: "hello"}, nil
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{
				ToolCalls: []ToolCall{{ID: "call_1", Name: "greet", Arguments: `{}`}},
				Usage: &TokenUsage{
					PromptTokens:     1,
					CompletionTokens: 2,
					TotalTokens:      3,
					PromptTokenDetails: map[string]int64{
						"cached_tokens": 4,
					},
				},
			},
			{
				Content: "done",
				Usage: &TokenUsage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
					CompletionTokenDetails: map[string]int64{
						"reasoning_tokens": 5,
					},
				},
			},
		},
	}

	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		Tools:    *tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Usage.PromptTokens != 11 || output.Usage.CompletionTokens != 22 || output.Usage.TotalTokens != 33 {
		t.Fatalf("usage = %#v, want aggregate 11/22/33", output.Usage)
	}
	if output.Usage.PromptTokenDetails["cached_tokens"] != 4 || output.Usage.CompletionTokenDetails["reasoning_tokens"] != 5 {
		t.Fatalf("usage details = %#v %#v", output.Usage.PromptTokenDetails, output.Usage.CompletionTokenDetails)
	}
	if len(output.UsageEvents) != 2 || output.UsageEvents[0].Round != 0 || output.UsageEvents[1].Round != 1 {
		t.Fatalf("usage events = %#v, want two rounds", output.UsageEvents)
	}
}

func TestCompletionSkipsUsageWhenProviderOmitsUsage(t *testing.T) {
	client := &fakeChatClient{
		responses: []*ChatResponse{{Content: "done"}},
	}

	var usageEvents int
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		Hooks: CompletionHooks{
			OnUsage: func(UsageEvent) {
				usageEvents++
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if usageEvents != 0 {
		t.Fatalf("usage events = %d, want 0", usageEvents)
	}
	if output.Usage.TotalTokens != 0 || len(output.UsageEvents) != 0 {
		t.Fatalf("usage output = %#v events=%#v, want empty", output.Usage, output.UsageEvents)
	}
}

func TestChainCompletionHooksCallsBothInOrder(t *testing.T) {
	var calls []string
	hooks := ChainCompletionHooks(
		CompletionHooks{
			OnGenerationStart: func(GenerationStartEvent) {
				calls = append(calls, "first")
			},
		},
		CompletionHooks{
			OnGenerationStart: func(GenerationStartEvent) {
				calls = append(calls, "second")
			},
		},
	)

	hooks.EmitGenerationStart(GenerationStartEvent{Model: "test-model"})
	if strings.Join(calls, ",") != "first,second" {
		t.Fatalf("calls = %v, want first,second", calls)
	}
}

func TestCompletionRetriesProviderErrorsByDefault(t *testing.T) {
	client := &fakeChatClient{
		errors:    []error{errors.New("temporary one"), errors.New("temporary two")},
		responses: []*ChatResponse{{Content: "done"}},
	}

	var callErrors []string
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("hello")},
		Hooks: CompletionHooks{
			OnCallError: func(err error) {
				callErrors = append(callErrors, err.Error())
			},
		},
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
	if strings.Join(callErrors, ",") != "temporary one,temporary two" {
		t.Fatalf("call errors = %v, want temporary one and two", callErrors)
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

func TestCompletionEmitsToolError(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("fail", "Fails.", func(context.Context, struct{}) (struct{}, error) {
		return struct{}{}, errors.New("boom")
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*ChatResponse{
			{ToolCalls: []ToolCall{{ID: "call_1", Name: "fail", Arguments: `{}`}}},
			{Content: "done"},
		},
	}

	var gotCall ToolCall
	var gotErr error
	output, err := Completion(CompletionCallInput{
		Client:   client,
		Model:    "test-model",
		Messages: []Message{NewUserMessage("call tool")},
		Tools:    *tools,
		Hooks: CompletionHooks{
			OnToolError: func(toolCall ToolCall, err error) {
				gotCall = toolCall
				gotErr = err
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if gotCall.Name != "fail" || gotCall.ID != "call_1" {
		t.Fatalf("tool call = %#v, want fail call_1", gotCall)
	}
	if gotErr == nil || gotErr.Error() != "boom" {
		t.Fatalf("tool error = %v, want boom", gotErr)
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

func TestCompletionRejectsInvalidReasoningEffort(t *testing.T) {
	input := CompletionCallInput{
		Model:           "test-model",
		Messages:        []Message{NewUserMessage("hello")},
		ReasoningEffort: ReasoningEffort("extra-crunchy"),
	}

	if err := input.Validate(); err == nil || !strings.Contains(err.Error(), "reasoning effort") {
		t.Fatalf("expected reasoning effort error, got %v", err)
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
	contexts   []context.Context
	models     []string
	onComplete func(ChatRequest, CompletionHooks)
}

func (c *fakeChatClient) Complete(ctx context.Context, request ChatRequest, hooks CompletionHooks) (*ChatResponse, error) {
	c.contexts = append(c.contexts, ctx)
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

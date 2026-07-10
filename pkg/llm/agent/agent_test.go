package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestNewAgentValidatesInput(t *testing.T) {
	_, err := NewAgent(NewAgentInput{
		Config: llm.CompletionCallInput{
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "name") {
		t.Fatalf("expected name error, got %v", err)
	}

	_, err = NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "model") {
		t.Fatalf("expected model error, got %v", err)
	}

	_, err = NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Model: "test-model",
		},
	})
	if err == nil || !strings.Contains(err.Error(), "message") {
		t.Fatalf("expected messages error, got %v", err)
	}
}

func TestNewAgentCopiesMessages(t *testing.T) {
	messages := []llm.Message{llm.NewSystemMessage("test")}
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Model:    "test-model",
			Messages: messages,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	messages[0] = llm.NewSystemMessage("changed")

	got := agent.MessagesSnapshot()
	if got[0].Content != "test" {
		t.Fatalf("message content = %q, want test", got[0].Content)
	}
}

func TestNewAgentPrependsSystemPrompt(t *testing.T) {
	agent, err := NewAgent(NewAgentInput{
		Name:         "test",
		SystemPrompt: "be useful",
		Config: llm.CompletionCallInput{
			Model:    "test-model",
			Messages: []llm.Message{llm.NewUserMessage("hello")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	messages := agent.MessagesSnapshot()
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[0].Type != llm.System || messages[0].Content != "be useful" {
		t.Fatalf("first message = %#v, want system prompt", messages[0])
	}
}

func TestNewAgentAllowsSystemPromptWithoutConfigMessages(t *testing.T) {
	agent, err := NewAgent(NewAgentInput{
		Name:         "test",
		SystemPrompt: "be useful",
		Config: llm.CompletionCallInput{
			Model: "test-model",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	messages := agent.MessagesSnapshot()
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Type != llm.System || messages[0].Content != "be useful" {
		t.Fatalf("message = %#v, want system prompt", messages[0])
	}
}

func TestCallWithUserMessageRejectsEmptyInput(t *testing.T) {
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := agent.CallWithUserMessage(" "); err == nil {
		t.Fatal("expected empty input error")
	}
}

func TestRunWithTransientMessageDoesNotPersistTransientInput(t *testing.T) {
	client := &fakeChatClient{
		responses: []*llm.ChatResponse{{Content: "done"}},
	}

	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("system prompt")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := agent.RunWithTransientMessage(llm.NewUserMessage("transient reminder event")); err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if len(client.requests[0].Messages) != 2 || client.requests[0].Messages[1].Content != "transient reminder event" {
		t.Fatalf("request missing transient message: %#v", client.requests[0].Messages)
	}
	messages := agent.MessagesSnapshot()
	if len(messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(messages))
	}
	if strings.Contains(messages[0].Content, "transient") || strings.Contains(messages[1].Content, "transient") {
		t.Fatalf("transient message persisted: %#v", messages)
	}
	if messages[1].Type != llm.Assistant || messages[1].Content != "done" {
		t.Fatalf("assistant message = %#v, want done", messages[1])
	}
}

func TestAgentForwardsToolErrorHook(t *testing.T) {
	tools := llmtool.NewToolbox()
	if err := tools.RegisterTool(llmtool.NewTool("fail", "Fails.", func(context.Context, struct{}) (struct{}, error) {
		return struct{}{}, errors.New("boom")
	})); err != nil {
		t.Fatal(err)
	}

	client := &fakeChatClient{
		responses: []*llm.ChatResponse{
			{ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "fail", Arguments: `{}`}}},
			{Content: "done"},
		},
	}

	var gotCall llm.ToolCall
	var gotErr error
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("system prompt")},
			Tools:    *tools,
		},
		Hooks: Hooks{
			OnToolError: func(toolCall llm.ToolCall, err error) {
				gotCall = toolCall
				gotErr = err
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := agent.Run(); err != nil {
		t.Fatal(err)
	}
	if gotCall.Name != "fail" || gotCall.ID != "call_1" {
		t.Fatalf("tool call = %#v, want fail call_1", gotCall)
	}
	if gotErr == nil || gotErr.Error() != "boom" {
		t.Fatalf("tool error = %v, want boom", gotErr)
	}
}

func TestAgentForwardsCallErrorHook(t *testing.T) {
	client := &fakeChatClient{
		errors:    []error{errors.New("temporary")},
		responses: []*llm.ChatResponse{{Content: "done"}},
	}

	var gotErr error
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("system prompt")},
		},
		Hooks: Hooks{
			OnCallError: func(err error) {
				gotErr = err
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := agent.Run(); err != nil {
		t.Fatal(err)
	}
	if gotErr == nil || gotErr.Error() != "temporary" {
		t.Fatalf("call error = %v, want temporary", gotErr)
	}
}

func TestAgentForwardsNewHooks(t *testing.T) {
	client := &fakeChatClient{
		responses: []*llm.ChatResponse{{Content: "done", Reasoning: "thinking"}},
		onComplete: func(_ llm.ChatRequest, hooks llm.CompletionHooks) {
			hooks.EmitReasoningDelta("thinking")
		},
	}

	var reasoning []string
	var assistantEvents []llm.AssistantMessageEvent
	var eventTypes []string
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("system prompt")},
		},
		Hooks: Hooks{
			OnReasoningDelta: func(delta string) {
				reasoning = append(reasoning, delta)
			},
			OnAssistantMessage: func(event llm.AssistantMessageEvent) {
				assistantEvents = append(assistantEvents, event)
			},
			OnEvent: func(event llm.Event) error {
				switch event.(type) {
				case llm.EventReasoningDelta:
					eventTypes = append(eventTypes, "reasoning")
				case llm.EventAssistantMessage:
					eventTypes = append(eventTypes, "assistant")
				}
				return nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	output, err := agent.Run()
	if err != nil {
		t.Fatal(err)
	}
	if output.Content != "done" {
		t.Fatalf("content = %q, want done", output.Content)
	}
	if strings.Join(reasoning, "") != "thinking" {
		t.Fatalf("reasoning = %q, want thinking", strings.Join(reasoning, ""))
	}
	if len(assistantEvents) != 1 || assistantEvents[0].Reasoning != "thinking" {
		t.Fatalf("assistant events = %#v, want reasoning", assistantEvents)
	}
	if strings.Join(eventTypes, ",") != "reasoning,assistant" {
		t.Fatalf("event types = %v, want reasoning,assistant", eventTypes)
	}
}

func TestAgentOnEventCanAbort(t *testing.T) {
	client := &fakeChatClient{
		responses: []*llm.ChatResponse{{Content: "blocked"}},
	}

	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("system prompt")},
		},
		Hooks: Hooks{
			OnEvent: func(event llm.Event) error {
				if _, ok := event.(llm.EventAssistantMessage); ok {
					return errors.New("agent stopped")
				}
				return nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	output, err := agent.Run()
	if err == nil || !errors.Is(err, llm.ErrCompletionEventAborted) {
		t.Fatalf("expected event abort, got output=%#v err=%v", output, err)
	}
	if output == nil || output.Content != "blocked" {
		t.Fatalf("output = %#v, want preserved content", output)
	}
	messages := agent.MessagesSnapshot()
	if len(messages) != 2 {
		t.Fatalf("agent messages = %#v, want system and partial assistant message", messages)
	}
	if messages[1].Type != llm.Assistant || messages[1].Content != "blocked" {
		t.Fatalf("partial assistant message = %#v, want blocked transcript", messages[1])
	}
}

func TestAgentDoesNotPersistMessageGuardrailBlockedPartialOutput(t *testing.T) {
	client := guardrailStreamingClient{}
	guardrail := llm.NewMessageGuardrail("block", func(_ context.Context, input llm.MessageGuardrailInput) (llm.MessageGuardrailDecision, error) {
		if input.Phase == llm.MessageGuardrailPhaseAssistantContentDelta && strings.Contains(input.Message.Content, "blocked") {
			return llm.MessageGuardrailDecision{Action: llm.MessageGuardrailBlock}, nil
		}
		return llm.MessageGuardrailDecision{Action: llm.MessageGuardrailAllow}, nil
	})
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:            client,
			Model:             "test-model",
			Messages:          []llm.Message{llm.NewSystemMessage("system")},
			MessageGuardrails: []llm.MessageGuardrail{guardrail},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	output, err := agent.Run()
	if err == nil || !errors.Is(err, llm.ErrMessageGuardrailBlocked) {
		t.Fatalf("expected guardrail error, got output=%#v err=%v", output, err)
	}
	if output == nil || output.Content != "safe " {
		t.Fatalf("output = %#v, want safe prefix", output)
	}
	messages := agent.MessagesSnapshot()
	if len(messages) != 1 || messages[0].Type != llm.System {
		t.Fatalf("messages = %#v, want system only", messages)
	}
}

func TestAgentChainsConfigAndAgentHooks(t *testing.T) {
	client := &fakeChatClient{
		responses: []*llm.ChatResponse{{
			Content: "done",
			Usage: &llm.TokenUsage{
				PromptTokens:     1,
				CompletionTokens: 2,
				TotalTokens:      3,
			},
		}},
	}

	var calls []string
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("system prompt")},
			Hooks: llm.CompletionHooks{
				OnGenerationStart: func(llm.GenerationStartEvent) {
					calls = append(calls, "config-start")
				},
				OnUsage: func(llm.UsageEvent) {
					calls = append(calls, "config-usage")
				},
			},
		},
		Hooks: Hooks{
			OnGenerationStart: func(llm.GenerationStartEvent) {
				calls = append(calls, "agent-start")
			},
			OnUsage: func(llm.UsageEvent) {
				calls = append(calls, "agent-usage")
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	output, err := agent.Run()
	if err != nil {
		t.Fatal(err)
	}
	if output.Usage.TotalTokens != 3 {
		t.Fatalf("usage = %#v, want total 3", output.Usage)
	}
	if strings.Join(calls, ",") != "config-start,agent-start,config-usage,agent-usage" {
		t.Fatalf("calls = %v, want config then agent hooks", calls)
	}
}

func TestAsToolUsesAgentMetadataAndQueryInput(t *testing.T) {
	agent, err := NewAgent(NewAgentInput{
		Name:        "Smart Home",
		Description: "Controls smart-home devices.",
		Config: llm.CompletionCallInput{
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	tool := agent.AsTool()
	if tool.Name != "agent_smart_home" {
		t.Fatalf("tool name = %q, want agent_smart_home", tool.Name)
	}
	if tool.Description != "Controls smart-home devices." {
		t.Fatalf("tool description = %q", tool.Description)
	}

	properties := tool.Parameters["properties"].(map[string]any)
	query := properties["query"].(map[string]any)
	if query["type"] != "string" {
		t.Fatalf("query.type = %v, want string", query["type"])
	}

	required := tool.Parameters["required"].([]string)
	if len(required) != 1 || required[0] != "query" {
		t.Fatalf("required = %v, want [query]", required)
	}
}

func TestAsToolRejectsEmptyQuery(t *testing.T) {
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	tool := agent.AsTool()
	result, err := tool.Call(context.Background(), []byte(`{"query":" "}`))
	if err == nil || !strings.Contains(err.Error(), "user message") {
		t.Fatalf("expected user message error, got result=%v err=%v", result, err)
	}
}

func TestAsToolPassesCallContextToAgentCompletion(t *testing.T) {
	type contextKey string
	key := contextKey("request-id")
	ctx := context.WithValue(context.Background(), key, "req-123")
	client := &fakeChatClient{
		responses: []*llm.ChatResponse{{Content: "done"}},
	}

	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	tool := agent.AsTool()
	result, err := tool.Call(ctx, []byte(`{"query":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || !strings.Contains(*result, `"response":"done"`) {
		t.Fatalf("result = %v, want delegated response", result)
	}
	if len(client.contexts) != 1 || client.contexts[0].Value(key) != "req-123" {
		t.Fatalf("contexts = %#v, want delegated caller context", client.contexts)
	}
}

func TestAgentConcurrentPublicMethods(t *testing.T) {
	client := &fakeChatClient{}
	agent, err := NewAgent(NewAgentInput{
		Name: "test",
		Config: llm.CompletionCallInput{
			Client:   client,
			Model:    "test-model",
			Messages: []llm.Message{llm.NewSystemMessage("test")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			agent.AddMessage(llm.NewUserMessage("manual"))
			_ = agent.MessagesSnapshot()
		}()
	}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := agent.CallWithUserMessage("hello"); err != nil {
				t.Error(err)
			}
		}()
	}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := agent.RunWithTransientMessage(llm.NewUserMessage("transient")); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()

	if len(agent.MessagesSnapshot()) == 0 {
		t.Fatal("messages snapshot is empty")
	}
}

type fakeChatClient struct {
	mu         sync.Mutex
	responses  []*llm.ChatResponse
	errors     []error
	requests   []llm.ChatRequest
	contexts   []context.Context
	onComplete func(llm.ChatRequest, llm.CompletionHooks)
}

type guardrailStreamingClient struct{}

func (guardrailStreamingClient) Complete(_ context.Context, _ llm.ChatRequest, hooks llm.CompletionHooks) (*llm.ChatResponse, error) {
	if err := hooks.EmitContentDeltaEvent("safe "); err != nil {
		return nil, err
	}
	if err := hooks.EmitContentDeltaEvent("blocked"); err != nil {
		return nil, err
	}
	return &llm.ChatResponse{Content: "safe blocked"}, nil
}

func (guardrailStreamingClient) ListModels(context.Context) ([]string, error) {
	return nil, nil
}

func (c *fakeChatClient) Complete(ctx context.Context, request llm.ChatRequest, hooks llm.CompletionHooks) (*llm.ChatResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
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
		return &llm.ChatResponse{}, nil
	}
	response := c.responses[0]
	c.responses = c.responses[1:]
	return response, nil
}

func (c *fakeChatClient) ListModels(context.Context) ([]string, error) {
	return nil, nil
}

package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
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

type fakeChatClient struct {
	responses []*llm.ChatResponse
	requests  []llm.ChatRequest
}

func (c *fakeChatClient) Complete(_ context.Context, request llm.ChatRequest, _ llm.CompletionHooks) (*llm.ChatResponse, error) {
	c.requests = append(c.requests, request)
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

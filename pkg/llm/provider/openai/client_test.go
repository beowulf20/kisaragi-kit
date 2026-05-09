package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

func TestOpenAIMessageSupportsToolResultMessages(t *testing.T) {
	message, err := openAIMessage(llm.NewToolMessage("call_123", `{"ok":true}`))
	if err != nil {
		t.Fatal(err)
	}

	data, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	result := string(data)
	for _, want := range []string{`"role":"tool"`, `"tool_call_id":"call_123"`, `\"ok\":true`} {
		if !strings.Contains(result, want) {
			t.Fatalf("message missing %q: %s", want, result)
		}
	}
}

func TestOpenAIMessageSupportsAssistantToolCalls(t *testing.T) {
	message, err := openAIMessage(llm.NewAssistantToolCallMessage("", []llm.ToolCall{
		{ID: "call_123", Name: "control_device", Arguments: `{"device_id":2}`},
	}))
	if err != nil {
		t.Fatal(err)
	}

	data, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	result := string(data)
	for _, want := range []string{`"role":"assistant"`, `"tool_calls"`, `"id":"call_123"`, `"name":"control_device"`, `\"device_id\":2`} {
		if !strings.Contains(result, want) {
			t.Fatalf("message missing %q: %s", want, result)
		}
	}
}

func TestClientCompleteStreamsContentDeltas(t *testing.T) {
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		requestBody = string(data)

		w.Header().Set("Content-Type", "text/event-stream")
		for _, data := range []string{
			`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":0,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":0,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":0,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client, _, err := NewClient(ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}

	var deltas []string
	output, err := client.Complete(t.Context(), llm.ChatRequest{
		Model: "test-model",
		Messages: []llm.Message{
			llm.NewUserMessage("say hello"),
		},
	}, llm.CompletionHooks{
		OnContentDelta: func(delta string) {
			deltas = append(deltas, delta)
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
	if !strings.Contains(requestBody, `"messages"`) {
		t.Fatalf("request missing messages:\n%s", requestBody)
	}
}

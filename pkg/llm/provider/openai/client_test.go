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
			`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":0,"model":"test-model","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":4,"audio_tokens":2},"completion_tokens_details":{"reasoning_tokens":7,"audio_tokens":3,"accepted_prediction_tokens":5,"rejected_prediction_tokens":6}}}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client, _, err := NewClient(ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
		ChatCompletionExtraFields: map[string]any{
			"provider_option": map[string]string{"mode": "custom"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var deltas []string
	output, err := client.Complete(t.Context(), llm.ChatRequest{
		Model:           "test-model",
		ReasoningEffort: llm.ReasoningEffortLow,
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
	if output.Usage == nil {
		t.Fatal("usage is nil")
	}
	if output.Usage.PromptTokens != 10 || output.Usage.CompletionTokens != 20 || output.Usage.TotalTokens != 30 {
		t.Fatalf("usage = %#v, want 10/20/30", output.Usage)
	}
	if output.Usage.PromptTokenDetails["cached_tokens"] != 4 || output.Usage.PromptTokenDetails["audio_tokens"] != 2 {
		t.Fatalf("prompt usage details = %#v", output.Usage.PromptTokenDetails)
	}
	if output.Usage.CompletionTokenDetails["reasoning_tokens"] != 7 ||
		output.Usage.CompletionTokenDetails["audio_tokens"] != 3 ||
		output.Usage.CompletionTokenDetails["accepted_prediction_tokens"] != 5 ||
		output.Usage.CompletionTokenDetails["rejected_prediction_tokens"] != 6 {
		t.Fatalf("completion usage details = %#v", output.Usage.CompletionTokenDetails)
	}
	if !strings.Contains(requestBody, `"messages"`) {
		t.Fatalf("request missing messages:\n%s", requestBody)
	}
	if !strings.Contains(requestBody, `"stream_options"`) || !strings.Contains(requestBody, `"include_usage":true`) {
		t.Fatalf("request missing include_usage stream option:\n%s", requestBody)
	}
	if !strings.Contains(requestBody, `"reasoning_effort":"low"`) {
		t.Fatalf("request missing reasoning effort:\n%s", requestBody)
	}
	if !strings.Contains(requestBody, `"provider_option":{"mode":"custom"}`) {
		t.Fatalf("request missing extra provider field:\n%s", requestBody)
	}
}

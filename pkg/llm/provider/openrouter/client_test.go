package openrouter

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

func TestNewClientNormalizesProviderPreferences(t *testing.T) {
	allowFallbacks := false
	topLevel := map[string]any{"models": []string{"fallback-model"}}
	providerExtra := map[string]any{"zdr": true}
	client, normalized, err := NewClient(ClientConfig{
		APIKey: "test-key",
		Provider: ProviderPreferences{
			Only:           []string{" deepinfra ", "alibaba", "deepinfra"},
			Order:          []string{"alibaba", "deepinfra"},
			AllowFallbacks: &allowFallbacks,
			ExtraFields:    providerExtra,
		},
		ChatCompletionExtraFields: topLevel,
	})
	if err != nil {
		t.Fatal(err)
	}
	if client == nil {
		t.Fatal("client is nil")
	}
	if normalized.BaseURL != DefaultBaseURL {
		t.Fatalf("BaseURL = %q, want %q", normalized.BaseURL, DefaultBaseURL)
	}
	if !reflect.DeepEqual(normalized.Provider.Only, []string{"deepinfra", "alibaba"}) {
		t.Fatalf("Only = %#v", normalized.Provider.Only)
	}
	if !reflect.DeepEqual(normalized.Provider.Order, []string{"alibaba", "deepinfra"}) {
		t.Fatalf("Order = %#v", normalized.Provider.Order)
	}
	if normalized.Provider.AllowFallbacks == nil || *normalized.Provider.AllowFallbacks {
		t.Fatalf("AllowFallbacks = %v, want explicit false", normalized.Provider.AllowFallbacks)
	}

	allowFallbacks = true
	topLevel["models"] = []string{"mutated"}
	providerExtra["zdr"] = false
	if *normalized.Provider.AllowFallbacks {
		t.Fatal("normalized AllowFallbacks aliases input")
	}
	if normalized.Provider.ExtraFields["zdr"] != true {
		t.Fatalf("normalized provider extras = %#v", normalized.Provider.ExtraFields)
	}
	if !reflect.DeepEqual(normalized.ChatCompletionExtraFields["models"], []string{"fallback-model"}) {
		t.Fatalf("normalized top-level extras = %#v", normalized.ChatCompletionExtraFields)
	}
}

func TestNewClientRejectsInvalidProviderPreferences(t *testing.T) {
	tests := []struct {
		name   string
		config ClientConfig
		want   string
	}{
		{name: "missing API key", config: ClientConfig{}, want: "API key cannot be empty"},
		{name: "blank only slug", config: testConfig(ProviderPreferences{Only: []string{" "}}), want: "provider only contains an empty slug"},
		{name: "only ignore overlap", config: testConfig(ProviderPreferences{Only: []string{"deepinfra"}, Ignore: []string{"deepinfra"}}), want: `provider "deepinfra" cannot appear in both only and ignore`},
		{name: "ignored order", config: testConfig(ProviderPreferences{Ignore: []string{"deepinfra"}, Order: []string{"deepinfra"}}), want: `ordered provider "deepinfra" cannot be ignored`},
		{name: "order outside only", config: testConfig(ProviderPreferences{Only: []string{"deepinfra"}, Order: []string{"alibaba"}}), want: `ordered provider "alibaba" must appear in only`},
		{name: "empty provider extra key", config: testConfig(ProviderPreferences{ExtraFields: map[string]any{" ": true}}), want: "provider extra field name cannot be empty"},
		{name: "typed provider collision", config: testConfig(ProviderPreferences{ExtraFields: map[string]any{"only": []string{"deepinfra"}}}), want: `provider extra field "only" is reserved by typed preferences`},
		{name: "top-level provider collision", config: ClientConfig{APIKey: "test-key", ChatCompletionExtraFields: map[string]any{"provider": map[string]any{}}}, want: "chat completion extra field provider is reserved; use Provider"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := NewClient(tt.config)
			if err == nil || err.Error() != tt.want {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestClientCompleteSendsProviderPreferences(t *testing.T) {
	requestBody := runCompletionRequest(t, ProviderPreferences{
		Only:           []string{"deepinfra", "alibaba"},
		Ignore:         []string{"deepseek"},
		Order:          []string{"alibaba", "deepinfra"},
		AllowFallbacks: boolPointer(false),
		ExtraFields:    map[string]any{"zdr": true},
	}, map[string]any{"models": []string{"fallback-model"}})

	var body map[string]any
	if err := json.Unmarshal([]byte(requestBody), &body); err != nil {
		t.Fatal(err)
	}
	provider, ok := body["provider"].(map[string]any)
	if !ok {
		t.Fatalf("provider = %#v", body["provider"])
	}
	if !reflect.DeepEqual(provider["only"], []any{"deepinfra", "alibaba"}) ||
		!reflect.DeepEqual(provider["ignore"], []any{"deepseek"}) ||
		!reflect.DeepEqual(provider["order"], []any{"alibaba", "deepinfra"}) ||
		provider["allow_fallbacks"] != false || provider["zdr"] != true {
		t.Fatalf("provider = %#v", provider)
	}
	if !reflect.DeepEqual(body["models"], []any{"fallback-model"}) {
		t.Fatalf("models = %#v", body["models"])
	}
}

func TestClientCompleteSendsTypedProviderWithoutRawExtras(t *testing.T) {
	requestBody := runCompletionRequest(t, ProviderPreferences{
		Only:           []string{"deepinfra"},
		AllowFallbacks: boolPointer(false),
	}, nil)
	if !strings.Contains(requestBody, `"provider":{"allow_fallbacks":false,"only":["deepinfra"]}`) {
		t.Fatalf("request missing typed provider preferences: %s", requestBody)
	}
}

func TestClientCompleteOmitsProviderWhenUnset(t *testing.T) {
	requestBody := runCompletionRequest(t, ProviderPreferences{}, nil)
	if strings.Contains(requestBody, `"provider"`) {
		t.Fatalf("request unexpectedly contains provider: %s", requestBody)
	}
}

func runCompletionRequest(t *testing.T, provider ProviderPreferences, extraFields map[string]any) string {
	t.Helper()
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		requestBody = string(data)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
		fmt.Fprint(w, "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	client, _, err := NewClient(ClientConfig{
		BaseURL:                   server.URL,
		APIKey:                    "test-key",
		Provider:                  provider,
		ChatCompletionExtraFields: extraFields,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Complete(t.Context(), llm.ChatRequest{
		Model:    "test-model",
		Messages: []llm.Message{llm.NewUserMessage("hello")},
	}, llm.CompletionHooks{}); err != nil {
		t.Fatal(err)
	}
	return requestBody
}

func testConfig(provider ProviderPreferences) ClientConfig {
	return ClientConfig{APIKey: "test-key", Provider: provider}
}

func boolPointer(value bool) *bool {
	return &value
}

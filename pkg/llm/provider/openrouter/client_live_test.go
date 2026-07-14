package openrouter

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

var requiredLiveProviders = []string{"alibaba", "deepseek"}

func TestClientCompleteRestrictedProviders(t *testing.T) {
	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	baseURL := strings.TrimSpace(os.Getenv("OPENROUTER_BASE_URL"))
	model := strings.TrimSpace(os.Getenv("OPENROUTER_MODEL"))
	providersValue := strings.TrimSpace(os.Getenv("OPENROUTER_PROVIDERS"))
	if apiKey == "" || baseURL == "" || model == "" || providersValue == "" {
		t.Skip("OpenRouter provider-routing live-test environment is not configured")
	}

	configured := make(map[string]struct{})
	for _, value := range strings.Split(providersValue, ",") {
		if provider := strings.TrimSpace(value); provider != "" {
			configured[provider] = struct{}{}
		}
	}
	for _, provider := range requiredLiveProviders {
		if _, exists := configured[provider]; !exists {
			t.Fatalf("OPENROUTER_PROVIDERS must include %q", provider)
		}
	}

	for _, provider := range requiredLiveProviders {
		t.Run(provider, func(t *testing.T) {
			client, _, err := NewClient(ClientConfig{
				BaseURL: baseURL,
				APIKey:  apiKey,
				Timeout: 90 * time.Second,
				Provider: ProviderPreferences{
					Only:           []string{provider},
					AllowFallbacks: boolPointer(false),
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Complete(t.Context(), llm.ChatRequest{
				Model:    model,
				Messages: []llm.Message{llm.NewUserMessage("Reply with OK only.")},
			}, llm.CompletionHooks{})
			if err != nil {
				t.Fatal(err)
			}
			if response.Usage == nil {
				t.Fatal("OpenRouter response omitted usage")
			}
		})
	}
}

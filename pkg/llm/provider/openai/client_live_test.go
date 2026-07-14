package openai

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

func TestClientCompleteOpenRouterCost(t *testing.T) {
	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	baseURL := strings.TrimSpace(os.Getenv("OPENROUTER_BASE_URL"))
	model := strings.TrimSpace(os.Getenv("OPENROUTER_MODEL"))
	if apiKey == "" || baseURL == "" || model == "" {
		t.Skip("OpenRouter live-test environment is not configured")
	}

	client, _, err := NewClient(ClientConfig{
		BaseURL: baseURL,
		APIKey:  apiKey,
		Timeout: 60 * time.Second,
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
	if response.Usage.CostUSD == nil {
		t.Fatalf("OpenRouter usage omitted cost (prompt=%d completion=%d total=%d)", response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
	}
}

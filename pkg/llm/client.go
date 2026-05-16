package llm

import (
	"context"
	"errors"
	"fmt"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

// ChatClient is the provider-neutral interface used by the completion loop.
type ChatClient interface {
	// Complete sends one chat request and returns the final assistant response.
	Complete(context.Context, ChatRequest, CompletionHooks) (*ChatResponse, error)
	// ListModels returns provider model IDs, with the preferred default first.
	ListModels(context.Context) ([]string, error)
}

// ChatRequest contains one provider-neutral chat completion request.
type ChatRequest struct {
	// Model is the provider model ID to use.
	Model string
	// Messages is the ordered conversation sent to the model.
	Messages []Message
	// Temperature controls response randomness.
	Temperature float64
	// Tools lists function tools the model may call.
	Tools []llmtool.ChatTool
}

// ChatResponse contains the assistant response returned by a chat client.
type ChatResponse struct {
	// Content is the assistant text content.
	Content string
	// ToolCalls contains function tool calls requested by the assistant.
	ToolCalls []ToolCall
	// Usage contains token usage reported by the provider when available.
	Usage *TokenUsage
}

// ResolveModel returns model when set, or asks the client for its first model.
func ResolveModel(ctx context.Context, client ChatClient, model string) (string, error) {
	if model != "" {
		return model, nil
	}
	if client == nil {
		return "", errors.New("model auto-detection failed; client cannot be nil")
	}

	models, err := client.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("model auto-detection failed; pass --model explicitly: %w", err)
	}
	if len(models) == 0 {
		return "", errors.New("model auto-detection returned no models; pass --model explicitly")
	}
	return models[0], nil
}

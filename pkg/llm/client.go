package llm

import (
	"context"
	"errors"
	"fmt"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type ChatClient interface {
	Complete(context.Context, ChatRequest, CompletionHooks) (*ChatResponse, error)
	ListModels(context.Context) ([]string, error)
}

type ChatRequest struct {
	Model       string
	Messages    []Message
	Temperature float64
	Tools       []llmtool.ChatTool
}

type ChatResponse struct {
	Content   string
	ToolCalls []ToolCall
}

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

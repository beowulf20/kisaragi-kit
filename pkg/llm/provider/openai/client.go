package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"github.com/tidwall/gjson"
)

// ClientConfig configures an OpenAI-compatible chat client.
type ClientConfig struct {
	// BaseURL is the OpenAI-compatible API base URL.
	BaseURL string
	// APIKey is the credential sent to the API.
	APIKey string
	// Timeout limits individual API requests when greater than zero.
	Timeout time.Duration
	// ChatCompletionExtraFields adds provider-specific JSON fields to every chat completion request.
	ChatCompletionExtraFields map[string]any
}

// Client adapts OpenAI-compatible chat completion APIs to llm.ChatClient.
type Client struct {
	config ClientConfig
	client openaisdk.Client
}

// NewClient validates config and returns a configured OpenAI-compatible client.
func NewClient(config ClientConfig) (*Client, ClientConfig, error) {
	config.BaseURL = strings.TrimRight(config.BaseURL, "/")
	if config.BaseURL == "" {
		return nil, config, errors.New("base URL cannot be empty")
	}
	if config.APIKey == "" {
		return nil, config, errors.New("API key cannot be empty")
	}
	config.ChatCompletionExtraFields = cloneExtraFields(config.ChatCompletionExtraFields)

	options := []option.RequestOption{
		option.WithBaseURL(config.BaseURL),
		option.WithAPIKey(config.APIKey),
	}
	if config.Timeout > 0 {
		options = append(options, option.WithRequestTimeout(config.Timeout))
	}

	return &Client{
		config: config,
		client: openaisdk.NewClient(options...),
	}, config, nil
}

// Complete sends a streaming chat completion request and returns the final result.
func (c *Client) Complete(ctx context.Context, request llm.ChatRequest, hooks llm.CompletionHooks) (*llm.ChatResponse, error) {
	if c == nil {
		return nil, errors.New("openai client is nil")
	}

	messages := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(request.Messages))
	for _, message := range request.Messages {
		openAIMessage, err := openAIMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, openAIMessage)
	}

	params := openaisdk.ChatCompletionNewParams{
		Model:       openaisdk.ChatModel(request.Model),
		Messages:    messages,
		Temperature: openaisdk.Float(request.Temperature),
		StreamOptions: openaisdk.ChatCompletionStreamOptionsParam{
			IncludeUsage: openaisdk.Bool(true),
		},
	}
	if len(c.config.ChatCompletionExtraFields) > 0 {
		params.SetExtraFields(cloneExtraFields(c.config.ChatCompletionExtraFields))
	}
	if request.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(request.ReasoningEffort)
	}
	tools := openAIChatTools(request.Tools)
	if len(tools) > 0 {
		params.Tools = tools
		params.ToolChoice = openaisdk.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openaisdk.Opt("auto")}
	}

	completion, reasoning, err := c.streamingChatCompletion(ctx, params, hooks)
	if err != nil {
		return nil, err
	}
	if len(completion.Choices) == 0 {
		return nil, errors.New("chat completion returned no choices")
	}

	message := completion.Choices[0].Message
	response := &llm.ChatResponse{
		Content:   message.Content,
		Reasoning: reasoning,
		Usage:     completionUsage(completion.Usage),
	}
	if response.Reasoning == "" {
		response.Reasoning = reasoningTextFromCompletionRaw(completion.RawJSON())
	}
	for _, toolCall := range message.ToolCalls {
		if toolCall.Type != "function" {
			return nil, fmt.Errorf("unsupported tool call type %q", toolCall.Type)
		}
		response.ToolCalls = append(response.ToolCalls, llm.ToolCall{
			ID:        toolCall.ID,
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		})
	}
	return response, nil
}

func cloneExtraFields(fields map[string]any) map[string]any {
	if len(fields) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(fields))
	for key, value := range fields {
		cloned[key] = value
	}
	return cloned
}

func completionUsage(usage openaisdk.CompletionUsage) *llm.TokenUsage {
	if usage.PromptTokens == 0 && usage.CompletionTokens == 0 && usage.TotalTokens == 0 {
		return nil
	}

	result := &llm.TokenUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}
	result.PromptTokenDetails = tokenDetails(map[string]int64{
		"audio_tokens":  usage.PromptTokensDetails.AudioTokens,
		"cached_tokens": usage.PromptTokensDetails.CachedTokens,
	})
	result.CompletionTokenDetails = tokenDetails(map[string]int64{
		"accepted_prediction_tokens": usage.CompletionTokensDetails.AcceptedPredictionTokens,
		"audio_tokens":               usage.CompletionTokensDetails.AudioTokens,
		"reasoning_tokens":           usage.CompletionTokensDetails.ReasoningTokens,
		"rejected_prediction_tokens": usage.CompletionTokensDetails.RejectedPredictionTokens,
	})
	return result
}

func tokenDetails(values map[string]int64) map[string]int64 {
	details := make(map[string]int64)
	for key, value := range values {
		if value != 0 {
			details[key] = value
		}
	}
	if len(details) == 0 {
		return nil
	}
	return details
}

// ListModels returns available model IDs from the OpenAI-compatible API.
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	if c == nil {
		return nil, errors.New("openai client is nil")
	}

	models, err := c.client.Models.List(ctx)
	if err != nil {
		return nil, err
	}

	ids := make([]string, 0, len(models.Data))
	for _, model := range models.Data {
		ids = append(ids, model.ID)
	}
	return ids, nil
}

func (c *Client) streamingChatCompletion(ctx context.Context, params openaisdk.ChatCompletionNewParams, hooks llm.CompletionHooks) (*openaisdk.ChatCompletion, string, error) {
	stream := c.client.Chat.Completions.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()
	acc := openaisdk.ChatCompletionAccumulator{}
	var reasoning strings.Builder

	for stream.Next() {
		chunk := stream.Current()
		if !acc.AddChunk(chunk) {
			return nil, "", errors.New("chat completion stream accumulation failed")
		}

		for _, delta := range reasoningDeltasFromChunkRaw(chunk.RawJSON()) {
			reasoning.WriteString(delta)
			if err := hooks.EmitReasoningDelta(delta); err != nil {
				return nil, "", err
			}
		}
		if reasoning.Len() == 0 {
			reasoning.WriteString(reasoningTextFromCompletionRaw(chunk.RawJSON()))
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if err := hooks.EmitContentDeltaEvent(choice.Delta.Content); err != nil {
					return nil, "", err
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, "", err
	}

	return &acc.ChatCompletion, reasoning.String(), nil
}

func reasoningDeltasFromChunkRaw(raw string) []string {
	if raw == "" {
		return nil
	}
	choices := gjson.Get(raw, "choices").Array()
	deltas := make([]string, 0, len(choices))
	for _, choice := range choices {
		for _, path := range []string{"delta.reasoning", "delta.reasoning_content"} {
			value := choice.Get(path)
			if value.Exists() && value.Type == gjson.String && value.String() != "" {
				deltas = append(deltas, value.String())
				break
			}
		}
	}
	return deltas
}

func reasoningTextFromCompletionRaw(raw string) string {
	if raw == "" {
		return ""
	}
	var reasoning strings.Builder
	for _, choice := range gjson.Get(raw, "choices").Array() {
		for _, path := range []string{"message.reasoning", "message.reasoning_content"} {
			value := choice.Get(path)
			if value.Exists() && value.Type == gjson.String && value.String() != "" {
				reasoning.WriteString(value.String())
				break
			}
		}
	}
	return reasoning.String()
}

func openAIMessage(message llm.Message) (openaisdk.ChatCompletionMessageParamUnion, error) {
	switch message.Type {
	case llm.System:
		return openaisdk.SystemMessage(message.Content), nil
	case llm.User:
		return openaisdk.UserMessage(message.Content), nil
	case llm.Assistant:
		if len(message.ToolCalls) > 0 {
			return assistantToolCallMessage(message), nil
		}
		return openaisdk.AssistantMessage(message.Content), nil
	case llm.Tool:
		if strings.TrimSpace(message.ToolCallID) == "" {
			return openaisdk.ChatCompletionMessageParamUnion{}, errors.New("tool message missing tool call ID")
		}
		return openaisdk.ToolMessage(message.Content, message.ToolCallID), nil
	default:
		return openaisdk.ChatCompletionMessageParamUnion{}, fmt.Errorf("unsupported message type %q", message.Type)
	}
}

func assistantToolCallMessage(message llm.Message) openaisdk.ChatCompletionMessageParamUnion {
	toolCalls := make([]openaisdk.ChatCompletionMessageToolCallUnionParam, 0, len(message.ToolCalls))
	for _, toolCall := range message.ToolCalls {
		toolCalls = append(toolCalls, openaisdk.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
				ID: toolCall.ID,
				Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      toolCall.Name,
					Arguments: toolCall.Arguments,
				},
			},
		})
	}

	return openaisdk.ChatCompletionMessageParamUnion{
		OfAssistant: &openaisdk.ChatCompletionAssistantMessageParam{
			Content:   openaisdk.ChatCompletionAssistantMessageParamContentUnion{OfString: openaisdk.String(message.Content)},
			ToolCalls: toolCalls,
		},
	}
}

func openAIChatTools(tools []llmtool.ChatTool) []openaisdk.ChatCompletionToolUnionParam {
	openAITools := make([]openaisdk.ChatCompletionToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		openAITools = append(openAITools, openaisdk.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        tool.Name,
			Description: openaisdk.String(tool.Description),
			Parameters:  shared.FunctionParameters(tool.Parameters),
		}))
	}
	return openAITools
}

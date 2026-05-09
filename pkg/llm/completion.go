package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type CompletionCallInput struct {
	Client      ChatClient
	Model       string
	Messages    []Message
	Temperature float64
	Tools       llmtool.Toolbox
	Hooks       CompletionHooks
}

func (input CompletionCallInput) Validate() error {
	if strings.TrimSpace(input.Model) == "" {
		return errors.New("model cannot be empty")
	}
	if len(input.Messages) == 0 {
		return errors.New("at least one message is required")
	}
	return nil
}

type CompletionCallOutput struct {
	Content   string
	ToolCalls []ToolCall
	Messages  []Message
}

const maxToolCallRounds = 8

func Completion(input CompletionCallInput) (*CompletionCallOutput, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Client == nil {
		return nil, errors.New("client cannot be nil")
	}

	ctx := context.Background()
	messages := append([]Message(nil), input.Messages...)
	output := &CompletionCallOutput{}
	tools := input.Tools.ChatTools()

	for round := 0; round <= maxToolCallRounds; round++ {
		response, err := input.Client.Complete(ctx, ChatRequest{
			Model:       input.Model,
			Messages:    messages,
			Temperature: input.Temperature,
			Tools:       tools,
		}, input.Hooks)
		if err != nil {
			return nil, fmt.Errorf("chat completion failed: %w", err)
		}
		if response == nil {
			return nil, errors.New("chat completion returned nil response")
		}

		if len(response.ToolCalls) == 0 {
			output.Content = response.Content
			if strings.TrimSpace(response.Content) != "" {
				output.Messages = append(output.Messages, NewAssistantMessage(response.Content))
			}
			return output, nil
		}

		if round == maxToolCallRounds {
			return nil, fmt.Errorf("exceeded %d tool call rounds", maxToolCallRounds)
		}

		assistantToolCalls := append([]ToolCall(nil), response.ToolCalls...)
		output.Messages = append(output.Messages, NewAssistantToolCallMessage(response.Content, assistantToolCalls))
		messages = append(messages, NewAssistantToolCallMessage(response.Content, assistantToolCalls))

		for _, toolCall := range response.ToolCalls {
			call := ToolCall{
				ID:        toolCall.ID,
				Name:      toolCall.Name,
				Arguments: toolCall.Arguments,
			}
			input.Hooks.EmitToolCall(call)

			result, err := input.Tools.Call(ctx, toolCall.Name, toolCall.Arguments)
			if err != nil {
				errorResult := shortToolError(err)
				call.Result = errorResult
				input.Hooks.EmitToolResult(call)
				output.ToolCalls = append(output.ToolCalls, call)
				output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, errorResult))
				messages = append(messages, NewToolMessage(toolCall.ID, errorResult))
				continue
			}
			if result == nil {
				errorResult := shortToolError(fmt.Errorf("tool %q returned nil result", toolCall.Name))
				call.Result = errorResult
				input.Hooks.EmitToolResult(call)
				output.ToolCalls = append(output.ToolCalls, call)
				output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, errorResult))
				messages = append(messages, NewToolMessage(toolCall.ID, errorResult))
				continue
			}

			call.Result = *result
			input.Hooks.EmitToolResult(call)
			output.ToolCalls = append(output.ToolCalls, call)
			output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, *result))
			messages = append(messages, NewToolMessage(toolCall.ID, *result))
		}
	}

	return nil, fmt.Errorf("exceeded %d tool call rounds", maxToolCallRounds)
}

func shortToolError(err error) string {
	message := strings.TrimSpace(err.Error())
	if len(message) > 240 {
		message = message[:240]
	}
	data, marshalErr := json.Marshal(map[string]string{"error": message})
	if marshalErr != nil {
		return `{"error":"tool failed"}`
	}
	return string(data)
}

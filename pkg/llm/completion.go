package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

// CompletionCallInput configures a complete LLM call, including tools and hooks.
type CompletionCallInput struct {
	// Context controls provider requests and tool execution. Nil uses context.Background().
	Context context.Context
	// Client sends provider-specific chat completion requests.
	Client ChatClient
	// Model is the provider model ID to use.
	Model string
	// Messages is the conversation state sent to the model.
	Messages []Message
	// Temperature controls response randomness.
	Temperature float64
	// Tools contains typed tools available to the model.
	Tools llmtool.Toolbox
	// Hooks receives streaming content and tool execution events.
	Hooks CompletionHooks
	// ToolErrorInterceptor can rewrite tool error feedback or abort on tool failure.
	ToolErrorInterceptor ToolErrorInterceptor
	// ApprovalDecisionMessages controls whether human approval decisions are added to output messages.
	ApprovalDecisionMessages ApprovalDecisionMessages
	// MaxToolCallRounds caps consecutive tool-call turns. Zero uses DefaultMaxToolCallRounds.
	MaxToolCallRounds int
	// MaxToolErrorLength caps tool error text sent back to the model. Zero uses DefaultMaxToolErrorLength.
	MaxToolErrorLength int
	// ProviderErrorRetries caps retries after provider completion errors. Nil uses DefaultProviderErrorRetries.
	ProviderErrorRetries *int
}

// ToolErrorInterceptor handles tool execution errors before feedback is sent to the model.
type ToolErrorInterceptor func(ToolErrorContext) ToolErrorDecision

// ToolErrorContext describes one failed tool execution.
type ToolErrorContext struct {
	// ToolCall is the failed tool call.
	ToolCall ToolCall
	// Err is the tool execution error.
	Err error
	// Round is the current tool-call round.
	Round int
	// DefaultFeedback is the JSON error feedback used when the interceptor returns no feedback.
	DefaultFeedback string
}

// ToolErrorDecision controls how a failed tool call is handled.
type ToolErrorDecision struct {
	// Feedback is sent to the model as the tool result. Empty uses ToolErrorContext.DefaultFeedback.
	Feedback string
	// Abort stops the completion immediately and returns the tool error.
	Abort bool
}

// ApprovalDecisionMessages controls which approval decisions are persisted in transcripts.
type ApprovalDecisionMessages struct {
	// AppendAccepted adds accepted approval decisions to CompletionCallOutput.Messages.
	AppendAccepted bool
	// AppendRejected adds rejected approval decisions to CompletionCallOutput.Messages.
	AppendRejected bool
}

// Validate checks that required completion input fields are present.
func (input CompletionCallInput) Validate() error {
	if strings.TrimSpace(input.Model) == "" {
		return errors.New("model cannot be empty")
	}
	if len(input.Messages) == 0 {
		return errors.New("at least one message is required")
	}
	if input.MaxToolCallRounds < 0 {
		return errors.New("max tool call rounds cannot be negative")
	}
	if input.MaxToolErrorLength < 0 {
		return errors.New("max tool error length cannot be negative")
	}
	if input.ProviderErrorRetries != nil && *input.ProviderErrorRetries < 0 {
		return errors.New("provider error retries cannot be negative")
	}
	return nil
}

// CompletionCallOutput contains the final content and generated transcript.
type CompletionCallOutput struct {
	// Content is the final assistant text content.
	Content string
	// ToolCalls records tool executions made during the call.
	ToolCalls []ToolCall
	// Messages contains assistant and tool messages produced by the call.
	Messages []Message
	// Usage aggregates token usage reported by provider generations.
	Usage TokenUsage
	// UsageEvents records token usage reported by each provider generation.
	UsageEvents []UsageEvent
}

const (
	// DefaultMaxToolCallRounds is the default cap for consecutive tool-call turns.
	DefaultMaxToolCallRounds = 8
	// DefaultMaxToolErrorLength is the default cap for tool error text sent back to the model.
	DefaultMaxToolErrorLength = 240
	// DefaultProviderErrorRetries is the default number of retries after provider completion errors.
	DefaultProviderErrorRetries = 2
)

// Completion runs a chat completion, executing requested tools until final text.
func Completion(input CompletionCallInput) (*CompletionCallOutput, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Client == nil {
		return nil, errors.New("client cannot be nil")
	}

	ctx := input.Context
	if ctx == nil {
		ctx = context.Background()
	}
	messages := append([]Message(nil), input.Messages...)
	output := &CompletionCallOutput{}
	tools := input.Tools.ChatTools()
	maxToolCallRounds := input.effectiveMaxToolCallRounds()
	maxToolErrorLength := input.effectiveMaxToolErrorLength()
	providerErrorRetries := input.effectiveProviderErrorRetries()

	for round := 0; round <= maxToolCallRounds; round++ {
		request := ChatRequest{
			Model:       input.Model,
			Messages:    messages,
			Temperature: input.Temperature,
			Tools:       tools,
		}
		response, attempt, err := completeWithProviderRetries(ctx, input.Client, request, input.Hooks, providerErrorRetries, round)
		if err != nil {
			return nil, fmt.Errorf("chat completion failed: %w", err)
		}
		if response == nil {
			return nil, errors.New("chat completion returned nil response")
		}
		output.recordUsage(input.Hooks, request.Model, round, attempt, response.Usage)

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

			callOutput, err := input.Tools.CallWithInfo(ctx, toolCall.Name, toolCall.Arguments)
			if err != nil {
				input.Hooks.EmitToolError(call, err)
				feedback, abort := input.interceptToolError(call, err, round, maxToolErrorLength)
				if abort {
					return nil, fmt.Errorf("tool %q failed: %w", toolCall.Name, err)
				}
				call.Result = feedback
				input.Hooks.EmitToolResult(call)
				output.ToolCalls = append(output.ToolCalls, call)
				output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, feedback))
				messages = append(messages, NewToolMessage(toolCall.ID, feedback))
				input.appendApprovalDecisionMessage(output, callOutput.Approval)
				continue
			}
			if callOutput.Result == nil {
				err := fmt.Errorf("tool %q returned nil result", toolCall.Name)
				input.Hooks.EmitToolError(call, err)
				feedback, abort := input.interceptToolError(call, err, round, maxToolErrorLength)
				if abort {
					return nil, fmt.Errorf("tool %q failed: %w", toolCall.Name, err)
				}
				call.Result = feedback
				input.Hooks.EmitToolResult(call)
				output.ToolCalls = append(output.ToolCalls, call)
				output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, feedback))
				messages = append(messages, NewToolMessage(toolCall.ID, feedback))
				input.appendApprovalDecisionMessage(output, callOutput.Approval)
				continue
			}

			call.Result = *callOutput.Result
			input.Hooks.EmitToolResult(call)
			output.ToolCalls = append(output.ToolCalls, call)
			output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, *callOutput.Result))
			messages = append(messages, NewToolMessage(toolCall.ID, *callOutput.Result))
			input.appendApprovalDecisionMessage(output, callOutput.Approval)
		}
	}

	return nil, fmt.Errorf("exceeded %d tool call rounds", maxToolCallRounds)
}

func (input CompletionCallInput) appendApprovalDecisionMessage(output *CompletionCallOutput, approval *llmtool.ApprovalRecord) {
	if output == nil || approval == nil {
		return
	}
	if approval.Approved && !input.ApprovalDecisionMessages.AppendAccepted {
		return
	}
	if !approval.Approved && !input.ApprovalDecisionMessages.AppendRejected {
		return
	}

	status := "rejected"
	if approval.Approved {
		status = "accepted"
	}

	message := map[string]string{
		"tool_approval": status,
		"tool":          approval.ToolName,
		"risk":          string(approval.Policy.Risk),
	}
	if approval.Reason != "" {
		message["reason"] = approval.Reason
	}

	data, err := json.Marshal(message)
	if err != nil {
		return
	}
	output.Messages = append(output.Messages, NewUserMessage(string(data)))
}

func (input CompletionCallInput) interceptToolError(toolCall ToolCall, err error, round int, maxToolErrorLength int) (string, bool) {
	defaultFeedback := shortToolError(err, maxToolErrorLength)
	if input.ToolErrorInterceptor == nil {
		return defaultFeedback, false
	}

	decision := input.ToolErrorInterceptor(ToolErrorContext{
		ToolCall:        toolCall,
		Err:             err,
		Round:           round,
		DefaultFeedback: defaultFeedback,
	})
	if decision.Abort {
		return "", true
	}
	if decision.Feedback == "" {
		return defaultFeedback, false
	}
	return decision.Feedback, false
}

func (input CompletionCallInput) effectiveMaxToolCallRounds() int {
	if input.MaxToolCallRounds > 0 {
		return input.MaxToolCallRounds
	}
	return DefaultMaxToolCallRounds
}

func (input CompletionCallInput) effectiveMaxToolErrorLength() int {
	if input.MaxToolErrorLength > 0 {
		return input.MaxToolErrorLength
	}
	return DefaultMaxToolErrorLength
}

func (input CompletionCallInput) effectiveProviderErrorRetries() int {
	if input.ProviderErrorRetries != nil {
		return *input.ProviderErrorRetries
	}
	return DefaultProviderErrorRetries
}

func (output *CompletionCallOutput) recordUsage(hooks CompletionHooks, model string, round int, attempt int, usage *TokenUsage) {
	if usage == nil {
		return
	}
	event := UsageEvent{
		Model:   model,
		Round:   round,
		Attempt: attempt,
		Usage:   usage.clone(),
	}
	hooks.EmitUsage(event)
	output.Usage.add(event.Usage)
	output.UsageEvents = append(output.UsageEvents, event)
}

func completeWithProviderRetries(ctx context.Context, client ChatClient, request ChatRequest, hooks CompletionHooks, retries int, round int) (*ChatResponse, int, error) {
	var lastErr error
	for attempt := 0; attempt <= retries; attempt++ {
		start := GenerationStartEvent{
			Model:              request.Model,
			Round:              round,
			Attempt:            attempt,
			MessageCount:       len(request.Messages),
			AvailableToolCount: len(request.Tools),
		}
		hooks.EmitGenerationStart(start)

		response, err := client.Complete(ctx, request, hooks)
		if err == nil {
			end := GenerationEndEvent{
				Model:              request.Model,
				Round:              round,
				Attempt:            attempt,
				MessageCount:       len(request.Messages),
				AvailableToolCount: len(request.Tools),
				Usage:              responseUsage(response),
			}
			if response != nil {
				end.ToolCallCount = len(response.ToolCalls)
			}
			hooks.EmitGenerationEnd(end)
			return response, attempt, nil
		}
		hooks.EmitGenerationEnd(GenerationEndEvent{
			Model:              request.Model,
			Round:              round,
			Attempt:            attempt,
			MessageCount:       len(request.Messages),
			AvailableToolCount: len(request.Tools),
			Err:                err,
		})
		hooks.EmitCallError(err)
		lastErr = err
	}
	return nil, retries, lastErr
}

func responseUsage(response *ChatResponse) *TokenUsage {
	if response == nil || response.Usage == nil {
		return nil
	}
	usage := response.Usage.clone()
	return &usage
}

func shortToolError(err error, maxLength int) string {
	message := strings.TrimSpace(err.Error())
	if maxLength > 0 && len(message) > maxLength {
		message = message[:maxLength]
	}
	data, marshalErr := json.Marshal(map[string]string{"error": message})
	if marshalErr != nil {
		return `{"error":"tool failed"}`
	}
	return string(data)
}

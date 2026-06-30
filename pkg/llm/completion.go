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
	// ReasoningEffort constrains reasoning for models/providers that support it.
	ReasoningEffort ReasoningEffort
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
	if !input.ReasoningEffort.valid() {
		return fmt.Errorf("reasoning effort must be one of none, minimal, low, medium, high, xhigh")
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
			Model:           input.Model,
			Messages:        messages,
			Temperature:     input.Temperature,
			ReasoningEffort: input.ReasoningEffort,
			Tools:           tools,
		}
		response, attempt, err := completeWithProviderRetries(ctx, input.Client, request, input.Hooks, providerErrorRetries, round)
		if err != nil {
			if errors.Is(err, ErrCompletionEventAborted) {
				if response != nil {
					output.Content = response.Content
					output.addUsage(request.Model, round, attempt, response.Usage)
				}
				return output, fmt.Errorf("chat completion aborted: %w", err)
			}
			return nil, fmt.Errorf("chat completion failed: %w", err)
		}
		if response == nil {
			return nil, errors.New("chat completion returned nil response")
		}
		if err := output.recordUsage(input.Hooks, request.Model, round, attempt, response.Usage); err != nil {
			output.Content = response.Content
			return output, fmt.Errorf("chat completion aborted: %w", err)
		}
		if err := input.Hooks.EmitAssistantMessage(assistantMessageEvent(round, attempt, response)); err != nil {
			output.Content = response.Content
			return output, fmt.Errorf("chat completion aborted: %w", err)
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
			if err := input.Hooks.EmitToolCallEvent(round, call); err != nil {
				return output, fmt.Errorf("chat completion aborted: %w", err)
			}

			callOutput, err := input.Tools.CallWithInfo(ctx, toolCall.Name, toolCall.Arguments)
			if err != nil {
				if hookErr := input.Hooks.EmitToolErrorEvent(round, call, err); hookErr != nil {
					return output, fmt.Errorf("chat completion aborted: %w", hookErr)
				}
				feedback, abort := input.interceptToolError(call, err, round, maxToolErrorLength)
				if abort {
					return nil, fmt.Errorf("tool %q failed: %w", toolCall.Name, err)
				}
				call.Result = feedback
				if hookErr := input.Hooks.EmitToolResultEvent(round, call); hookErr != nil {
					return output, fmt.Errorf("chat completion aborted: %w", hookErr)
				}
				output.ToolCalls = append(output.ToolCalls, call)
				output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, feedback))
				messages = append(messages, NewToolMessage(toolCall.ID, feedback))
				input.appendApprovalDecisionMessage(output, callOutput.Approval)
				continue
			}
			if callOutput.Result == nil {
				err := fmt.Errorf("tool %q returned nil result", toolCall.Name)
				if hookErr := input.Hooks.EmitToolErrorEvent(round, call, err); hookErr != nil {
					return output, fmt.Errorf("chat completion aborted: %w", hookErr)
				}
				feedback, abort := input.interceptToolError(call, err, round, maxToolErrorLength)
				if abort {
					return nil, fmt.Errorf("tool %q failed: %w", toolCall.Name, err)
				}
				call.Result = feedback
				if hookErr := input.Hooks.EmitToolResultEvent(round, call); hookErr != nil {
					return output, fmt.Errorf("chat completion aborted: %w", hookErr)
				}
				output.ToolCalls = append(output.ToolCalls, call)
				output.Messages = append(output.Messages, NewToolMessage(toolCall.ID, feedback))
				messages = append(messages, NewToolMessage(toolCall.ID, feedback))
				input.appendApprovalDecisionMessage(output, callOutput.Approval)
				continue
			}

			call.Result = *callOutput.Result
			if err := input.Hooks.EmitToolResultEvent(round, call); err != nil {
				return output, fmt.Errorf("chat completion aborted: %w", err)
			}
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

func (output *CompletionCallOutput) recordUsage(hooks CompletionHooks, model string, round int, attempt int, usage *TokenUsage) error {
	event, ok := output.addUsage(model, round, attempt, usage)
	if !ok {
		return nil
	}
	return hooks.EmitUsageEvent(event)
}

func (output *CompletionCallOutput) addUsage(model string, round int, attempt int, usage *TokenUsage) (UsageEvent, bool) {
	if usage == nil {
		return UsageEvent{}, false
	}
	event := UsageEvent{
		Model:   model,
		Round:   round,
		Attempt: attempt,
		Usage:   usage.clone(),
	}
	output.Usage.add(event.Usage)
	output.UsageEvents = append(output.UsageEvents, event)
	return event, true
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
		if err := hooks.EmitGenerationStartEvent(start); err != nil {
			return nil, attempt, err
		}

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
			if err := hooks.EmitGenerationEndEvent(end); err != nil {
				return response, attempt, err
			}
			return response, attempt, nil
		}
		endErr := hooks.EmitGenerationEndEvent(GenerationEndEvent{
			Model:              request.Model,
			Round:              round,
			Attempt:            attempt,
			MessageCount:       len(request.Messages),
			AvailableToolCount: len(request.Tools),
			Err:                err,
		})
		if endErr != nil {
			return nil, attempt, endErr
		}
		if errors.Is(err, ErrCompletionEventAborted) {
			return nil, attempt, err
		}
		hooks.EmitCallError(err)
		lastErr = err
	}
	return nil, retries, lastErr
}

func assistantMessageEvent(round int, attempt int, response *ChatResponse) AssistantMessageEvent {
	event := AssistantMessageEvent{
		Round:   round,
		Attempt: attempt,
	}
	if response == nil {
		return event
	}
	event.Content = response.Content
	event.Reasoning = response.Reasoning
	event.ToolCalls = append([]ToolCall(nil), response.ToolCalls...)
	if response.Usage != nil {
		usage := response.Usage.clone()
		event.Usage = &usage
	}
	return event
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

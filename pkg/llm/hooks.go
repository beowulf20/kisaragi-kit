package llm

import "fmt"

// CompletionHooks contains optional callbacks for streaming, generation, and tool events.
type CompletionHooks struct {
	// OnContentDelta receives streamed assistant text chunks.
	OnContentDelta func(string)
	// OnReasoningDelta receives streamed assistant reasoning chunks when a provider exposes them.
	OnReasoningDelta func(string)
	// OnAssistantMessage runs after one provider response is returned and before requested tools run.
	OnAssistantMessage func(AssistantMessageEvent)
	// OnEvent receives typed lifecycle events and may abort the completion by returning an error.
	OnEvent func(Event) error
	// OnGenerationStart runs before one provider generation attempt starts.
	OnGenerationStart func(GenerationStartEvent)
	// OnGenerationEnd runs after one provider generation attempt ends.
	OnGenerationEnd func(GenerationEndEvent)
	// OnUsage runs after a provider generation reports token usage.
	OnUsage func(UsageEvent)
	// OnCallError runs after a provider chat call fails.
	OnCallError func(error)
	// OnToolCall runs before a requested tool is executed.
	OnToolCall func(ToolCall)
	// OnToolError runs after a tool returns an error or invalid result.
	OnToolError func(ToolCall, error)
	// OnToolResult runs after a tool returns or fails.
	OnToolResult func(ToolCall)
}

// ErrCompletionEventAborted identifies errors returned by abortable event hooks.
var ErrCompletionEventAborted = fmt.Errorf("completion event hook aborted")

// Event is a typed lifecycle event emitted by CompletionHooks.OnEvent.
type Event interface {
	completionEvent()
}

// EventAbortError wraps an error returned by CompletionHooks.OnEvent.
type EventAbortError struct {
	Event Event
	Err   error
}

// Error returns a human-readable abort message.
func (err *EventAbortError) Error() string {
	if err == nil {
		return "<nil>"
	}
	if err.Err == nil {
		return fmt.Sprintf("%v on %T", ErrCompletionEventAborted, err.Event)
	}
	return fmt.Sprintf("%v on %T: %v", ErrCompletionEventAborted, err.Event, err.Err)
}

// Unwrap returns the hook error that caused the abort.
func (err *EventAbortError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Err
}

// Is reports whether target is ErrCompletionEventAborted.
func (err *EventAbortError) Is(target error) bool {
	return target == ErrCompletionEventAborted
}

// GenerationStartEvent describes one provider generation attempt before it starts.
type GenerationStartEvent struct {
	Model              string
	Round              int
	Attempt            int
	MessageCount       int
	AvailableToolCount int
}

// GenerationEndEvent describes one provider generation attempt after it ends.
type GenerationEndEvent struct {
	Model              string
	Round              int
	Attempt            int
	MessageCount       int
	AvailableToolCount int
	ToolCallCount      int
	Usage              *TokenUsage
	Err                error
}

// UsageEvent describes token usage reported by one provider generation.
type UsageEvent struct {
	Model   string
	Round   int
	Attempt int
	Usage   TokenUsage
}

// AssistantMessageEvent describes one provider assistant response before tool execution.
type AssistantMessageEvent struct {
	Round     int
	Attempt   int
	Content   string
	Reasoning string
	ToolCalls []ToolCall
	Usage     *TokenUsage
}

// EventGenerationStart is emitted before one provider generation attempt starts.
type EventGenerationStart struct {
	GenerationStartEvent
}

func (EventGenerationStart) completionEvent() {}

// EventGenerationEnd is emitted after one provider generation attempt ends.
type EventGenerationEnd struct {
	GenerationEndEvent
}

func (EventGenerationEnd) completionEvent() {}

// EventContentDelta is emitted for streamed assistant text chunks.
type EventContentDelta struct {
	Delta string
}

func (EventContentDelta) completionEvent() {}

// EventReasoningDelta is emitted for streamed assistant reasoning chunks.
type EventReasoningDelta struct {
	Delta string
}

func (EventReasoningDelta) completionEvent() {}

// EventAssistantMessage is emitted after one provider response and before requested tools run.
type EventAssistantMessage struct {
	AssistantMessageEvent
}

func (EventAssistantMessage) completionEvent() {}

// EventToolCall is emitted before a requested tool is executed.
type EventToolCall struct {
	Round    int
	ToolCall ToolCall
}

func (EventToolCall) completionEvent() {}

// EventToolError is emitted after a tool returns an error or invalid result.
type EventToolError struct {
	Round    int
	ToolCall ToolCall
	Err      error
}

func (EventToolError) completionEvent() {}

// EventToolResult is emitted after a tool returns or fails.
type EventToolResult struct {
	Round    int
	ToolCall ToolCall
}

func (EventToolResult) completionEvent() {}

// EventUsage is emitted after a provider generation reports token usage.
type EventUsage struct {
	UsageEvent
}

func (EventUsage) completionEvent() {}

// ChainCompletionHooks returns hooks that invoke first and then second.
func ChainCompletionHooks(first, second CompletionHooks) CompletionHooks {
	return CompletionHooks{
		OnContentDelta: func(delta string) {
			if first.OnContentDelta != nil {
				first.OnContentDelta(delta)
			}
			if second.OnContentDelta != nil {
				second.OnContentDelta(delta)
			}
		},
		OnReasoningDelta: func(delta string) {
			if first.OnReasoningDelta != nil {
				first.OnReasoningDelta(delta)
			}
			if second.OnReasoningDelta != nil {
				second.OnReasoningDelta(delta)
			}
		},
		OnAssistantMessage: func(event AssistantMessageEvent) {
			if first.OnAssistantMessage != nil {
				first.OnAssistantMessage(event)
			}
			if second.OnAssistantMessage != nil {
				second.OnAssistantMessage(event)
			}
		},
		OnEvent: func(event Event) error {
			if err := first.EmitEvent(event); err != nil {
				return err
			}
			return second.EmitEvent(event)
		},
		OnGenerationStart: func(event GenerationStartEvent) {
			if first.OnGenerationStart != nil {
				first.OnGenerationStart(event)
			}
			if second.OnGenerationStart != nil {
				second.OnGenerationStart(event)
			}
		},
		OnGenerationEnd: func(event GenerationEndEvent) {
			if first.OnGenerationEnd != nil {
				first.OnGenerationEnd(event)
			}
			if second.OnGenerationEnd != nil {
				second.OnGenerationEnd(event)
			}
		},
		OnUsage: func(event UsageEvent) {
			if first.OnUsage != nil {
				first.OnUsage(event)
			}
			if second.OnUsage != nil {
				second.OnUsage(event)
			}
		},
		OnCallError: func(err error) {
			if first.OnCallError != nil {
				first.OnCallError(err)
			}
			if second.OnCallError != nil {
				second.OnCallError(err)
			}
		},
		OnToolCall: func(toolCall ToolCall) {
			if first.OnToolCall != nil {
				first.OnToolCall(toolCall)
			}
			if second.OnToolCall != nil {
				second.OnToolCall(toolCall)
			}
		},
		OnToolError: func(toolCall ToolCall, err error) {
			if first.OnToolError != nil {
				first.OnToolError(toolCall, err)
			}
			if second.OnToolError != nil {
				second.OnToolError(toolCall, err)
			}
		},
		OnToolResult: func(toolCall ToolCall) {
			if first.OnToolResult != nil {
				first.OnToolResult(toolCall)
			}
			if second.OnToolResult != nil {
				second.OnToolResult(toolCall)
			}
		},
	}
}

// EmitToolCall invokes OnToolCall when it is configured.
func (hooks CompletionHooks) EmitToolCall(toolCall ToolCall) {
	_ = hooks.EmitToolCallEvent(0, toolCall)
}

// EmitToolCallEvent invokes OnToolCall and emits a round-aware tool call event.
func (hooks CompletionHooks) EmitToolCallEvent(round int, toolCall ToolCall) error {
	if hooks.OnToolCall != nil {
		hooks.OnToolCall(toolCall)
	}
	return hooks.EmitEvent(EventToolCall{Round: round, ToolCall: toolCall})
}

// EmitContentDelta invokes OnContentDelta when it is configured.
func (hooks CompletionHooks) EmitContentDelta(delta string) {
	_ = hooks.EmitContentDeltaEvent(delta)
}

// EmitContentDeltaEvent invokes OnContentDelta and emits an abortable content delta event.
func (hooks CompletionHooks) EmitContentDeltaEvent(delta string) error {
	if hooks.OnContentDelta != nil {
		hooks.OnContentDelta(delta)
	}
	return hooks.EmitEvent(EventContentDelta{Delta: delta})
}

// EmitReasoningDelta invokes OnReasoningDelta when it is configured.
func (hooks CompletionHooks) EmitReasoningDelta(delta string) error {
	if hooks.OnReasoningDelta != nil {
		hooks.OnReasoningDelta(delta)
	}
	return hooks.EmitEvent(EventReasoningDelta{Delta: delta})
}

// EmitAssistantMessage invokes OnAssistantMessage when it is configured.
func (hooks CompletionHooks) EmitAssistantMessage(event AssistantMessageEvent) error {
	if hooks.OnAssistantMessage != nil {
		hooks.OnAssistantMessage(event)
	}
	return hooks.EmitEvent(EventAssistantMessage{AssistantMessageEvent: event})
}

// EmitEvent invokes OnEvent when it is configured.
func (hooks CompletionHooks) EmitEvent(event Event) error {
	if hooks.OnEvent == nil || event == nil {
		return nil
	}
	if err := hooks.OnEvent(event); err != nil {
		if _, ok := err.(*EventAbortError); ok {
			return err
		}
		return &EventAbortError{Event: event, Err: err}
	}
	return nil
}

// EmitGenerationStart invokes OnGenerationStart when it is configured.
func (hooks CompletionHooks) EmitGenerationStart(event GenerationStartEvent) {
	_ = hooks.EmitGenerationStartEvent(event)
}

// EmitGenerationStartEvent invokes OnGenerationStart and emits an abortable generation start event.
func (hooks CompletionHooks) EmitGenerationStartEvent(event GenerationStartEvent) error {
	if hooks.OnGenerationStart != nil {
		hooks.OnGenerationStart(event)
	}
	return hooks.EmitEvent(EventGenerationStart{GenerationStartEvent: event})
}

// EmitGenerationEnd invokes OnGenerationEnd when it is configured.
func (hooks CompletionHooks) EmitGenerationEnd(event GenerationEndEvent) {
	_ = hooks.EmitGenerationEndEvent(event)
}

// EmitGenerationEndEvent invokes OnGenerationEnd and emits an abortable generation end event.
func (hooks CompletionHooks) EmitGenerationEndEvent(event GenerationEndEvent) error {
	if hooks.OnGenerationEnd != nil {
		hooks.OnGenerationEnd(event)
	}
	return hooks.EmitEvent(EventGenerationEnd{GenerationEndEvent: event})
}

// EmitUsage invokes OnUsage when it is configured.
func (hooks CompletionHooks) EmitUsage(event UsageEvent) {
	_ = hooks.EmitUsageEvent(event)
}

// EmitUsageEvent invokes OnUsage and emits an abortable usage event.
func (hooks CompletionHooks) EmitUsageEvent(event UsageEvent) error {
	if hooks.OnUsage != nil {
		hooks.OnUsage(event)
	}
	return hooks.EmitEvent(EventUsage{UsageEvent: event})
}

// EmitCallError invokes OnCallError when it is configured.
func (hooks CompletionHooks) EmitCallError(err error) {
	if hooks.OnCallError != nil {
		hooks.OnCallError(err)
	}
}

// EmitToolError invokes OnToolError when it is configured.
func (hooks CompletionHooks) EmitToolError(toolCall ToolCall, err error) {
	_ = hooks.EmitToolErrorEvent(0, toolCall, err)
}

// EmitToolErrorEvent invokes OnToolError and emits a round-aware tool error event.
func (hooks CompletionHooks) EmitToolErrorEvent(round int, toolCall ToolCall, err error) error {
	if hooks.OnToolError != nil {
		hooks.OnToolError(toolCall, err)
	}
	return hooks.EmitEvent(EventToolError{Round: round, ToolCall: toolCall, Err: err})
}

// EmitToolResult invokes OnToolResult when it is configured.
func (hooks CompletionHooks) EmitToolResult(toolCall ToolCall) {
	_ = hooks.EmitToolResultEvent(0, toolCall)
}

// EmitToolResultEvent invokes OnToolResult and emits a round-aware tool result event.
func (hooks CompletionHooks) EmitToolResultEvent(round int, toolCall ToolCall) error {
	if hooks.OnToolResult != nil {
		hooks.OnToolResult(toolCall)
	}
	return hooks.EmitEvent(EventToolResult{Round: round, ToolCall: toolCall})
}

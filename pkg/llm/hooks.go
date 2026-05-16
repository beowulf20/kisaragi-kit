package llm

// CompletionHooks contains optional callbacks for streaming, generation, and tool events.
type CompletionHooks struct {
	// OnContentDelta receives streamed assistant text chunks.
	OnContentDelta func(string)
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

// ChainCompletionHooks returns hooks that invoke first and then second.
func ChainCompletionHooks(first, second CompletionHooks) CompletionHooks {
	return CompletionHooks{
		OnContentDelta: func(delta string) {
			first.EmitContentDelta(delta)
			second.EmitContentDelta(delta)
		},
		OnGenerationStart: func(event GenerationStartEvent) {
			first.EmitGenerationStart(event)
			second.EmitGenerationStart(event)
		},
		OnGenerationEnd: func(event GenerationEndEvent) {
			first.EmitGenerationEnd(event)
			second.EmitGenerationEnd(event)
		},
		OnUsage: func(event UsageEvent) {
			first.EmitUsage(event)
			second.EmitUsage(event)
		},
		OnCallError: func(err error) {
			first.EmitCallError(err)
			second.EmitCallError(err)
		},
		OnToolCall: func(toolCall ToolCall) {
			first.EmitToolCall(toolCall)
			second.EmitToolCall(toolCall)
		},
		OnToolError: func(toolCall ToolCall, err error) {
			first.EmitToolError(toolCall, err)
			second.EmitToolError(toolCall, err)
		},
		OnToolResult: func(toolCall ToolCall) {
			first.EmitToolResult(toolCall)
			second.EmitToolResult(toolCall)
		},
	}
}

// EmitToolCall invokes OnToolCall when it is configured.
func (hooks CompletionHooks) EmitToolCall(toolCall ToolCall) {
	if hooks.OnToolCall != nil {
		hooks.OnToolCall(toolCall)
	}
}

// EmitContentDelta invokes OnContentDelta when it is configured.
func (hooks CompletionHooks) EmitContentDelta(delta string) {
	if hooks.OnContentDelta != nil {
		hooks.OnContentDelta(delta)
	}
}

// EmitGenerationStart invokes OnGenerationStart when it is configured.
func (hooks CompletionHooks) EmitGenerationStart(event GenerationStartEvent) {
	if hooks.OnGenerationStart != nil {
		hooks.OnGenerationStart(event)
	}
}

// EmitGenerationEnd invokes OnGenerationEnd when it is configured.
func (hooks CompletionHooks) EmitGenerationEnd(event GenerationEndEvent) {
	if hooks.OnGenerationEnd != nil {
		hooks.OnGenerationEnd(event)
	}
}

// EmitUsage invokes OnUsage when it is configured.
func (hooks CompletionHooks) EmitUsage(event UsageEvent) {
	if hooks.OnUsage != nil {
		hooks.OnUsage(event)
	}
}

// EmitCallError invokes OnCallError when it is configured.
func (hooks CompletionHooks) EmitCallError(err error) {
	if hooks.OnCallError != nil {
		hooks.OnCallError(err)
	}
}

// EmitToolError invokes OnToolError when it is configured.
func (hooks CompletionHooks) EmitToolError(toolCall ToolCall, err error) {
	if hooks.OnToolError != nil {
		hooks.OnToolError(toolCall, err)
	}
}

// EmitToolResult invokes OnToolResult when it is configured.
func (hooks CompletionHooks) EmitToolResult(toolCall ToolCall) {
	if hooks.OnToolResult != nil {
		hooks.OnToolResult(toolCall)
	}
}
